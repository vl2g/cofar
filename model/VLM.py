import logging
import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
import math

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertPooler,
    BertOnlyMLMHead,
    BertEncoder,
    BertModel,
    BertPreTrainedModel,
    BertPredictionHeadTransform,
)

# BertLayerNorm is removed - https://github.com/huggingface/transformers/issues/10892
# fix
BertLayerNorm = torch.nn.LayerNorm


# from transformers.modeling_bert import BertLayerNorm
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

class VLM(nn.Module):

    CLS_TOKEN_IDX = 0
    SEP_TOKEN_IDX = 1

    def __init__(self, config):
        super(VLM, self).__init__()

        self.finetune_modules = []

        # config 
        self.config = config
        self.vlm_config = BertConfig(**self.config.VLMConfig)
        
        self._is_pretrained = False
        self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.LayerNorm = BertLayerNorm(self.vlm_config.hidden_size, eps=self.vlm_config.layer_norm_eps)

        self.specialTokenEmb = nn.Embedding(num_embeddings=2, embedding_dim=self.vlm_config.hidden_size)

        # dimensions
        self.text_dim = self.vlm_config.hidden_size
        self.vis_dim = config.vis_dim
        self.loc_dim = config.loc_dim

        # loss - currently set to defaults - with an option to customize at later point
        # self.mlm_loss = config.mlm_loss
        # self.itm_loss = config.itm_loss
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.itm_loss = nn.CrossEntropyLoss()


        # # token embeddings 
        # self.text_token_type_ids = torch.zeros(self.config.max_position_embeddings, dtype=torch.long,device=config.device)
        # self.vis_token_type_ids = torch.ones(self.config.max_visual_embeddings, dtype=torch.long,device=config.device)
        self.new_token_type_embeddings = nn.Embedding(config.token_vocab_size, self.vlm_config.hidden_size)
        # self.new_token_type_embeddings.to(torch.device(config.device))

        # print(self.text_token_type_ids.get_device(), self.vis_token_type_ids.get_device())

        # self.text_token_embedding = self.new_token_type_embeddings(self.text_token_type_ids)
        # self.vis_token_embedding = self.new_token_type_embeddings(self.vis_token_type_ids)

        # torch modules - "TEXT"
        # self.text_emb = TextEmbeddings(self.vlm_config, self.new_token_type_embeddings)
        self.text_bert_config = BertConfig()

        if self.config.text_bert_config.init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                "bert-base-uncased", config=self.text_bert_config
            )

            self.finetune_modules.append(
                {"module": self.text_bert, "lr_scale": self.config.lr_scale_text_bert}
            )


        # torch modules - "IMAGE"
        self.visual_emb = VisualEmbeddings(self.vlm_config, self.vis_dim, self.loc_dim, self.new_token_type_embeddings)

        self.encoder = BertEncoder(self.vlm_config)
        self.pooler = BertPooler(self.vlm_config)
        self.heads = VLMPretrainingHead(self.vlm_config)

        # init weights
        self.encoder.apply(self._init_weights)
        self.pooler.apply(self._init_weights)
        self.heads.apply(self._init_weights)

        # tie word embeddings in encoder and decoder!
        self._tie_weights()



    def _tie_weights(self):

        self.heads.predictions.decoder.weight = self.text_bert.embeddings.word_embeddings.weight
    
        

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.vlm_config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def get_optimizer_parameters(self, config=None):
        optimizer_param_groups = []

        if config is not None :
            base_lr = config.optimizer.params.lr
        else :
            base_lr = 0.0001
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

        
    
    def forward(self, inputs, valFlag=False):
        
        #TODO: caption embeddings initialized with a pretrained BERT
        # raw text ids [b * max_seq_length]
        text_ids = inputs['text_ids'].squeeze(1)

        # text attention mask - ideally ignore pad indices. [b * max_seq_length]
        text_mask = inputs['text_mask'].squeeze(1)
        # print(text_mask.shape)

        # token embs
        text_token_ids = inputs['txt_token_type_ids']
        vis_token_ids = inputs['vis_token_type_ids']

        if self.config.use_mlm and not valFlag:
            # mlm mask - (0,1) binary mask of [b * max_seq_len] 15% tokens in input caption
            # 0 for masked tokens, 1 for not masked ones

            mlm_mask = inputs['mlm_mask'].squeeze(1)
            # mask the text mask on mlm_mask labels


            text_mask = torch.where(
                                mlm_mask > 0,
                                torch.zeros_like(text_mask) * (0),
                                text_mask
                                )

        # MLM target caption ids - ideally a copy of text_ids (with out any masking)
        # [b * max_seq_len]
        target_caption_ids = inputs['target_caption_ids']

        # visual features - faster r-cnn used here 
        #TODO: use patch wise image blocks as inputs
        # [b*max_visual_seq_len*2048]
        region_features = inputs['region_features']

        # location feature - bbox as position encoder for images (refer VisualEmbeddings)
        region_location = inputs['region_loc']

        # modify the following mask if doing masked region prediction and 
        # also add region target - either (i) soft labels as given by backbone detector
        # or (ii) reconstruction loss
        region_attention_mask = inputs['vis_mask'].squeeze(1)

        # itm label [b * 1]
        image_caption_match_label = inputs['itm_label']

        # conver to float32
        text_mask = text_mask.to(torch.float32)
        region_attention_mask = region_attention_mask.to(torch.float32)
        # mlm_mask = mlm_mask.to(torch.float32)

        # print(f"text_id.shape {text_ids.shape}, obj_emb.shape {region_features.shape} , txt_maks.shape {text_mask.shape}, MLM mask shape {mlm_mask.shape}, vis_mask {region_attention_mask.shape}")

        # no. of words in caption        
        num_words = text_mask.sum(dim=1)

        # print(text_mask.shape)
        batch_size, max_num_words = text_mask.shape

        # get text embedding [b*max_seq_len*vlm_hidden_size]
        # text_embedding = self.text_emb(text_ids, token_type_ids=text_token_ids)
        text_embedding = self.text_bert(text_ids, text_mask)

        # get image embedding [b*max_vis_seq_len*vlm_hidden_size]
        image_embedding = self.visual_emb(region_features, region_location, vis_token_type_ids=vis_token_ids)


        special_token_embeddings = self.specialTokenEmb(torch.LongTensor([VLM.CLS_TOKEN_IDX, VLM.SEP_TOKEN_IDX]).to(text_embedding.device))
        cls_token_emb = special_token_embeddings[VLM.CLS_TOKEN_IDX]
        sep_token_emb = special_token_embeddings[VLM.SEP_TOKEN_IDX]

        # making it broadcastable
        cls_token_embs = cls_token_emb.unsqueeze(0).expand(text_ids.size(0), -1).unsqueeze(1)
        cls_mask = torch.ones([text_ids.size(0),1]).to(text_mask.device)

        sep_token_embs = sep_token_emb.unsqueeze(0).expand(text_ids.size(0), -1).unsqueeze(1)
        sep_mask = torch.ones([text_ids.size(0),1]).to(text_mask.device)

        # print(cls_mask, text_mask, region_attention_mask)

        # concat text and image embeddings [b * (max_seq_len + max_vis_seq_len) * vlm_hidden_size]
        # print(text_embedding.shape, image_embedding.shape, text_mask.shape, region_attention_mask.shape)
        embedded_tokens = torch.cat([cls_token_embs, text_embedding, sep_token_embs, image_embedding], dim=1)

        # concat text and image attention masks [b * (max_seq_len + max_vis_seq_len) * vlm_hidden_size]
        attention_mask = torch.cat([cls_mask, text_mask, sep_mask, region_attention_mask], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
     
        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        # head_mask = [None] * self.config.num_hidden_layers


        # input to VLM encoder!
        # also check if we need to set attention values from {0,1} to {-10000, 0}
        sequence_output = self.encoder(
                                        embedded_tokens,
                                        attention_mask[:,None,None,:], # equivalent to unsqueeze(0).unsqueeze(1)
                                        head_mask = [None] * self.vlm_config.num_hidden_layers,
                                        output_attentions = True,
                                        output_hidden_states = False,
                                        )
        
        # get pooled output for all text and visual embeddings (last hidden layer)
        pooled_output = self.pooler(sequence_output[0])

        # select the only text sequenc embeddings (last hidden layer)
        text_sequence_output = sequence_output[0][:,1:max_num_words+1]

        # get prediction scores for mlm and itm
        text_prediction_scores, seq_relationship_score = self.heads(text_sequence_output, pooled_output)
        
        # print(seq_relationship_score)
        # calculate mlm loss
        # print(text_prediction_scores.shape, target_caption_ids.shape)

        if self.config.use_mlm:
            mlm_loss_calculated = self.mlm_loss(
                text_prediction_scores.view(-1, self.vlm_config.vocab_size),
                target_caption_ids.view(-1),
            )
        else:
            mlm_loss_calculated = torch.tensor(0.0).cuda()

        # calculate itm loss
        if self.config.use_itm:
            # print(seq_relationship_score.shape, image_caption_match_label.shape)
            itm_loss_calculated = self.itm_loss(
                seq_relationship_score.view(-1, 2),
                image_caption_match_label.view(-1)
            )
            # print(itm_loss_calculated)

        else:
            itm_loss_calculated = torch.tensor(0.0).cuda()

        total_loss = mlm_loss_calculated + itm_loss_calculated

        attentions = sequence_output[1]


        results = {
                    "total_loss" : total_loss,
                    "MLM_Loss" : mlm_loss_calculated,
                    "ITM_Loss" : itm_loss_calculated,
                    "cls_output_emb" : pooled_output,
                    "attentions" : attentions,
                    "logits" : seq_relationship_score
                   }

        return results


def create_position_ids_from_input_ids(input_ids, padding_idx):
    
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.
    :param torch.Tensor x:
    :return torch.Tensor:
    """
    
    """
    to test working uncomment prints and run
    print(create_position_ids_from_input_ids(torch.Tensor([[-1,0,1,2,32,33, -1]]), -1))

    """

    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()

    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask

    return incremental_indices.long() + padding_idx



class TextEmbeddings(nn.Module):

    def __init__(self, config, new_token_type_embeddings):
        super(TextEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.token_type_embeddings = new_token_type_embeddings

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):

        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids,0)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

        

class VisualEmbeddings(nn.Module):

    def __init__(self, config, vis_dim, loc_dim, new_token_type_embeddings):
        super(VisualEmbeddings, self).__init__()

        self.config = config
        self.token_type_embeddings = new_token_type_embeddings
        
        self.img_linear = nn.Linear(vis_dim, 
                                    config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.pos_layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        """
        #TODO:7 dim position vector for image feature (faster R-CNN)
        [x1, y1, x2, y2, w, h, w ∗ h], which denotes the normalized
        top left coordinates, bottom right coordinates, width, height,
        and the area of the detected region box.
        """
        # currently restricting to 4 (x1, y1, x2, y2)

        self.pos_linear = nn.Linear(loc_dim, self.config.hidden_size) 
        # self.mask_embedding = nn.Embedding(2, self.config.img_dim, padding_idx=0)

        self.visual_dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, visual_input_features, visual_position_features, vis_token_type_ids=None):

        # token_type_ids = torch.ones(visual_input_features.size(), dtype=torch.long, device=visual_input_features.device)

        img_embeddings = self.img_linear(visual_input_features)
        img_pos_embeddings = self.pos_linear(visual_position_features)

        visual_embeddings = self.LayerNorm(img_embeddings
                                        + img_pos_embeddings
                                        + self.token_type_embeddings(vis_token_type_ids))

        visual_embeddings = self.visual_dropout(visual_embeddings)

        return visual_embeddings


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size,
                                config.vocab_size,
                                bias=False
                                )

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class VLMPretrainingHead(nn.Module):

    def __init__(self, config):
        super(VLMPretrainingHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Sequential(
                                            nn.Linear(config.hidden_size, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 2)
                                            )

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score



class TextBert(BertPreTrainedModel): 

    """
    Model as per the documentation https://huggingface.co/transformers/v2.0.0/_modules/transformers/modeling_bert.html 
    """

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class TextBert(BertPreTrainedModel): 

    """
    Model as per the documentation https://huggingface.co/transformers/v2.0.0/_modules/transformers/modeling_bert.html 
    """

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output



# if __name__ == "__main__":
#     model = VLM({'VLMConfig'})
#     print(model)