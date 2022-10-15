# import os
# # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


from pathlib import Path
from nltk.corpus.reader.chasen import test
from torch.utils.data import Dataset
import os.path as op
import torch
import numpy as np
import json
import random
from collections import OrderedDict
from transformers import BertTokenizer
from copy import deepcopy
from pathlib import Path
import pickle
import os


import json, random, time, os, base64
from pprint import pprint
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# from KMIS.VLM.main import test
np.set_printoptions(precision=4)
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
from multiprocessing import Pool
import shutil
from sentence_transformers import SentenceTransformer
import string
import re
# from nltk.corpus import stopwords
# stop = set(stopwords.words("english"))  
from IPython.display import display


from numpy.core.numeric import count_nonzero
import torch
from html import unescape

from torch.utils import data
from torch.utils.data.dataset import Dataset
# from data_loaders.data_loader import VLDataset
from transformers import BertTokenizer

from utils import set_seed, mkdir, setup_logger, load_config_file, synchronize
from model.VLM import VLM

from omegaconf import OmegaConf
from torch.optim.lr_scheduler import StepLR
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

RANDOM_SEED = 500
np.random.seed(RANDOM_SEED)
torch.random.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('/nlsasfs/home/nltmocr/abhirama/bert_tokenizer/')


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

data_config_path = 'config/config_itm/data_config.yaml'
model_config_path = 'config/config_itm/model_config.yaml'
train_config_path = 'config/config_itm/train_config.yaml'
test_config_path = 'config/test_config.yaml'

data_config = load_config_file(data_config_path)
train_config = load_config_file(train_config_path)
model_config = load_config_file(model_config_path)
test_config = load_config_file(test_config_path)


parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, help="path to queries", default='brand')
args = parser.parse_args()

config = OmegaConf.merge(train_config, data_config, model_config, test_config)

# merging cli arguments
config = OmegaConf.merge(config, OmegaConf.create(vars(args)))

data_root = 'cofar_' + test_config.category

cap2img = {}
captions_file = 'data/' + data_root + '/cofar_queries_1K.json'
img_cap_dict = read_json(captions_file)
cap2gt = {}
for key, captions in img_cap_dict.items():
    cap2gt[key] = "_".join(key.split('_')[:-1])
        

class COFARDataset(Dataset):

    def __init__(self, config, tokenizer=None, split='val', is_train=False):
        super(COFARDataset).__init__()

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True) # saves bert files locally
        else:
            # self.tokenizer = tokenizer
            tokenizer = BertTokenizer.from_pretrained('/nlsasfs/home/nltmocr/abhirama/bert_tokenizer/')
        # train flag
        self.is_train = is_train
        self.dir_split = 'train' if split in ['train', 'val'] else 'test'
    

        # file paths

        data_root = 'cofar_' + test_config.category

        self.captions_file = 'data/' + data_root + '/cofar_queries_1K.json' # change it to 5K when running evaluation for 5K gallery size
        self.img_features_path = 'data/' + data_root + '/train_obj_frcn_features'
        self.image_set = 'data/' + data_root + '/cofar_gallery_ids_1K.json' # change it to 5K when running evaluation for 5K gallery size

        if test_config.use_oracle_knowledge:
            self.knowledge = 'data/' + data_root + '/KB_oracle.json'
        elif test_config.use_wikified_knowledge:
            self.knowledge = 'data/' + data_root + '/KB_wikified.json'
        
            
        # max lens
        self.max_seq_len = config.max_seq_len
        self.max_vis_seq_len = config.max_vis_seq_len

        # mask probability
        # TODO: replace masked tokens with random token
        self.mask_probability = config.mask_probability


        self.config = config

        if is_train:
            self.num_captions_per_img = 1
        else:
            self.num_captions_per_img = 1
        

        self.images = []
        self.captions = []
        self.capIds = []
        self.labels = []

        img_cap_dict = read_json(self.captions_file)
        gallery_json = json.load(open(self.image_set))
        knowledge = json.load(open(self.knowledge))
        
        caption_id = 0

        for key, value in img_cap_dict.items():

            for each_image_id in gallery_json:

                    if test_config.use_cofar_knowledge:
                
                        if test_config.use_oracle_knowledge:
                            img_id = each_image_id.split('_')[0]           
                        else:
                            img_id = each_image_id        
                        
                        if img_id in knowledge.keys():
   
                            self.captions.append(value + '. ' + knowledge[img_id])
   
                        else:
                            self.captions.append(value)
                    else:
                        self.captions.append(value)

                    
                    self.images.append(each_image_id)
                    self.capIds.append(key)

                    if each_image_id == key:
                        self.labels.append([1])

                    else:
                        self.labels.append([0])

        print(len(self.captions))

    def get_img_features_and_bbox(self, img_key):

        features_path = op.join(self.img_features_path, img_key + '.npy')
        feature_info_path = op.join(self.img_features_path, img_key + '_info.npy')
        img_feature = torch.tensor(np.load(features_path))
        img_info = np.load(feature_info_path, allow_pickle=True).item()
        img_bbox = torch.tensor(img_info['bbox'])

        # normalizing the bbox (not normalized)
        if (img_bbox > 1).any():
            img_width = img_info['image_width']
            img_height = img_info['image_height']
            img_bbox = img_bbox * torch.tensor([1.0/img_width, 1.0/img_height, 1.0/img_width, 1.0/img_height])
        

        # removing roi features of background (class = 0)
        img_feature = img_feature[[img_info['objects'] != 0]]
        img_bbox = img_bbox[[img_info['objects'] != 0]]

        return img_feature, img_bbox



    def get_txt_mask(self, input_ids, probability):

        # Function to mask some input tokens with a probability of 0.15

        rand = torch.rand(input_ids.shape)

        # create mask array 101 - start token and 102 - end token
        mask_arr = ((rand < probability) * (input_ids != 101) * (input_ids != 102) * (input_ids != 0)).int()

        return mask_arr


    def tensorize_example(self, 
                         caption,
                         img_feature,
                         img_bbox=None,
                         ):

        text_input = self.tokenizer(caption, return_tensors='pt', max_length=self.max_seq_len, truncation=True, padding='max_length')
        text_ids = text_input.input_ids
        text_mask = text_input.attention_mask
        mlm_mask = self.get_txt_mask(text_ids, 0.0)
        seq_len = text_mask.sum(dim=1) # seq len of tokens 


        text_token_type_ids = torch.zeros(self.config.max_seq_len, dtype=torch.long)
        vis_token_type_ids = torch.ones(self.config.max_vis_seq_len, dtype=torch.long)
        

        target_caption_ids = deepcopy(text_ids)

        # for MLM we are interested to calculate loss for only masked tokens
        # therefore in label targets corresponding to not masked tokens will be set to -1
        # and CrossEntropy Loss is set to ignore -1 token
        target_caption_ids = torch.where(
            mlm_mask > 0,
            target_caption_ids,
            torch.ones_like(target_caption_ids) * (-1)
        )

        # get no of object features
        # pad with zeros if less than max_vis_seq_len
        # else truncate to max_vis_seq_len
        img_feature_len = img_feature.shape[0]
        vis_mask = torch.ones_like(img_feature)

        if img_feature_len > self.max_vis_seq_len:
            img_feature = img_feature[0 : self.max_vis_seq_len, :]
            img_bbox = img_bbox[0 : self.max_vis_seq_len, :]
            img_padding_len = 0
            vis_mask = torch.ones((1, self.max_seq_len))
            
        else:
            img_padding_len = self.max_vis_seq_len - img_feature_len
            img__padding_matrix = torch.zeros((img_padding_len, img_feature.shape[1]))
            bbox_padding_matrix = torch.zeros((img_padding_len, img_bbox.shape[1]))
            img_feature = torch.cat([img_feature, img__padding_matrix], dim=0)
            img_bbox = torch.cat([img_bbox, bbox_padding_matrix], dim=0)
            vis_mask_1 = torch.ones((1,img_feature_len))
            vis_mask_2 = torch.zeros((1,img_padding_len))
            vis_mask = torch.cat([vis_mask_1, vis_mask_2], dim=1)

        # return (text_ids, seq_len, text_mask, mlm_mask, target_caption_ids, torch.zeros_like(img_feature), torch.zeros_like(img_bbox), img_feature_len, torch.zeros_like(vis_mask), text_token_type_ids, vis_token_type_ids)
        return (text_ids, seq_len, text_mask, mlm_mask, target_caption_ids, img_feature, img_bbox, img_feature_len, vis_mask, text_token_type_ids, vis_token_type_ids)

    
    def __getitem__(self, index: int):
        
        # if self.is_train:

            img_key = self.images[index]
            

            img_feature, img_bbox = self.get_img_features_and_bbox(img_key)

            assert len(img_feature) == len(img_bbox)

            caption = self.captions[index]
            example = self.tensorize_example(caption, img_feature, img_bbox)

            example_pair = tuple(list(example) + self.labels[index])

            return index, example_pair, self.capIds[index], self.images[index]
        


    def __len__(self):
        # if not self.is_train and self.config['cross_image_eval']:
        #     return len(self.img_keys) ** 2 * self.num_captions_per_img
        return len(self.images)



def calculate_retrieval_score(results_dict, result_dict_2=None):
    
    count = 0
    
    h_s = [0]*19
    rankList = []
    h_1 = 0
    h_3 = 0
    h_10 = 0
    h_2 = 0
    h_5 = 0
    h_15 = 0
    h_20 = 0
    h_50 = 0
    precision = 0
    recall = 0
    c = 0
    avg_candidates = 0
    
    processed_capIds = len(list(results_dict.keys()))
    # print(i)
    
    print(processed_capIds)
    posIds_count = 0

    for index, (key, value) in enumerate(results_dict.items()):
        
        c = 0
        if test_config.cofar_brand:
                posIds = cap2gt[key]
                ordered_imgIds = [y[0] for y in sorted(value['imgId2scores'], key = lambda x:x[-1], reverse=True)]
                count += 1
        else:
            posIds = cap2gt[key]
            ordered_imgIds = [y[0] for y in sorted(value['imgId2scores'], key = lambda x:x[-1], reverse=True)]
            count += 1

        posIds_count += len(posIds)
        for eachPosId in [posIds]:

            if eachPosId in ordered_imgIds:
            
                idx = ordered_imgIds.index(str(eachPosId))+1

                if idx == 1:
                    h_1 += 1
                                    
                if idx <= 2:
                    h_2 += 1
                
                if idx <= 3:
                    h_3 += 1
                
                if idx <= 5:
                    h_5 += 1

                if idx <= 10:
                    h_10 += 1

                if test_config.cofar:

                    if idx <= 15:
                        h_15 += 1
        
                    if idx <= 20:
                        h_20 += 1

                    if idx <= 50:
                        h_50 += 1
        
                rankList.append(idx)

                for x in range(len(h_s)):
                    
                    if idx <= x:
                        h_s[x] += 1


        posIds = []
        ordered_imgIds = []

    medianRank = np.median(rankList)
    meanRank = np.mean(rankList)
    # print(index, count)


    # print(f"avg candidates : {avg_candidates/index}")
    print(f"Count: {count}")
    h_s = [x/count for x in h_s]

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.75,0.75]) # axis starts at 0.1, 0.1
    ax.bar([x+1 for x in range(len(h_s))],h_s)
    plt.xticks(range(0,len(h_s)))
    ax.set_xlabel('k')
    ax.set_ylabel('Hits @k')
    ax.legend()
    
    # if test_config.plot_hits:
    #     fig.savefig(os.path.join(test_config.path_to_plots , test_config.plot_name))

    return [x/count for x in h_s]


if test_config.run_results_from_pkl:

    a_file = open(test_config.path_to_results_pickle, 'rb')
    result_dict = pickle.load(a_file)
    calculate_retrieval_score(result_dict)

elif test_config.run_results_from_model:
    
    config.checkpoint_path = test_config.test_checkpoint_path
    logger = setup_logger(test_config.logger_name, test_config.log_dir, 0)
    
    mkdir(path=test_config.log_dir)

    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = VLM(model_config).cuda()
    logger.info(f"i value {0}, checkpoint_path - {config.checkpoint_path}")
    
    checkpoint = torch.load(config.checkpoint_path)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])

    softmax = torch.nn.Softmax(dim=1)
    model.eval()

    result_dict = {}
    start = time.time()


    dataset_to_model = COFARDataset(data_config)
    sampler = SequentialSampler(dataset_to_model)
    dataloader = DataLoader(dataset_to_model, sampler = sampler, batch_size = test_config.batch_size)

    itm_loss = 0
    mlm_loss = 0
    print(f"Image-Caption Pairs {len(dataloader.dataset)}")
    for step, (_, batch, qIds, imgIds) in enumerate(dataloader):

        text_ids, seq_len, text_mask, mlm_mask, target_caption_ids, img_feature, img_bbox, img_feature_len, vis_mask, txt_token_type_ids, vis_token_type_ids, itm_label = batch[:12]
        
        inputs_to_model = {
            'text_ids' : text_ids,
            'text_mask': text_mask,
            'mlm_mask' : mlm_mask,
            'target_caption_ids' : target_caption_ids,
            'region_features' : img_feature,
            'region_loc' : img_bbox,
            'vis_mask' : vis_mask,
            'txt_token_type_ids': txt_token_type_ids,
            'vis_token_type_ids': vis_token_type_ids,
            'itm_label' : torch.tensor(itm_label),
        }
        
        with torch.no_grad():
            fwd_results = model(inputs_to_model)
            # print(fwd_results['ITM_Loss'])
            itm_loss += fwd_results['ITM_Loss'].mean().item()
            mlm_loss += fwd_results['MLM_Loss'].mean().item()
            alignment_score = softmax(fwd_results['logits']).select(1,1).cpu().detach().numpy()
            #print(qIds, imgIds, alignment_score)
            
            for each_qId in qIds:
                if each_qId not in result_dict.keys():
                    result_dict[each_qId] = {}
                    result_dict[each_qId]['imgId2scores'] = []

            for index, each_imgId in enumerate(imgIds):
                result_dict[qIds[index]]['imgId2scores'].append([each_imgId, alignment_score[index]])
        
        end = time.time()
        # print(end)
        if (step+1) % 100 == 0:
            print(f"{(step+1)*test_config.batch_size} Ques-Image pairs took {end-start:.5f} secs time, processed {len(result_dict.keys())}")
            start = time.time()
            calculate_retrieval_score(result_dict)
            if test_config.save_to_pickle:
                a_file = open(os.path.join(test_config.output_dir_pickle, test_config.pickle_file_name), "wb")
                pickle.dump(result_dict, a_file)
                a_file.close()

        # if step 

    print(f"Average ITM Loss : {itm_loss/len(dataloader.dataset)}")
    print(f"Average MLM Loss : {mlm_loss/len(dataloader.dataset)}")

    if test_config.save_to_pickle:
        a_file = open(os.path.join(test_config.output_dir_pickle, test_config.pickle_file_name), "wb")
        pickle.dump(result_dict, a_file)
        a_file.close()

    calculate_retrieval_score(result_dict)

