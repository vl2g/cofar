from numpy.core.numeric import count_nonzero
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pickle

from torch.utils import data
from torch.utils.data.dataset import Dataset
from data_loaders.data_loader import VLDataset, VL_TestDataset
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
import time
import random


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.random.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_dataloader(config, dataset, is_train = True):
    
    if is_train:
        if config.distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
            batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
        else:
            sampler = RandomSampler(dataset)
            batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    else:
        sampler = SequentialSampler(dataset)
        batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)

    dataloader = DataLoader(dataset, sampler=sampler, 
            batch_size=batch_size, num_workers=config.num_workers)

    return dataloader


# function to train!

def train(config, train_dataset, model):

    # set batch size for e.g 128 per 1 gpu, 256 for 2 and 384 for 3 so on!
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    
    # dataloader
    train_dataloader = get_dataloader(config, train_dataset, is_train=True)

    # slightly confusing - check later
    if config.max_steps > 0:
        t_total = config.max_steps
        config.num_train_epochs = config.max_steps // (len(train_dataloader) // config.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs


    # optimizer call
    grouped_parameters = model.get_optimizer_parameters()
    optimizer = AdamW(grouped_parameters, lr=config.optimizer.params.lr, eps=config.optimizer.params.eps)

    if config.distributed: 
        # initialize distributed data parallel (DDP)
        model = DDP(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            find_unused_parameters=True,
        )

    if not config.distributed:
        model = torch.nn.DataParallel(model)
        model = model.to(config.device)

    if config.load_from_checkpoint:

        # print(config)
        
        checkpoint_path = config.checkpoint_path
        print(f"Checkpoint Path {checkpoint_path}")
        assert checkpoint_path is not None
        assert os.path.isfile(checkpoint_path)
        logger.info("Checkpoint used: %s", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        # model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Preparing Scheduler
    if config.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=config.warmup_steps)
    elif config.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)
    elif config.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    else:
        raise ValueError("Unknown scheduler type: {}".format(config.scheduler))

    # Data Parallel!
    # if config.n_gpu > 1 and not config.distributed:
    #     model = torch.nn.DataParallel(model)


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   config.train_batch_size * config.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # initialize global step, loss, acc to 0
    global_step, global_itm_loss, global_mlm_loss, global_itm_acc = 0, 0.0, 0.0, 0.0
    

    # zero if any previous grads exist
    model.zero_grad()
    # print(model)

    t_loss = v_loss = []

    # val_dataset = VLDataset(config, split='val')
    # val_dataloader = get_dataloader(config, val_dataset, is_train=False)

    for epoch in range(int(config.start_epoch), int(config.start_epoch)+int(config.num_train_epochs)):

        start = time.time()

        avg_epoch_itm_loss = avg_epoch_mlm_loss = 0.0
        avg_epoch_itm_acc = avg_epoch_mlm_acc = 0.0

        model.train()

        # let all processes sync up before starting with a new epoch of training
        if config.distributed:
            dist.barrier()
            # print("inside dist barrier")


        print(len(train_dataloader), len(train_dataloader.dataset))
        for step, (idx, batch) in enumerate(train_dataloader):

            # try:

                # batch = tuple(t.to(torch.device(config.local_rank)) for t in batch)
                batch = tuple(t.to(config.device) for t in batch)

                if config.auto_neg:
                    pos_text_ids, pos_seq_len, pos_text_mask, pos_mlm_mask, pos_target_caption_ids, pos_img_feature, pos_img_bbox, pos_img_feature_len, pos_vis_mask, pos_txt_token_type_ids, pos_vis_token_type_ids, pos_itm_label = batch[:12]
                    neg_text_ids, neg_seq_len, neg_text_mask, neg_mlm_mask, neg_target_caption_ids, neg_img_feature, neg_img_bbox, neg_img_feature_len, neg_vis_mask, neg_txt_token_type_ids, neg_vis_token_type_ids, neg_itm_label = batch[12:]


                    input_batch = {
                        'text_ids' : torch.cat([pos_text_ids, neg_text_ids], dim=0),
                        'text_mask': torch.cat([pos_text_mask, neg_text_mask], dim=0),
                        'mlm_mask' : torch.cat([pos_mlm_mask, neg_mlm_mask], dim=0),
                        'target_caption_ids' : torch.cat([pos_target_caption_ids, neg_target_caption_ids], dim=0),
                        'region_features' : torch.cat([pos_img_feature, neg_img_feature], dim=0),
                        'region_loc' : torch.cat([pos_img_bbox, neg_img_bbox], dim=0),
                        'vis_mask' : torch.cat([pos_vis_mask, neg_vis_mask], dim=0),
                        'txt_token_type_ids': torch.cat([pos_txt_token_type_ids, neg_txt_token_type_ids], dim=0),
                        'vis_token_type_ids': torch.cat([pos_vis_token_type_ids, neg_vis_token_type_ids], dim=0),
                        'itm_label' : torch.cat([pos_itm_label, neg_itm_label], dim=0) 
                    }

                # print("Batch size: ", pos_text_ids.size()[0])
                # print(pos_text_mask.shape)


                # print(itm_label)


                fwd_result = model(input_batch)
                itm_loss = fwd_result['ITM_Loss']
                mlm_loss = fwd_result['MLM_Loss']
                logits = fwd_result['logits']
                total_loss = config.itm_loss_weight * itm_loss + config.mlm_loss_weight * mlm_loss
                batch_itm_acc = compute_itm_accuacy_from_logits_train(logits, input_batch['itm_label']).sum()

                # break
                if config.gradient_accumulation_steps > 1:
                    total_loss = total_loss / config.gradient_accumulation_steps
                
                if config.use_mlm and config.use_itm:

                    if not config.distributed:
                        total_loss.mean().backward()


                        avg_epoch_itm_acc += (batch_itm_acc)
                        avg_epoch_itm_loss += itm_loss.mean().item()
                        avg_epoch_mlm_loss += mlm_loss.mean().item()                    

                    else:
                        total_loss.backward()

                        avg_epoch_itm_acc += (batch_itm_acc)
                        avg_epoch_itm_loss += (itm_loss.item())
                        avg_epoch_mlm_loss += mlm_loss.item()
                    

                elif config.use_mlm:
                    if not config.distributed:
                        mlm_loss.mean().backward()
                        avg_epoch_mlm_loss += mlm_loss.mean().item()   
                    else:
                        mlm_loss.backward()
                        avg_epoch_mlm_loss += mlm_loss.item()
                
                else:
                    if not config.distributed:
                        itm_loss.mean().backward()
                        avg_epoch_itm_acc += (batch_itm_acc.mean())
                        avg_epoch_itm_loss += (itm_loss.mean().item())

                    else:
                        itm_loss.backward()
                        global_itm_loss += itm_loss.item()
                        global_itm_acc += batch_itm_acc
                        # print(batch_itm_acc, logits.size()[0])
                        avg_epoch_itm_acc += (batch_itm_acc)
                        avg_epoch_itm_loss += (itm_loss.item())
                

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    
                    global_step += 1

                    if config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                    # print(config.local_rank)
                    # print(global_step)

                    if (global_step % config.logging_steps == 0) and (config.local_rank == 0):
                        batch_mlm_loss = mlm_loss/(input_batch['text_ids'].size()[0])
                        batch_itm_loss = itm_loss/(input_batch['text_ids'].size()[0])
                        logger.info(f"Epoch: {epoch+1}, device: {config.local_rank}, global_step: {global_step}, lr: {optimizer.param_groups[0]['lr']:.6f}, batch_mlm_loss: {batch_mlm_loss:}, batch_itm_loss: {batch_itm_loss:}, batch_itm_acc: {batch_itm_acc/input_batch['text_ids'].size()[0]:.4f}")


        print("finished frwd-backwrd loop \n")


        if config.local_rank == 0:

            torch.save(
                {
                'epoch' : epoch,
                'global_step' : global_step,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 
                # "/mnt/DATA3/mishra/abhiram/KMIS/VLM/vlm_eval/" + f"mmt2_checkpoint_{epoch}_{global_step}.pth",
            os.path.join(config.eval_model_dir, f'checkpoint_{epoch}.pt')
            )

                
        if config.local_rank == 0:
            t_loss.append((avg_epoch_itm_loss*2)/len(train_dataloader))
            #v_loss.append(avg_val_itm_loss/len(val_dataloader.dataset))
            logger.info(f"\n---------------------\n")
            logger.info(f"Epoch {epoch+1}, epoch_mlm_train_loss: {avg_epoch_mlm_loss/len(train_dataloader.dataset)}, epoch_itm_train_loss: {avg_epoch_itm_loss/len(train_dataloader.dataset)}, epoch_itm_train_acc: {avg_epoch_itm_acc/len(train_dataloader.dataset):.4f}, time:{time.time()-start:.2f}")
            #logger.info(f"Epoch {epoch+1}, epoch_itm_train_loss: {avg_epoch_itm_loss/len(train_dataloader.dataset):.6f}, epoch_itm_train_acc: {avg_epoch_itm_acc/len(train_dataloader.dataset):.4f}, epoch_itm_val_loss: {avg_val_itm_loss/len(val_dataloader.dataset):.6f}, epoch_itm_val_acc: {avg_val_itm_acc/len(val_dataloader.dataset):.4f} time:{time.time()-start:.2f}")            
            logger.info(f"\n---------------------\n")
            # break        
        # break
    
    # assert len(t_loss) == len(v_loss)
    plt.plot([i for i in range(len(t_loss))],t_loss,label = "Train Loss")
    plt.xlabel('No. of Epochs')
    plt.ylabel('Avg Loss')
    plt.title('Loss vs No. of Epochs')
    plt.legend()
    plt.savefig('plots/loss_curves_ft_easy.png')
    plt.close()
    # plt.show()
    return 0,0


def test(config, model, eval_dataset):
    print('TEST function here')

    config.eval_batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)    
    eval_dataloader = get_dataloader(config, eval_dataset, is_train=False)

    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(config.eval_batch_size))
    model.eval()
    results = {}
    softmax = torch.nn.Softmax(dim=1)
    global_itm_acc = 0.0
    c = 0
    for indexs, (_, batch) in enumerate(eval_dataloader):

        batch = tuple(t.to(torch.device(config.local_rank)) for t in batch) 
        with torch.no_grad():

            pos_text_ids, pos_seq_len, pos_text_mask, pos_mlm_mask, pos_target_caption_ids, pos_img_feature, pos_img_bbox, pos_img_feature_len, pos_vis_mask, pos_txt_token_type_ids, pos_vis_token_type_ids, pos_itm_label = batch[:12]
            neg_text_ids, neg_seq_len, neg_text_mask, neg_mlm_mask, neg_target_caption_ids, neg_img_feature, neg_img_bbox, neg_img_feature_len, neg_vis_mask, neg_txt_token_type_ids, neg_vis_token_type_ids, neg_itm_label = batch[12:]

            # print("Batch size: ", pos_text_ids.size()[0])
            # print(pos_text_mask.shape)

            input_batch = {
                'text_ids' : torch.cat([pos_text_ids, neg_text_ids], dim=0),
                'text_mask': torch.cat([pos_text_mask, neg_text_mask], dim=0),
                'mlm_mask' : torch.cat([pos_mlm_mask, neg_mlm_mask], dim=0),
                'target_caption_ids' : torch.cat([pos_target_caption_ids, neg_target_caption_ids], dim=0),
                'region_features' : torch.cat([pos_img_feature, neg_img_feature], dim=0),
                'region_loc' : torch.cat([pos_img_bbox, neg_img_bbox], dim=0),
                'vis_mask' : torch.cat([pos_vis_mask, neg_vis_mask], dim=0),
                'txt_token_type_ids': torch.cat([pos_txt_token_type_ids, neg_txt_token_type_ids], dim=0),
                'vis_token_type_ids': torch.cat([pos_vis_token_type_ids, neg_vis_token_type_ids], dim=0),
                'itm_label' : torch.cat([pos_itm_label, neg_itm_label], dim=0) 
            }
            c += input_batch['itm_label'].size()[0]
            fwd_result = model(input_batch)
            # loss = fwd_result['loss']
            logits = fwd_result['logits']

            batch_itm_acc = compute_itm_accuacy_from_logits_test(logits, input_batch['itm_label']).item()
            global_itm_acc += batch_itm_acc

            logger.info(f"Test Batch: {indexs}, Accuracy:{batch_itm_acc}")

    final_acc = global_itm_acc/(indexs+1)
    logger.info(f"Total Validation Accuracy for ITM is {final_acc}")
    return final_acc


# written only for binary classifier (ITM)
def compute_itm_accuacy_from_logits_train(logits, labels):
    # print(logits)
    # print(torch.max(logits, 1))[1]
    logits = torch.max(logits, 1)[1].data # argmax
    # print(logits)
    score = (logits == labels).sum()
    return score 


# written only for binary classifier (ITM)
def compute_itm_accuacy_from_logits_test(logits, labels):
    # print(logits)
    logits = torch.max(logits, 1)[1].data # argmax
    # print(logits)
    score = (logits == labels).sum()
    return score / logits.size()[0]



def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint_path", default=None, type=str, 
    #                     required=False,
    #                     help="if you want to resume training or eval"    
    #                     )
    parser.add_argument("--do_train", action='store_true', help="whether to train the model")
    parser.add_argument("--do_test", action='store_true', help="whether to test the model")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation."
                       "do not activate if we want to inference on dataset without gt labels.")   
    parser.add_argument('--do_1vs1000', action='store_true', help='set to true when testing 1 vs 1000,' 
                         'caption - image retrieval task')
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    parser.add_argument("--start_eval_id", type=int, default=0, help='start eval id')
    parser.add_argument("--end_eval_id", type=int, default=1000, help='end eval id')
    parser.add_argument("--id", type=int, default=1, help='part for 1 vs 100 eval result append')
    parser.add_argument('--best_epoch_test', action='store_true', help="set to true when you have some checkpoints, from which"
                                                                "best checkpoint has to be chosen")
    parser.add_argument('--mode', type=str, default='itm', help='mlm or itm; for itm model expects mlm prior finetuning.')

    args = parser.parse_args()

    # paths 
    data_config_path = 'config/config_' + args.mode + '/' + 'data_config.yaml'
    model_config_path = 'config/config_' + args.mode + '/' + 'model_config.yaml'
    train_config_path = 'config/config_' + args.mode + '/' + 'train_config.yaml'


    # load all configs
    data_config = load_config_file(data_config_path)
    train_config = load_config_file(train_config_path)
    model_config = load_config_file(model_config_path)

    config = OmegaConf.merge(train_config, data_config, model_config)


    # merging cli arguments
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))

    if config.distributed and not config.do_eval:
        args.is_master = args.local_rank == 0

        # set the device
        # args.device = torch.device(args.local_rank)

        # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`, but I find using environment variables makes it so that you can easily use the same script on different machines)
        dist.init_process_group(backend='nccl', init_method='env://')
        print(args.local_rank, "is initialized")
        torch.cuda.set_device(args.local_rank)



        # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
        torch.cuda.manual_seed_all(config.SEED)

    
    # setup logger and stuff
    global logger
    mkdir(path=config.output_dir)
    mkdir(path=config.eval_model_dir)

    logger = setup_logger("COFAR_ft", config.output_dir, 0)


    # config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # config.n_gpu = torch.cuda.device_count() 
    # uncomment above line if you have access to
    # all gpus in the machine (divinely :p)


    logger.warning("Device: %s, n_gpu: %s", torch.device(config.device), config.n_gpu)

    # tokenizer (already defined inside dataloader - can be skipped)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True) # saves bert files locally
    
    # tokenizer = BertTokenizer.from_pretrained('/nlsasfs/home/nltmocr/abhirama/bert_tokenizer')
    # creating an instance of our VLM - Visual + Language MultiModal Transformer
    
    model = VLM(model_config)

    # snippet to load model weights from a checkpoint
    # print(train_config.checkpoint_path)
    if train_config.load_from_checkpoint and not config.do_train:
        
        checkpoint_path = train_config.checkpoint_path
        assert checkpoint_path is not None
        assert os.path.isfile(checkpoint_path)
        logger.info("Checkpoint used: %s", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'])

    # put model on cuda device if available
    model = model.to(config.device)
    print(model)

 
    logger.info(f"Training/evaluation parameters {train_config}")
    logger.info(f"Whole config {config}")

    if config.do_train:

        train_dataset = VLDataset(config)

        global_step, avg_loss = train(config, train_dataset, model)
    
        

if __name__ == "__main__":
    main()
