from pathlib import Path
from torch.utils import data
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
import pandas as pd


import json, random, time, os, base64
from pprint import pprint
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
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
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))  

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# torch.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


class VLDataset(Dataset):

    def __init__(self, config, tokenizer=None, split='train', is_train=True):
        super(VLDataset).__init__()

        if tokenizer is None:
            # self.tokenizer = BertTokenizer.from_pretrained('/nlsasfs/home/nltmocr/abhirama/bert_tokenizer', local_files_only=True) # saves bert files locally
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        # train flag
        self.is_train = is_train
        # self.dir_split = 'train' if split in ['train', 'val'] else 'test'
        self.dir_split = config.dir_split    

        # file paths
        # self.captions_file = config.captions_file_path
        # self.img_features_path = config.img_features_path
        self.captions_file = op.join('data', '{}_captions.json'.format(config.dir_split))
        self.img_features_path = op.join(config.data_dir, 'train_obj_frcn_features/')

        # max lens
        self.max_seq_len = config.max_seq_len
        self.max_vis_seq_len = config.max_vis_seq_len

        # mask probability
        # TODO: replace masked tokens with random token
        self.mask_probability = config.mask_probability


        self.config = config

        if is_train:
            self.num_captions_per_img = config.num_captions_per_img
        else:
            self.num_captions_per_img = config.num_captions_per_img_val
        
        # image-caption data info
        self.captions = read_json(self.captions_file)
        self.img_keys = list(self.captions.keys())
        
        if config.use_cofar_knowledge:

            self.knowledge_file = op.join(config.data_dir, 'KB_oracle_brand.json') # brand


            self.knowledge = read_json(self.knowledge_file)
            self.img_knowledge_pairs = {}

            for each_img_key in self.img_keys:
                img_id = each_img_key.split('_')[0]

                if img_id in self.knowledge.keys():
                    self.img_knowledge_pairs[each_img_key] = self.knowledge[img_id]
                else:
                    self.img_knowledge_pairs[each_img_key] = ''

            

    def get_image_caption_idx(self, index):

        #TODO: loader comaptability for ~val~ and test set eval
        
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img

        return img_idx, [self.img_keys[img_idx], cap_idx]

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

    def get_label(self, index):

        img_idx, cap_idx = self.get_image_caption_idx(index)
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0


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
        mlm_mask = self.get_txt_mask(text_ids, self.mask_probability)
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

        return (text_ids, seq_len, text_mask, mlm_mask, target_caption_ids, img_feature, img_bbox, img_feature_len, vis_mask, text_token_type_ids, vis_token_type_ids)

    
    def __getitem__(self, index: int):
        
        # if self.is_train:


        if not self.config.use_cofar_knowledge:                

            img_idx, cap_idxs = self.get_image_caption_idx(index)
            img_key = self.img_keys[img_idx]

            img_feature, img_bbox = self.get_img_features_and_bbox(img_key)

            assert len(img_feature) == len(img_bbox)

            caption = self.captions[cap_idxs[0]][cap_idxs[1]]

            example = self.tensorize_example(caption, img_feature, img_bbox)

            # select a negative pair
            # TODO: hard negative implementation.

            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
            img_idx_neg = random.choice(neg_img_indexs)

            if random.random() <= 0.5:
                # randomly select a random caption as negative caption
                # for current example from the negative image
                cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                example_neg = self.tensorize_example(caption_neg, img_feature, img_bbox)
            
            else:
                # select a random image
                img_feature_neg, img_bbox_neg = self.get_img_features_and_bbox(self.img_keys[img_idx_neg])
                assert len(img_feature_neg) == len(img_bbox_neg)
                example_neg = self.tensorize_example(caption, img_feature_neg, img_bbox_neg)

            # a pair of len = 12, 
            # first 6 features - caption, img_feature, img_bbox, and label=1 for +ve example
            # last 6 features - caption, img_feature, img_bbox, and label=0 for -ve example
            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])

            return index, example_pair

        # IMPORTANT-- SET use_cofar_knowledge inside data_config to False if you dont want to use knowledge track.
        elif self.config.use_cofar_knowledge:

            img_idx, cap_idxs = self.get_image_caption_idx(index)
            img_key = self.img_keys[img_idx]

            img_feature, img_bbox = self.get_img_features_and_bbox(img_key)

            assert len(img_feature) == len(img_bbox)

            # caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            caption = self.captions[cap_idxs[0]][cap_idxs[1]] + ". " + self.img_knowledge_pairs[img_key]
            # print(self.img_knowledge_pairs[img_key])
            example = self.tensorize_example(caption, img_feature, img_bbox)

            # select a negative pair
            # TODO: hard negative implementation.

            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
            img_idx_neg = random.choice(neg_img_indexs)

            # select a random image
            img_feature_neg, img_bbox_neg = self.get_img_features_and_bbox(self.img_keys[img_idx_neg])
            assert len(img_feature_neg) == len(img_bbox_neg)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]] + ". " + self.img_knowledge_pairs[self.img_keys[img_idx_neg]]
            # print(self.img_knowledge_pairs[self.img_keys[img_idx_neg]])
            example_neg = self.tensorize_example(caption, img_feature_neg, img_bbox_neg)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])

            return index, example_pair


    def __len__(self):
        # if not self.is_train and self.config['cross_image_eval']:
        #     return len(self.img_keys) ** 2 * self.num_captions_per_img
        print(len(self.img_keys) * self.num_captions_per_img)
        return len(self.img_keys) * self.num_captions_per_img


""" 
    data loader to generate data <caption against 1000 images, where 1 image is true caption's corresponding
    image and remaining 999 images are negative images, 
    This data loader generates, one positive pair and 1000 negative images.
    
"""
class VL_TestDataset(Dataset):

    def __init__(self, config, tokenizer=None, split='train', is_train=True, start_index=0, end_index=1000):
        super(VLDataset).__init__()

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True) # saves bert files locally
        else:
            self.tokenizer = tokenizer

        # train flag
        self.is_train = is_train
        self.dir_split = 'train' if split in ['train', 'val'] else 'test'
    

        # file paths
        # self.captions_file = config.captions_file_path
        # self.img_features_path = config.img_features_path
        image_captions = read_json('/mnt/DATA3/mishra/abhiram/KMIS/knowledge-aware-mmt/data/coco/coco_val_ir_data.json')

        # self.captions_file = op.join(config.data_dir, '{}_captions.json'.format(split))
        self.img_features_path = op.join(config.data_dir, 'train_obj_frcn_features/')

        # max lens
        self.max_seq_len = config.max_seq_len
        self.max_vis_seq_len = config.max_vis_seq_len

        # mask probability
        # TODO: replace masked tokens with random token
        self.mask_probability = config.mask_probability


        self.config = config

        if is_train:
            self.num_captions_per_img = config.num_captions_per_img
        else:
            self.num_captions_per_img = config.num_captions_per_img_val
        
        # image-caption data info
        self.captions = read_json(self.captions_file)
        self.img_keys = list(self.captions.keys())
        
        imageIds = list(self.captions.keys())
        # print(len(captionIds))

        self.final_capIds = []
        self.final_captions = []
        self.final_images = []
        self.final_labels = []

        for index, (key, value) in enumerate(self.captions.items()):

            if index < start_index:
                continue

        #     print(key, value)
            self.final_capIds.append(index)
            pos_caption = random.sample(value, 1)
            self.final_captions.append(pos_caption)
            self.final_images.append(key)
            self.final_labels.append(1)

            # neg_images = random.sample(list(set(imageIds).difference(set([key]))), 999)
            neg_images = [x for x in self.captions.keys() if x != key]  
        #   print(len(neg_images), key in neg_images)

            for each_neg_image in neg_images:
                self.final_capIds.append(index)
                self.final_captions.append(pos_caption)
                self.final_images.append(each_neg_image)
                self.final_labels.append(0)
            
            if index == end_index:
                break
        
        

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
        mlm_mask = self.get_txt_mask(text_ids, self.mask_probability)
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

        return (text_ids, seq_len, text_mask, mlm_mask, target_caption_ids, img_feature, img_bbox, img_feature_len, vis_mask, text_token_type_ids, vis_token_type_ids)

    
    def __getitem__(self, index: int):
        
        # if self.is_train:

            
            caption = self.final_captions[index]
            img_key = self.final_images[index]

            img_feature, img_bbox = self.get_img_features_and_bbox(img_key)

            assert len(img_feature) == len(img_bbox)

            example = self.tensorize_example(caption, img_feature, img_bbox)

            example_pair = tuple(list(example) + [self.final_labels[index]])

            return index, example_pair, self.final_capIds[index], self.final_images[index]
        

    def __len__(self):

        return len(self.final_capIds)
    

