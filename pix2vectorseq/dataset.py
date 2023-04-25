import gc
import os
import cv2
import math
import random
from glob import glob
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import timm
from timm.models.layers import trunc_normal_

import transformers
from transformers import top_k_top_p_filtering
from transformers import get_linear_schedule_with_warmup


class CFG:
    img_path = "/content/VOCdevkit/VOC2012/JPEGImages"
    xml_path = "/content/VOCdevkit/VOC2012/Annotations"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_len = 300
    img_size = 384
    num_bins = 384  #  img_size

    batch_size = 10
    epochs = 10

    model_name = 'deit3_small_patch16_384_in21ft1k'
    num_patches = 576
    lr = 1e-4
    weight_decay = 1e-4

    generation_steps = 101


def split_df(df, n_folds=5, training_fold=0):
    return df, df


def load_dataset(csv_path="dataset.csv"):
    df = pd.read_csv(csv_path)
    df['id'] = np.arange(len(df))

    train_df, valid_df = split_df(df)
    print("Train size: ", train_df['id'].nunique())
    print("Valid size: ", valid_df['id'].nunique())

    return train_df, valid_df


def get_transform_train(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(size, size),
        A.Normalize(),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_transform_valid(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None, tokenizer=None):
        self.ids = np.arange(len(df))
        self.df = df
        self.transforms = transforms
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img_path = sample['image']

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:     # we have an alpha channel
            a1 = ~img[:, :, 3]        # extract and invert that alpha
            # add up values (with clipping)
            img = cv2.add(cv2.merge([a1, a1, a1, a1]), img)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)[
                :100, :100, :]    # strip alpha channel
            img = cv2.resize(img, (384, 384))

        # img = img[..., ::-1]

        sentence = sample['sentence']
        objects, colors = parse_sentence(sentence)
        objects = np.array(objects)
        colors = np.array(colors)

        '''
        if self.transforms is not None:
            transformed = self.transforms(**{
                'image': img,
                'bboxes': bboxes,
                'labels': labels
            })
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        '''

        img = torch.FloatTensor(img).permute(2, 0, 1)

        if self.tokenizer is not None:
            seqs = self.tokenizer(np.array(objects), np.array(colors))
            seqs = torch.LongTensor(seqs)
            return img, seqs

        return img, objects, colors

    def __len__(self):
        return len(self.ids)


def parse_sentence(sentence):
    #  remove the <BOS> and <EOS> tokens
    sentence = sentence.replace("  <CLR>", " <CLR>")
    sentence = sentence.split(' ')
    sentence.pop(0)
    sentence.pop(-1)

    sentence = np.array(sentence)

    #  split by <OBJ> token
    objects = np.split(sentence, np.where(sentence == '<OBJ>')[0])[1:]
    objects = [obj[1:] for obj in objects]

    #  split each object by <CLR> token, first element is the object, second is the color
    objects = [np.split(obj, np.where(obj == '<CLR>')[0]) for obj in objects]

    #  remove the <CLR> token
    objects = [[obj[obj != '<CLR>'] for obj in obj] for obj in objects]

    #  convert the objects and colors to floats
    objects = [[list(np.array(obj).astype('float32'))
                for obj in obj] for obj in objects]

    objects = np.array(objects)

    return list(objects[:, 0]), list(objects[:, 1])


class VectorTokenizer:
    def __init__(self, num_bins: int, width: int, height: int, max_len=500):
        # self.num_classes = num_classes
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.max_len = max_len

        # BOS - Begining Of Sentence
        self.BOS_code = num_bins

        # OBJ - Object
        self.OBJ_code = self.BOS_code + 1

        # CLR - Color
        self.CLR_code = self.OBJ_code + 1

        #  EOS - End Of Sentence
        self.EOS_code = self.CLR_code + 1

        # PAD - Padding
        self.PAD_code = self.EOS_code + 1

        self.vocab_size = num_bins + 5

    def quantize(self, x: np.array):
        """
        x is a real number in [0, 1]
        """
        return (x * (self.num_bins - 1)).astype('int')

    def dequantize(self, x: np.array):
        """
        x is an integer between [0, num_bins-1]
        """
        return x.astype('float32') / (self.num_bins - 1) - 50

    def __call__(self, objects: list, colors: list, shuffle=True):
        assert len(objects) == len(
            colors), "objects and colors must have the same length"
        objects = np.array(objects)
        colors = np.array(colors)

        #  pick every even and odd number in objects
        objects[:, ::2] = ((objects[:, ::2]) + 50) / (self.width)
        objects[:, 1::2] = ((objects[:, 1::2]) + 50) / (self.height)

        #  pick every number in color
        colors = colors / 255

        #  quantize the objects and colors
        objects = self.quantize(objects)[:self.max_len]
        colors = self.quantize(colors)[:self.max_len]

        if shuffle:
            rand_idxs = np.arange(0, len(objects))
            np.random.shuffle(rand_idxs)
            objects = objects[rand_idxs]
            colors = colors[rand_idxs]

        #  start with the BOS token
        tokenized = [self.BOS_code]
        i = 0
        for object, color in zip(objects, colors):
            tokens = [self.OBJ_code]
            tokens.extend(list(object))
            tokens.extend([self.CLR_code])
            tokens.extend(list(color))

            tokenized.extend(list(map(int, tokens)))

        tokenized = tokenized[:self.max_len - 1]
        tokenized.append(self.EOS_code)

        return tokenized

    def decode(self, tokens: torch.tensor):
        """
        toekns: torch.LongTensor with shape [L]
        """
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[1:-1]
        # assert len(tokens) % 5 == 0, "invalid tokens"

        objects = []
        colors = []

        tokens = np.array(tokens)

        #  split tokens by <OBJ>
        tokens = np.split(tokens, np.where(tokens == self.OBJ_code)[0])[1:]

        #  remove the <OBJ> tokens
        tokens = [token[token != self.OBJ_code] for token in tokens]

        #  split tokens by <CLR>
        tokens = [np.split(token, np.where(token == self.CLR_code)[0])
                  for token in tokens]

        #  remove the <CLR> tokens
        tokens = [[token[token != self.CLR_code]
                   for token in token] for token in tokens]

        #  dequantize the objects and colors
        objects = [self.dequantize(token[0]) for token in tokens]
        colors = [self.dequantize(token[1]) for token in tokens]

        return objects, colors


def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length
    """
    image_batch, seq_batch = [], []
    for image, seq in batch:
        image_batch.append(image)
        seq_batch.append(seq)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len -
                         seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    return image_batch, seq_batch


def get_loaders(train_df, valid_df, tokenizer, img_size, batch_size, max_len, pad_idx, num_workers=2):

    train_ds = VectorDataset(train_df, transforms=get_transform_train(
        img_size), tokenizer=tokenizer)

    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_ds = VectorDataset(valid_df, transforms=get_transform_valid(
        img_size), tokenizer=tokenizer)

    validloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=2,
        pin_memory=True,
    )

    return trainloader, validloader


class VectorDatasetTest(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        self.image_paths = img_paths

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        print(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:     # we have an alpha channel
            a1 = ~img[:, :, 3]        # extract and invert that alpha
            # add up values (with clipping)
            img = cv2.add(cv2.merge([a1, a1, a1, a1]), img)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)[
                :100, :100, :]    # strip alpha channel
            img = cv2.resize(img, (384, 384))

        '''
        if self.transforms is not None:
            transformed = self.transforms(**{
                'image': img,
                'bboxes': bboxes,
                'labels': labels
            })
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        '''

        img = torch.FloatTensor(img).permute(2, 0, 1)

        return img

    def __len__(self):
        return len(self.image_paths)

def postprocess(batch_preds, batch_confs, tokenizer):
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    ## sanity check
    invalid_idxs = ((EOS_idxs - 1) % 5 != 0).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0
    
    all_bboxes = []
    all_labels = []
    all_confs = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0:
            all_bboxes.append(None)
            all_labels.append(None)
            all_confs.append(None)
            continue
        labels, bboxes = tokenizer.decode(batch_preds[i, :EOS_idx+1])
        confs = [round(batch_confs[j][i].item(), 3) for j in range(len(bboxes))]
        
        all_bboxes.append(bboxes)
        all_labels.append(labels)
        all_confs.append(confs)
        
    return all_bboxes, all_labels, all_confs