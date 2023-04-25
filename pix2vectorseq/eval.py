import argparse
import gc
import math
import os
import random
import xml.etree.ElementTree as ET
from functools import partial
from glob import glob

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import transformers
from dataset import (CFG, VectorDatasetTest, VectorTokenizer, get_loaders,
                     load_dataset, postprocess)
from models import Decoder, Encoder, EncoderDecoder
from sklearn.model_selection import StratifiedGroupKFold
from timm.models.layers import trunc_normal_
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, top_k_top_p_filtering
from utils import AvgMeter, get_lr, seed_everything


def generate(model, x, tokenizer, max_len=50, top_k=0, top_p=1):
    x = x.to(CFG.device)
    batch_preds = torch.ones(x.size(0), 1).fill_(
        tokenizer.BOS_code).long().to(CFG.device)
    confs = []

    if top_k != 0 or top_p != 1:
        def sample(preds): return torch.softmax(
            preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        def sample(preds): return torch.softmax(
            preds, dim=-1).argmax(dim=-1).view(-1, 1)

    with torch.no_grad():
        for i in range(max_len):
            preds = model.predict(x, batch_preds)
            # If top_k and top_p are set to default, the following line does nothing!
            preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)
            if i % 4 == 0:
                confs_ = torch.softmax(
                    preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                confs.append(confs_)
            preds = sample(preds)
            batch_preds = torch.cat([batch_preds, preds], dim=1)

    return batch_preds.cpu(), confs


def eval(model_path='./best_valid_loss.pth', image_path='./test_images/0.png'):
    tokenizer = VectorTokenizer(num_bins=CFG.num_bins,
                                width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=4, num_layers=3)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)
    msg = model.load_state_dict(torch.load(
        model_path, map_location=CFG.device))
    print(msg)
    model.eval()

    img_paths = [image_path]

    test_dataset = VectorDatasetTest(img_paths)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(img_paths), shuffle=False, num_workers=0)

    all_bboxes = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for x in tqdm(test_loader):
            batch_preds, batch_confs = generate(
                model, x, tokenizer, max_len=101, top_k=0, top_p=1)
            print(batch_preds, batch_confs)
            bboxes, labels, confs = postprocess(
                batch_preds, batch_confs, tokenizer)
            #  all_bboxes.extend(bboxes)
            #  all_labels.extend(labels)
            #  all_confs.extend(confs)

    os.mkdir("results")
    for i, (bboxes, labels, confs) in enumerate(zip(all_bboxes, all_labels, all_confs)):
        img_path = img_paths[i]
        img = cv2.imread(img_path)[..., ::-1]
        img = cv2.resize(img, (CFG.img_size, CFG.img_size))
        img = visualize(img, bboxes, labels, id2cls, show=False)

        cv2.imwrite("results/" + img_path.split("/")[-1], img[..., ::-1])


""" 
   parse args and run training

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate vector sequence from image")
    parser.add_argument(
        "--model_path", type=str, default="./best_valid_loss.pth", help="path to model")
    parser.add_argument(
        "--image_path", type=str, default="./test_images/0.png", help="path to image")
    
    args = parser.parse_args()

    #  parse args
    eval(**vars(args))
