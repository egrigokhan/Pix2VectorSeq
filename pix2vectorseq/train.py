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

from dataset import CFG, load_dataset, VectorTokenizer, get_loaders
from models import Encoder, Decoder, EncoderDecoder
from utils import get_lr, AvgMeter, seed_everything

import argparse

def train_epoch(model, train_loader, optimizer, lr_scheduler, criterion, logger=None):
    model.train()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for x, y in tqdm_object:
        x, y = x.to(CFG.device, non_blocking=True), y.to(
            CFG.device, non_blocking=True)

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        preds = model(x, y_input)

        loss = criterion(
            preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_meter.update(loss.item(), x.size(0))

        lr = get_lr(optimizer)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.6f}")
        if logger is not None:
            logger.log({"train_step_loss": loss_meter.avg, 'lr': lr})

    return loss_meter.avg


def valid_epoch(model, valid_loader, criterion):
    model.eval()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for x, y in tqdm_object:
            x, y = x.to(CFG.device, non_blocking=True), y.to(
                CFG.device, non_blocking=True)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            preds = model(x, y_input)
            loss = criterion(
                preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))

            loss_meter.update(loss.item(), x.size(0))

    return loss_meter.avg


def train_eval(model,
               train_loader,
               valid_loader,
               criterion,
               optimizer,
               lr_scheduler,
               step,
               logger):

    best_loss = float('inf')

    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}")
        if logger is not None:
            logger.log({"Epoch": epoch + 1})

        train_loss = train_epoch(model, train_loader, optimizer,
                                 lr_scheduler if step == 'batch' else None,
                                 criterion, logger=logger)

        valid_loss = valid_epoch(model, valid_loader, criterion)
        print(f"Valid loss: {valid_loss:.3f}")

        if step == 'epoch':
            pass

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_valid_loss.pth')
            print("Saved Best Model")

        if logger is not None:
            logger.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss
            })
            logger.save('best_valid_loss.pth')


def train(
    dataset_path,
    num_heads=4,
    num_layers=3,
):
    train_df, valid_df = load_dataset(
        f"{dataset_path}/dataset.csv")

    tokenizer = VectorTokenizer(num_bins=CFG.num_bins,
                                width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    train_loader, valid_loader = get_loaders(
        train_df, valid_df, tokenizer, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)

    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=num_heads, num_layers=num_layers)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    num_training_steps = CFG.epochs * \
        (len(train_loader.dataset) // CFG.batch_size)
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_training_steps=num_training_steps,
                                                   num_warmup_steps=num_warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx)

    train_eval(model,
               train_loader,
               valid_loader,
               criterion,
               optimizer,
               lr_scheduler=lr_scheduler,
               step='batch',
               logger=None)


""" 
   parse args and run training

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Pix2VectorSeq Model")
    parser.add_argument(
        "--dataset_path",
        default="/Users/gokhanegri/Documents/Pix2VectorSeq",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--num_heads",
        default=4,
        type=int,
        help="number of heads",
    )
    parser.add_argument(
        "--num_layers",
        default=3,
        type=int,
        help="number of layers",
    )
    args = parser.parse_args()

    #  parse args
    train(**vars(args))
