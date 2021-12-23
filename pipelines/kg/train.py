import subprocess
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"files under cur folder: {os.listdir(dir_path)}")
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", f"{dir_path}/requirements.txt"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard<2.4"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard<2.4"])

import logging
import argparse
import pathlib
import json
from datetime import datetime
import shutil

import numpy as np
import boto3
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SubjectModel, ObjectModel
from dataset import TrainDataset, DevDataset, train_collate_fn, dev_collate_fn
from utils import evaluate

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.005)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--schema', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Model-specific parameters
    parser.add_argument('--bert-model-name', type=str, default='bert-base-chinese')
    parser.add_argument('--bert-dict-len', type=int, default=21127)
    parser.add_argument('--word-emb', type=int, default=128)
    parser.add_argument('--max-sent-len', type=int, default=128) # around 2% of the sentences would be truncated for sent_len of 128
    
    # Logs
    parser.add_argument('--logname', type=str, default='')

    return parser.parse_known_args()


def get_train_loader(args):
    train_data = json.load(open(f"{args.train}/train.json"))
    _, predicate2id = json.load(open(f"{args.schema}/schema.json"))
    train_dataset = TrainDataset(train_data, args.bert_model_name, args.max_sent_len, predicate2id)
    train_loader = DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=args.batch_size,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=1,
        collate_fn=train_collate_fn,      # subprocesses for loading data
    )
    return train_loader


def get_val_loader(args):
    val_data = json.load(open(f"{args.val}/val.json"))
    val_dataset = DevDataset(val_data, args.bert_model_name, args.max_sent_len)
    val_loader = DataLoader(
        dataset=val_dataset,      # torch TensorDataset format
        batch_size=args.batch_size,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=1,
        collate_fn=dev_collate_fn,      # subprocesses for loading data
        multiprocessing_context='spawn',
    )
    return val_loader


def get_model(args):
    _, predicate2id = json.load(open(f"{args.schema}/schema.json"))
    num_class = len(predicate2id)
    subject_model = SubjectModel(args.bert_dict_len, args.word_emb)
    object_model = ObjectModel(args.word_emb, num_class)
    return subject_model, object_model


def get_tb_writer(args):
    now = datetime.now()
    dt_str = now.strftime("%m_%d_%H_%M")
    dt_str = '2021_Aug' if dt_str is None else dt_str
    if args.logname == '':
        log_dir = os.path.join(args.output_data_dir, 'tb', dt_str)
    else:
        log_dir = os.path.join(args.output_data_dir, 'tb', args.logname + '_' + dt_str)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"Tensorboard logs are saved at {log_dir}")
    return writer


def train(subject_model, object_model, train_loader, optimizer, epoch, device=torch.device('cpu'), writer=None, log_interval=10):
    subject_model.train()
    object_model.train()
    train_tqdm = tqdm(enumerate(train_loader), desc="Train")
    for step, batch in train_tqdm:
        token_ids, attention_masks, subject_ids, subject_labels, object_labels = batch
        token_ids, attention_masks, subject_ids, subject_labels, object_labels = \
            token_ids.to(device), attention_masks.to(device), subject_ids.to(device), \
            subject_labels.to(device), object_labels.to(device)
        # predict
        subject_preds, hidden_states = subject_model(token_ids, attention_mask=attention_masks)
        object_preds = object_model(hidden_states, subject_ids, attention_masks)
        # calc loss
        subject_loss = F.binary_cross_entropy(subject_preds, subject_labels, reduction='none') # (bsz, sent_len)
        attention_masks = attention_masks.unsqueeze(dim=2)
        subject_loss = torch.sum(subject_loss * attention_masks) / torch.sum(attention_masks) # ()
        object_loss = F.binary_cross_entropy(object_preds, object_labels, reduction='none') # (bsz, sent_len, n_classes, 2)
        object_loss = torch.mean(object_loss, dim=2) # (bsz, sent_len, 2)
        object_loss = torch.sum(object_loss * attention_masks) / torch.sum(attention_masks) # ()
        loss_sum = subject_loss + object_loss * 10
        train_tqdm.set_postfix(loss=loss_sum.item())
        #updates
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        with torch.no_grad():
            exists_subject = subject_labels.sum().item()
            correct_subject = torch.logical_and(subject_preds > 0.6, subject_labels > 0.6).sum().item()
            exists_object = object_labels.sum().item()
            correct_object = torch.logical_and(object_preds > 0.5, object_labels > 0.5).sum().item()

            if step % log_interval == 0:
                logger.info(f"epoch {epoch}, step: {step}, loss: {loss_sum.item()}, subject_recall: {correct_subject}/{exists_subject}, object_recall: {correct_object}/{exists_object}")
                if writer:
                    writer.add_scalar('train/loss', loss_sum.item(), step + epoch * len(train_loader))
                    writer.add_scalar('train/loss_subject', subject_loss.item(), step + epoch * len(train_loader))
                    writer.add_scalar('train/loss_object', object_loss.item(), step + epoch * len(train_loader))
                    writer.add_scalar('train/recall_subject', correct_subject/exists_subject, step + epoch * len(train_loader))
                    writer.add_scalar('train/recall_object', correct_object/exists_object, step + epoch * len(train_loader))
                    

if __name__ == "__main__":
    
    args, _ = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = get_tb_writer(args)

    logger.info('Training data location: {}'.format(args.train))
    logger.info('Validation data location: {}'.format(args.val))
    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    logger.info('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))


    subject_model, object_model = get_model(args)
    subject_model = subject_model.to(device)
    object_model = object_model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info('Using', torch.cuda.device_count(), "GPUs!")
        subject_model = nn.DataParallel(subject_model)
        object_model = nn.DataParallel(object_model)
    
    params = subject_model.parameters()
    params = list(params) + list(object_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    id2predicate, _ = json.load(open(f"{args.schema}/schema.json"))
    
    best_f1 = 0
    for e in range(args.epochs):
        train(subject_model, object_model, train_loader, optimizer, e, device=device, writer=writer, log_interval=10)    
        # Evaluate on dev set
        f1, precision, recall = evaluate(subject_model, object_model, val_loader, id2predicate, e, writer, device=device)
        # Check whether to save every 5 epochs
        if e % 5 == 0 and f1 > best_f1:
            best_f1 = f1
            # save model
            torch.save(subject_model.state_dict(), f"{args.model_dir}/subject.pt")
            torch.save(object_model.state_dict(), f"{args.model_dir}/object.pt")
    
    # Save resources for inference: id2predicate, requirements.txt and inference.py
    # this folder will be uploaded together with model
    pathlib.Path(f"{args.model_dir}/resources").mkdir(parents=True, exist_ok=True)
    with open(f"{args.model_dir}/resources/id2predicate.json", 'w', encoding='utf-8') as f:
        json.dump(id2predicate, f, indent=4, ensure_ascii=False)
    with open(f"{args.model_dir}/resources/config.json", 'w') as f:
        json.dump(vars(args), f)
    pathlib.Path(f"{args.model_dir}/code").mkdir(parents=True, exist_ok=True)
    shutil.copy('./requirements.txt', f"{args.model_dir}/code/")
    shutil.copy('./inference.py', f"{args.model_dir}/code/")
    shutil.copy('./model.py', f"{args.model_dir}/code/")
    shutil.copy('./utils.py', f"{args.model_dir}/code/")
    shutil.copy('./dataset.py', f"{args.model_dir}/code/")