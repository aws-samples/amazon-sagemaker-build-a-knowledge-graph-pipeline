import os
import json
import sys
import logging

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
logger.info(f"Current file path: {dir_path}")
logger.info(f"files under current file path: {os.listdir(dir_path)}")
logger.info(f"Working directory: {cwd}")
logger.info(f"files under working directory: {os.listdir(cwd)}")

from model import SubjectModel, ObjectModel
from dataset import DevDataset, dev_collate_fn
from utils import extract_spoes


bert_model_name = 'bert-base-chinese'
max_sent_len = 128
bert_dict_len = 21127
word_emb = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################
### SAGEMKAER LOAD MODEL FUNCTION
###################################

def model_fn(model_dir):
    subject_path = os.path.join(model_dir, 'subject.pt')
    object_path = os.path.join(model_dir, 'object.pt')
    id2predicate_path = os.path.join(model_dir, 'resources', 'id2predicate.json')
    
    logger.info(f"model dir is {model_dir}")
    logger.info(f"files under model dir: {os.listdir(model_dir)}")
    for file in os.listdir(model_dir):
        if os.path.isdir(file):
            logger.info(f"files under {file}: {os.listdir(file)}")
    
    if os.path.isfile(subject_path):
        logger.info(f"Model locates at: {model_dir}")
    else:
        logger.error(f"Can't find model at {model_dir}")
        logger.error(f"files under {model_dir}: {os.listdir(model_dir)}")
        logger.error(f"files in up folder: {os.listdir(os.path.join(model_dir, '..'))}")
    if os.path.isfile(id2predicate_path):
        logger.info(f"id2predicate locates at: {id2predicate_path}")
        id2predicate = json.load(open(id2predicate_path))
    else:
        logger.error(f"Can't find id2predicates at {id2predicate_path}")
        id2predicate = json.load(open('./model/resources/id2predicate.json'))
    subject_model = SubjectModel(bert_dict_len, word_emb).to(device)
    object_model = ObjectModel(word_emb, len(id2predicate)).to(device)
    subject_model.load_state_dict(torch.load(subject_path, map_location=device))
    object_model.load_state_dict(torch.load(object_path, map_location=device))
    if torch.__version__ == '1.5.1':
        import torcheia
        subject_model = subject_model.eval()
        
        # attach_eia() is introduced in PyTorch Elastic Inference 1.5.1,
        object_model = torcheia.jit.attach_eia(object_model, 0)
    return (subject_model, object_model, id2predicate)


###################################
### SAGEMKAER PREDICT FUNCTION
###################################

def predict_fn(input_data, model):
    '''
    Args:
    input_data (list(dict)): list of text and spoes pairs [{'text':, 'spo_list':},]
    model (tuple): subject model and object model
    
    Returns:
    rel_df (pd.Dataframe): a panda data frame that saves (subject, predicate, object) pairs
    '''
    
    subject_model, object_model, id2predicate = model
    subject_model.eval()
    object_model.eval()
    
    dataset = DevDataset(input_data, bert_model_name, 128)
    loader = DataLoader(
        dataset=dataset,  
        batch_size=256, 
        shuffle=False,
        num_workers=1,
        collate_fn=dev_collate_fn,
        multiprocessing_context='spawn',
    )
    
    rel_df = pd.DataFrame({'subject':[], 'predicate':[], 'object':[]})
    for batch in loader:
        texts, tokens, spoes, att_masks, offset_mappings = batch
        tokens, att_masks = tokens.to(device), att_masks.to(device)
        rels = extract_spoes(texts, tokens, offset_mappings, subject_model, object_model, \
                              id2predicate, attention_mask=att_masks)
        for rel in rels:
            rel_df.loc[len(rel_df)] = rel

    return rel_df


###################################
### SAGEMKAER MODEL INPUT FUNCTION
###################################

def input_fn(serialized_input_data, content_type="application/json"):
    input_data = json.loads(serialized_input_data)
    logger.info(f"input type: {type(input_data)}")
    logger.info(f"input data length: {len(input_data)}")
    if len(input_data) > 1:
        logger.info(f"first piece of data: {input_data[0]}")
    return input_data


###################################
### SAGEMKAER MODEL OUTPUT FUNCTION
###################################

def output_fn(prediction_output, accept="application/json"):
    return prediction_output.to_json(force_ascii=False), accept

