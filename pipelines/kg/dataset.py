from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from utils import sequence_padding


class DevDataset(Dataset):
    def __init__(self, data, bert_model_name, max_sent_len):
        super().__init__()
        self.data = data
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        self.max_sent_len = max_sent_len

    def __getitem__(self, index):
        return self.process_data(self.data[index])

    def __len__(self):
        return len(self.data)

    def process_data(self, d):
        text = d['text']
        output = self.tokenizer.encode_plus(text, max_length=self.max_sent_len, truncation=True, 
            padding=True, return_offsets_mapping=True)
        token = output['input_ids']
        att_mask = output['attention_mask']
        offset_mapping = output['offset_mapping']
        return text, token, d['spo_list'], att_mask, offset_mapping


def dev_collate_fn(data):
    texts = [item[0] for item in data]
    tokens = [item[1] for item in data] # bsz *[(1, sent_len)]
    # tokens = torch.cat(tokens, dim=0) # (bsz, sent_len)
    spoes = [item[2] for item in data] # bsz * [list of spoes]
    att_masks = [item[3] for item in data] # bsz * [(1, sent_len)]
    offset_mappings = [item[4] for item in data]
    tokens = torch.tensor(sequence_padding(tokens))
    att_masks = torch.tensor(sequence_padding(att_masks))
    return texts, tokens, spoes, att_masks, offset_mappings


class TrainDataset(Dataset):
    def __init__(self, data, bert_model_name, max_sent_len, predicate2id):
        super().__init__()
        self.data = data
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name) # "bert-base-chinese"
        self.max_sent_len = max_sent_len
        self.predicate2id = predicate2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_data = self.data[index]
        return self.process_data(raw_data)
    
    def process_data(self, d):
        encoded = self.tokenizer(d['text'], max_length=self.max_sent_len, 
            padding=True, truncation=True)
        token_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        # {s: [(o, p)]}
        spoes = defaultdict(list)
        for s, p, o in d['spo_list']:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.predicate2id[p]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1, p)
                spoes[s].append(o)
        
        subject_labels = np.zeros((len(token_ids), 2))
        object_labels = np.zeros((len(token_ids), len(self.predicate2id), 2))
        
        # assign subject as (0, 0) if there's no subject in this sentence. i.e. truncated
        subject_ids = (0, 0)
        if spoes:
            for s in spoes:
                subject_labels[s[0], 0] = 1
                subject_labels[s[1], 1] = 1
            # Pick a subject randomly
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            end = np.random.choice(end[end >= start])
            subject_ids = (start, end)
            
            # Corresponding object label
            for o in spoes.get(subject_ids, []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1
        
        return token_ids, attention_mask, subject_ids, subject_labels, object_labels

def train_collate_fn(data):
    batch_token_ids = [item[0] for item in data]
    batch_attention_masks = [item[1] for item in data]
    batch_subject_ids = [item[2] for item in data]
    batch_subject_labels = [item[3] for item in data]
    batch_object_labels = [item[4] for item in data]

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids))
    batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks))
    batch_subject_ids = torch.tensor(batch_subject_ids)
    batch_subject_labels = torch.FloatTensor(sequence_padding(batch_subject_labels))
    batch_object_labels = torch.FloatTensor(sequence_padding(batch_object_labels))
    return batch_token_ids, batch_attention_masks, batch_subject_ids, batch_subject_labels, batch_object_labels


def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence) - len(pattern)):
        if sequence[i:i + n] == pattern:
            return i
    return -1
