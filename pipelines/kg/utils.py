import os
import torch
import numpy as np
import time

import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bad_tokens = ['，', ',', '》', '《', '"', '：', '  ', '…', '、']


def get_now_time():
    a = time.time()
    return time.ctime(a)


def seq_padding_vec(X):
    L = [len(x) for x in X]
    ML = max(L)
    # print("ML",ML)
    return [x + [[1, 0]] * (ML - len(x)) for x in X]


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def extract_spoes(texts, token_ids, offset_mappings, subject_model, object_model, id2predicate, attention_mask=None, writer=None, global_step=None):    
    with torch.no_grad():
        subject_preds, hidden_states = subject_model(token_ids) #(batch_size, sent_len, 2)
        # magic numbers come from https://github.com/bojone/bert4keras/blob/master/examples/task_relation_extraction.py

        extracted_subjects = (subject_preds > 0.55).sum().item()
        if writer is not None and global_step is not None:
            writer.add_scalar('eval/extracted_subject', extracted_subjects/2, global_step)
        
        batch_size = subject_preds.shape[0]
        spoes = []
        all_subjects_text = []
        for k in range(batch_size):
            sub_start = torch.where(subject_preds[k, :, 0] > 0.6)[0]
            sub_end = torch.where(subject_preds[k, :, 1] > 0.5)[0]
            subjects = []
            for i in sub_start:
                j = sub_end[sub_end >= i]
                if len(j) > 0:
                    j = j[0]
                    subjects.append((i, j))
            text = texts[k]
            offset_mapping = offset_mappings[k]
            subjects_text = [text[offset_mapping[i][0]: offset_mapping[j][-1]] for i, j in subjects]
            all_subjects_text += subjects_text
            if subjects:
                # print("len(text)", len(text))
                # print("len(token_ids)", len(token_ids))
                # print("subject_preds.shape", subject_preds.shape)
                subjects = torch.tensor(subjects)
                # create pseudo batch: repeat k-th embedding on newly inserted dim 0
                pseudo_states = torch.stack([hidden_states[k]]*len(subjects), dim=0) # (len(subjects), sent_len, emb_size)
                pseudo_mask = torch.stack([attention_mask[k]]*len(subjects), dim=0)
                object_preds = object_model(pseudo_states, subjects, attention_mask=pseudo_mask)
                for subject, object_pred in zip(subjects, object_preds):
                    sub_text_head = offset_mapping[subject[0]][0]
                    sub_text_tail = offset_mapping[subject[1]][-1]
                    subject_text = text[sub_text_head: sub_text_tail]
                    # if subject contrain bad tokens, skip this subject
                    if any(t in subject_text for t in bad_tokens):
                        continue
                    obj_start = torch.where(object_pred[:, :, 0] > 0.6)
                    obj_end = torch.where(object_pred[:, :, 1] > 0.5)
                    for _start, predicate1 in zip(*obj_start):
                        for _end, predicate2 in zip(*obj_end):
                            if _start <= _end and predicate1 == predicate2:
                                text = texts[k]
                                offset_mapping = offset_mappings[k]
                                obj_text_head = offset_mapping[_start][0]
                                obj_text_tail = offset_mapping[_end][-1]
                                # id2predicate has str keys
                                predicate_text = id2predicate[str(int(predicate1.item()))]
                                object_text = text[obj_text_head: obj_text_tail]
                                if not any(t in object_text for t in bad_tokens):
                                    spoes.append(
                                        (subject_text,
                                        predicate_text,
                                        object_text)
                                    )
                                break
            if writer is not None and global_step is not None and global_step % 500 == 0:
                writer.add_text('eval/extracted_subject', str(all_subjects_text), global_step)
    return spoes


def para_eval(subject_model, object_model, loader, id2predicate, device=torch.device('cpu'), batch_eval=False, epoch=None, writer=None):
    """
    Returns:
    f1, precision, recall
    """
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for step, batch in tqdm(iter(enumerate(loader)), desc='Eval'):
        texts, tokens, spoes, att_masks, offset_mappings = batch
        tokens, att_masks = tokens.to(device), att_masks.to(device)
        global_step = epoch*len(loader)+step
        R = set(extract_spoes(texts, tokens, offset_mappings, subject_model, object_model, id2predicate, attention_mask=att_masks, writer=writer, global_step=global_step))
        T = set()
        for spo_list in spoes:
            T.update([tuple(spo) for spo in spo_list])
        A += len(R & T)
        B += len(R)
        C += len(T)
        if writer is not None and global_step % 500 == 0:
            writer.add_text("eval/extracted_spo", str(R), epoch*len(loader)+step)
            writer.add_text("eval/gold_spo", str(T), epoch*len(loader)+step)
        
        cnt += 1
    return 2 * A / (B + C), A / B, A / C


def seq_max_pool(x):
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)


# helper functions to upload data to s3
def write_to_s3(filename, bucket, prefix):
    import boto3
    # put one file in a separate folder. This is helpful if you read and prepare data with Athena
    filename_key = filename.split("/")[-1]
    key = os.path.join(prefix, filename_key)
    s3 = boto3.resource('s3')
    return s3.Bucket(bucket).upload_file(filename, key)


def upload_to_s3(bucket, prefix, filename):
    url = "s3://{}/".format(bucket, os.path.join(prefix, filename.split('/')[-1]))
    print("Writing to {}".format(url))
    write_to_s3(filename, bucket, prefix)


def evaluate(subject_model, object_model, loader, id2predicate, epoch, writer=None, device=torch.device('cpu')):
    subject_model.eval()
    object_model.eval()
    f1, precision, recall = para_eval(subject_model, object_model, loader, id2predicate, device=device, epoch=epoch, writer=writer)
    print(f"Eval epoch {epoch}: f1: {f1}, precision: {precision}, recall: {recall}")
    if writer:
        writer.add_scalar('eval/f1', f1, epoch)
        writer.add_scalar('eval/precision', precision, epoch)
        writer.add_scalar('eval/recall', recall, epoch)
    return f1, precision, recall