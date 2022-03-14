# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:46:33 2021

@author: Yiwen Jiang @Winning Health Group
"""

import os
import json
import random
import pickle
import argparse
import pandas as pd

'''
Special Tokens for Dialogue Speaker.
Vocab File is Exactly the Same for Different Pre-trained Language Models in my Experiments.
'''
special_token2unused = {'医生':'[unused1]',
                        '患者':'[unused2]'}


def read_json_file(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        content = file.readlines()
        content = [json.loads(i.strip()) for i in content]
    file.close()
    return content


def save_json_file(file_path, data):
    with open(file_path,'w',encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    json_file.close()


def read_example_ids(fn):
    example_ids = pd.read_csv(fn)
    return example_ids


def truncation_corpus(dialog_content, ner_mention):
    sid = ner_mention[0]
    name = ner_mention[3]
    
    # Input Format: [CLS] ... [SEP] [unused2] [SEP] [entity_name] [SEP]
    extra_length = 5 + len(name) # 5 includes one [CLS], one [unused2] and three [SEP]
    max_sentence_len = 512 - extra_length
    
    if len(dialog_content[sid]) > max_sentence_len:
        start_idx = ner_mention[1]
        end_index = ner_mention[2]
        assert dialog_content[sid][0] in ['[unused1]', '[unused2]']
        if end_index <= max_sentence_len:
            input_x = dialog_content[sid][:max_sentence_len]
            assert input_x[start_idx:end_index] == list(name)
            assert len(input_x) <= max_sentence_len
        elif len(dialog_content[sid]) - start_idx < max_sentence_len:
            input_x = [dialog_content[sid][0]] + dialog_content[sid][-(max_sentence_len-1):]
            remove_len = len(dialog_content[sid]) - len(input_x)
            start_idx = start_idx - remove_len
            end_index = end_index - remove_len
            assert input_x[start_idx:end_index] == list(name)
            assert len(input_x) <= max_sentence_len
        else:
            forward_info = start_idx - (max_sentence_len - len(name) - 1) // 2
            backward_info = end_index + (max_sentence_len - len(name) - 1) // 2
            input_x = [dialog_content[sid][0]] + dialog_content[sid][forward_info:backward_info]
            start_idx = start_idx - forward_info + 1
            end_index = end_index - forward_info + 1
            assert input_x[start_idx:end_index] == list(name)
            assert len(input_x) <= max_sentence_len
    else:
        current_sentence_len = len(dialog_content[sid])
        forward_search = list(range(sid-1,-1,-1))
        backward_search =list(range(sid+1,len(dialog_content)))
        search_seq = list()
        contextual_info = [sid]
        if dialog_content[sid][0] == '[unused1]':
            for i in zip(backward_search, forward_search):
                search_seq.append(i[0])
                search_seq.append(i[1])
        elif dialog_content[sid][0] == '[unused2]':
            for i in zip(forward_search, backward_search):
                search_seq.append(i[0])
                search_seq.append(i[1])
        else:
            assert dialog_content[sid][0] in ['[unused1]', '[unused2]']
        if len(forward_search) > len(backward_search):
            search_seq += forward_search[len(backward_search):]
        elif len(forward_search) < len(backward_search):
            search_seq += backward_search[len(forward_search):]
        else:
            assert len(forward_search) == len(backward_search) and len(search_seq) == len(forward_search) + len(backward_search)
        for i in search_seq:
            contextual_info = sorted(contextual_info)
            if not (i == contextual_info[0] - 1 or i == contextual_info[-1] + 1):
                break
            if len(dialog_content[i]) + current_sentence_len <= max_sentence_len:
                current_sentence_len += len(dialog_content[i])
                contextual_info.append(i)
            else:
                continue
        contextual_info = sorted(contextual_info)
        assert sid in contextual_info
        forward_len = 0
        for i in range(contextual_info.index(sid)):
            forward_len += len(dialog_content[contextual_info[i]])
        input_x = list()
        for i in contextual_info:
            input_x += dialog_content[i]
        start_idx = ner_mention[1] + forward_len
        end_index = ner_mention[2] + forward_len
        try:
            assert input_x[start_idx:end_index] == list(name)
        except:
            print(input_x[start_idx:end_index], list(name))
        assert len(input_x) <= max_sentence_len
        
    return input_x, start_idx, end_index


def save_data(data, example_ids, mode, fn, fn_sample, fn_ids):
    assert mode in ['train','valid']
    eids = example_ids[example_ids['split_' + str(fn_ids)] == mode]['example_id'].to_list()
    input_x, input_mention, input_index, output_y = [], [], [], []
    for i in data:
        if i['dialog_id'] not in eids:
            continue
        dialog_content, target_labels = list(), list()
        for t in i['dialog_info']:
            sender = special_token2unused[t['sender']]
            content = [sender] + list(t['text'])
            dialog_content.append(content)
            sentence_id = int(t['sentence_id'])
            
            # Fix BAD sentence ids in original train.jsonl file released by CHIP2021
            if i['dialog_id'] == 1424 and sentence_id >= 44 and sentence_id <= 92:
                sentence_id -= 10
            elif i['dialog_id'] == 1424 and sentence_id > 93:
                sentence_id -= 11
            
            assert len(dialog_content) == sentence_id
            sentence_id -= 1
            for ner in t['ner']:
                assert ''.join(content[ner['range'][0]+1:ner['range'][1]+1]) == ner['mention']
                target_labels.append((sentence_id, ner['range'][0]+1, ner['range'][1]+1, ner['mention'], ner['attr']))
        target_labels = sorted(target_labels)
        for t in target_labels:
            input_tmp, start_idx, end_index = truncation_corpus(dialog_content, t)
            input_x.append(input_tmp)
            input_mention.append(t[3])
            assert input_tmp[start_idx:end_index] == list(t[3])
            input_index.append([start_idx, end_index])
            output_y.append(t[4])
    
    assert len(input_x) == len(input_mention) == len(input_index) == len(output_y)
    
    corpus = []
    for i in zip(input_x, input_mention, input_index, output_y):
        if i[3] == '':
            continue
        corpus.append({'content': i[0],
                       'mention': i[1],
                       'index': i[2],
                       'label': i[3]})
    
    with open(fn, 'wb') as pickle_file:
        pickle.dump(corpus, pickle_file)
    pickle_file.close()
    
    with open(fn_sample, 'wb') as pickle_file:
        pickle.dump(random.sample(corpus, 100), pickle_file)
    pickle_file.close()


def preprocess_badcase(oridata, fixdata):
    idx2id = dict()
    for i, d in enumerate(fixdata):
        idx2id[d['dialog_id']] = i
    for i, d in enumerate(oridata):
        if d['dialog_id'] in idx2id.keys():
            oridata[i] = fixdata[idx2id[d['dialog_id']]]
    return oridata
    

def preprocess(config):
    dataset = read_json_file(config.input_file)
    badcase = read_json_file('./data/dataset/fix_badcase.jsonl')
    dataset = preprocess_badcase(dataset, badcase)
    
    example_ids = read_example_ids(config.split_file)
    data_dir = config.output_path
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir + '/train')
        os.makedirs(data_dir + '/valid')
        
    # FIVE-Fold Cross Validation
    fold_num = 5
    for i in range(fold_num):
        train_path = os.path.join(data_dir, 'train', 'train' + str(i) + '.pkl')
        train_sample_path = os.path.join(data_dir, 'train', 'train_sample' + str(i) + '.pkl')
        valid_path = os.path.join(data_dir, 'valid', 'valid' + str(i) + '.pkl')
        valid_sample_path = os.path.join(data_dir, 'valid', 'valid_sample' + str(i) + '.pkl')
        
        save_data(dataset, example_ids, 'train', train_path, train_sample_path, i)
        save_data(dataset, example_ids, 'valid', valid_path, valid_sample_path, i)
    print('Data Preprocess Finished!')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="./data/dataset/CHIP-MDCFNPC_train.jsonl", type=str, help="Input file to preprocess for training")
    parser.add_argument("--split_file", default="./data/dataset/split.csv", type=str, help="Input file for multi-fold cross validation")
    parser.add_argument("--output_path", default="./data/dialog_data", type=str, help="Output file after preprocess")
    
    config = parser.parse_args()
    preprocess(config)
    
    