# -*- coding: utf-8 -*-
"""
@author: Yiwen Jiang @Winning Health Group
"""

import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance, Vocabulary
from allennlp.predictors.predictor import Predictor

from utils import init_logger
from trainer import build_model
from data_loader import DialogueDatasetReader
from data_preprocess import read_json_file, truncation_corpus

logger = logging.getLogger(__name__)

LABEL_ORDER = ['阳性','阴性','其他','不标注']
special_token2unused = {'医生':'[unused1]', '患者':'[unused2]'}


class ClinicalFindingsPredictor(Predictor):    
    def __init__(self, model, dataset_reader) -> None:
        super().__init__(model, dataset_reader)
        self.vocab = model.vocab
        self.order = self.vocab.get_token_to_index_vocabulary(namespace='labels')
    
    def predict(self, input_sentence, input_mention, input_index) -> JsonDict:
        result = self.predict_batch_json([{"sentence":input_sentence[idx],
                                           "mention":input_mention[idx],
                                           "index":input_index[idx]} for idx in range(0, len(input_sentence))])
        
        instances = [i['probs'].index(max(i['probs'])) for i in result]
        instance_probs = [i['probs'] for i in result]
        for idx, i in enumerate(instances):
            instances[idx] = self.vocab.get_token_from_index(i, namespace='labels')
        for idx, i in enumerate(instance_probs):
            instance_probs[idx] = [instance_probs[idx][self.order[LABEL_ORDER[i]]] for i in range(len(LABEL_ORDER))]
        return instances, instance_probs
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict["sentence"], json_dict['mention'], json_dict['index'])


def get_device(pred_config):
    return pred_config.cuda_id if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def read_input_file(input_path):
    data = read_json_file(input_path)
    input_info, input_x, input_mention, input_index = [], [], [], []
    for i in data:
        dialog_content, target_labels = list(), list()
        for t in i['dialog_info']:
            sender = special_token2unused[t['sender']]
            content = [sender] + list(t['text'])
            dialog_content.append(content)
            sentence_id = int(t['sentence_id'])
            assert len(dialog_content) == sentence_id
            sentence_id -= 1
            for ner in t['ner']:
                assert ''.join(content[ner['range'][0]+1:ner['range'][1]+1]) == ner['mention']
                target_labels.append((sentence_id, ner['range'][0]+1, ner['range'][1]+1, ner['mention']))
                input_info.append((i['dialog_id'], sentence_id, ner['range'][0], ner['range'][1], ner['mention']))
        for t in target_labels:
            input_tmp, start_idx, end_index = truncation_corpus(dialog_content, t)
            input_x.append(input_tmp)
            input_mention.append(t[3])
            input_index.append([start_idx, end_index])
    assert len(input_info) == len(input_x) == len(input_mention) == len(input_index)
    return input_info, input_x, input_mention, input_index


def write_predict_file(input_path, predict_result, pred_path):
    with open(input_path, 'r', encoding = 'utf-8') as input_data, open(pred_path, 'w', encoding = 'utf-8') as output_data:
        for line in input_data:
            line = line.strip()
            json_content = json.loads(line)
            dialog_id = json_content['dialog_id']
            for block in json_content['dialog_info']:
                sentence_id = int(block['sentence_id']) - 1
                for ner_block in block['ner']:
                    idx_start, index_end = ner_block['range']
                    mention = ner_block['mention']
                    ner_block['attr'] = predict_result[(dialog_id,sentence_id,idx_start,index_end,mention)]
                    
                    # Due to BAD index from original files released by CHIP2021
                    # 4 BAD cases in testb.txt
                    if (dialog_id,sentence_id,idx_start,index_end,mention) == (2576,0,23,28,'有痰呈白色'):
                        ner_block['range'] = [23,29]
                    if (dialog_id,sentence_id,idx_start,index_end,mention) == (302,10,24,32,'大便粘三四天一次'):
                        ner_block['range'] = [24,33]
                    if (dialog_id,sentence_id,idx_start,index_end,mention) == (6202,8,36,43,'肿块地方不舒服'):
                        ner_block['range'] = [36,44]
                    if (dialog_id,sentence_id,idx_start,index_end,mention) == (2670,5,1,7,'齿状痕很明显'):
                        ner_block['range'] = [1,8]
                    # 2 BAD cases in testa.txt
                    if (dialog_id,sentence_id,idx_start,index_end,mention) == (2944,0,7,13,'痰都是硬块状'):
                        ner_block['range'] = [7,14]
                    if (dialog_id,sentence_id,idx_start,index_end,mention) == (4126,3,15,21,'拉稀伴随着血'):
                        ner_block['range'] = [15,23]
                    
            output_data.write(json.dumps(json_content, ensure_ascii = False) + '\n')


def write_predict_probs(predict_probs, pred_probs_path):
    with open(pred_probs_path,'w',encoding='utf-8') as json_file:
        json.dump(predict_probs, json_file, ensure_ascii=False, indent=4)
    json_file.close()


def predict(pred_config):
    serialization_dir = pred_config.model_dir
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    model_dir = os.path.join(serialization_dir, pred_config.model_name)
    vocab = Vocabulary.from_files(vocabulary_dir)
    model = build_model(vocab, model_path=pred_config.pretrained_model_dir)
    device = get_device(pred_config)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.to(device)
    dataset_reader = DialogueDatasetReader(transformer_load_path=pred_config.pretrained_model_dir)
    predictor = ClinicalFindingsPredictor(model=model, dataset_reader=dataset_reader)
    input_info, input_x, input_mention, input_index = read_input_file(os.path.join(pred_config.test_input_file))
    batch_size = pred_config.batch_size
    predict_result, predict_probs = dict(), list()
    for i in tqdm(range(0,len(input_info),batch_size)):
        result, result_probs = predictor.predict(input_sentence = input_x[i:i+batch_size],
                                                 input_mention = input_mention[i:i+batch_size],
                                                 input_index = input_index[i:i+batch_size])
        for j in zip(input_info[i:i+batch_size], result):
            uid, res = j
            predict_result[uid] = res
        for j in zip(input_info[i:i+batch_size], result_probs):
            uid, probs = j
            predict_probs.append({'uid':uid, 'probs':probs})
    pred_path = os.path.join(pred_config.test_output_file)
    pred_probs_path = os.path.join(pred_config.test_probs_file)
    write_predict_file(os.path.join(pred_config.test_input_file), predict_result, pred_path)
    write_predict_probs(predict_probs, pred_probs_path)
    logger.info("Prediction Done!")


if __name__ == "__main__":
    
    init_logger()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_input_file", default="./data/dataset/testb.txt", type=str, help="Input file for prediction")
    parser.add_argument("--test_output_file", default="submission.txt", type=str, help="Output file for prediction")
    parser.add_argument("--test_probs_file", default="probs.json", type=str, help="Output Probs file for prediction")
    
    parser.add_argument("--model_dir", default="./save_model", type=str, help="Path to load model")
    parser.add_argument("--model_name", default="best.th", type=str, help="model name to load")
    parser.add_argument("--pretrained_model_dir", default="./PLMs", type=str, help="Path to pretrained language model")
    
    parser.add_argument("--batch_size", default=48, type=int, help="Batch size for prediction")
    parser.add_argument("--cuda_id", default='cuda:0', type=str, help="GPU Selection")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    pred_config = parser.parse_args()
    model = predict(pred_config)
    
    