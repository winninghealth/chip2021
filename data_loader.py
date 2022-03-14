# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:57:06 2021

@author: Yiwen Jiang @Winning Health Group
"""

import torch
import pickle
import logging
from tqdm import tqdm
from typing import Dict, List
from overrides import overrides
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TensorField, LabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from transformers import BertTokenizer
logger = logging.getLogger(__name__)


class DialogueDatasetReader(DatasetReader):
    def __init__(
        self,
        transformer_load_path : str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(transformer_load_path)
    
    @overrides
    def _read(self, file_path):
        with open(file_path, "rb") as data_file:
            data = pickle.load(data_file)
            for i in tqdm(data):
                tokens = i['content']
                mention = i['mention']
                index = i['index']
                label = i['label']
                yield self.text_to_instance(tokens, mention, index, label)
    
    def text_to_instance(
        self,
        tokens: List[str], mention: str, index: List[int], label: str = None
    ) -> Instance:
        
        fields: Dict[str, Field] = {}
        
        input_sequence = ['[CLS]'] + tokens + ['[SEP]', '[unused2]', '[SEP]']
        token_type = (len(input_sequence) - 2) * [0] + [1] * 2
        position_ids = list(range(len(token_type)))
        
        input_sequence += [i for i in mention] + ['[SEP]']
        token_type += len([i for i in mention] + ['[SEP]']) * [1]
        position_ids += list(range(index[0]+1, index[1]+1))
        position_ids += range(len(position_ids),len(position_ids)+1)
        
        speaker_ids = []
        speaker_type = 0
        for i in input_sequence:
            if i in ['[SEP]','[CLS]']:
                speaker_ids.append(0)
                continue
            if i == '[unused2]':
                speaker_type = 2
            if i == '[unused1]':
                speaker_type = 1
            speaker_ids.append(speaker_type)
        for i in range(2,len(mention)+2):
            speaker_ids[-i] = speaker_ids[index[0]+i-1]
        
        assert len(input_sequence) <= 512 and len(input_sequence) == len(token_type) == len(position_ids) == len(speaker_ids)
        
        # input_ids
        sequence_bert = [self.tokenizer.convert_tokens_to_ids(i) for i in input_sequence]
        sequence_bert = torch.tensor([sequence_bert, sequence_bert])
        fields["input_ids"] = TensorField(sequence_bert)
        
        # token_type_ids
        token_type = torch.tensor([token_type, token_type])
        fields["token_type_ids"] = TensorField(token_type)
        
        # position_ids
        position_ids = torch.tensor([position_ids, position_ids])
        fields["position_ids"] = TensorField(position_ids)
        
        # speaker_ids
        speaker_ids = torch.tensor([speaker_ids, speaker_ids])
        fields["speaker_ids"] = TensorField(speaker_ids)
        
        # gold labels
        if label is not None:
            fields["labels"] = LabelField(label)
        
        return Instance(fields)

