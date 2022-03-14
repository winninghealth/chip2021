# -*- coding: utf-8 -*-
"""
@author: Yiwen Jiang @Winning Health Group
"""

import torch
import torch.nn as nn
from typing import Dict
from overrides import overrides
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import FBetaMeasure

from modeling_cpt import CPTModel

LABEL_NAME = ['阳性','阴性','其他','不标注']
OUTPUT_LABEL_ABBR = ['Pos','Neg','Other','Empty']

class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class ClinicalFindingsClassificationModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_path: str,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.bert = CPTModel.from_pretrained(model_path)
        self.cls_head = ClassificationHead(input_dim=self.bert.config.d_model*2,
                                           inner_dim=self.bert.config.d_model*2,
                                           num_classes=vocab.get_vocab_size("labels"),
                                           pooler_dropout=self.bert.config.classifier_dropout)
        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            "Overall-Metric": FBetaMeasure(average='macro'),
            "Class-Metric": FBetaMeasure(labels=[vocab.get_token_index(i, namespace='labels') for i in LABEL_NAME])
        }
        self.ensemble_metrics = {
            "Ensemble-Metric": FBetaMeasure(average='macro'),
            "Ensemble-Class": FBetaMeasure(labels=[vocab.get_token_index(i, namespace='labels') for i in LABEL_NAME])
        }
    
    @overrides
    def forward(
        self,
        input_ids, token_type_ids, position_ids, speaker_ids,
        labels = None, **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        # R-Drop
        input_ids = input_ids.reshape(input_ids.shape[0] * input_ids.shape[1], -1)
        token_type_ids = token_type_ids.reshape(token_type_ids.shape[0] * token_type_ids.shape[1], -1)
        position_ids = position_ids.reshape(position_ids.shape[0] * position_ids.shape[1], -1)
        speaker_ids = speaker_ids.reshape(speaker_ids.shape[0] * speaker_ids.shape[1], -1)
        
        # PLM
        attention_mask = input_ids != 0
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            speaker_ids=speaker_ids,
                            decoder_attention_mask=attention_mask,
                            use_cache=True,
                            return_dict=True
                            )
        
        # encoder
        enc_hidden_states = outputs.encoder_last_hidden_state
        enc_rep = enc_hidden_states[:,0]
        
        # decoder
        dec_hidden_states = outputs.last_hidden_state
        eos_mask = input_ids.eq(self.bert.config.eos_token_id)
        dec_rep = dec_hidden_states[eos_mask,:].view(dec_hidden_states.size(0),-1,dec_hidden_states.size(-1))[:,-1,:]
        
        # classification head
        rep = torch.cat([enc_rep, dec_rep], dim=-1)
        logits = self.cls_head(rep)
        probs = torch.nn.functional.softmax(logits, -1)
        predict_res = torch.argmax(probs, dim=-1)
        output = {
            "logits": logits.reshape(logits.shape[0] // 2, 2, -1)[:,0],
            "probs": probs.reshape(probs.shape[0] // 2, 2, -1)[:,0],
            "results": predict_res.reshape(predict_res.shape[0] // 2, -1)[:,0]
            }
        
        # loss and metrics
        if labels is not None:
            # cross entropy loss
            labels_ce = labels.unsqueeze(1).repeat(1,2).reshape(labels.shape[0]*2)
            loss_ce = self.loss(logits, labels_ce)
            
            # Kullback-Leibler divergence Loss
            probs_kl = probs.reshape(probs.shape[0]//2, 2, -1)
            loss_kl = torch.nn.functional.kl_div(probs_kl[:,0].log(), probs_kl[:,1], 
                                                 reduction='batchmean') + \
                      torch.nn.functional.kl_div(probs_kl[:,1].log(), probs_kl[:,0],
                                                 reduction='batchmean')
            
            # total loss
            output["loss"] = loss_ce + loss_kl / 4 * 4
            
            # overall metrics
            for metric in self.metrics.values():
                metric(probs, labels_ce)
            # ensemble metrics
            ensemble_probs = (probs_kl[:,0] + probs_kl[:,1]) / 2
            for metric in self.ensemble_metrics.values():
                metric(ensemble_probs, labels)
        
        return output
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        
        results = self.metrics["Overall-Metric"].get_metric(reset)
        metrics_to_return['Macro-P'] = results['precision']
        metrics_to_return['Macro-R'] = results['recall']
        metrics_to_return['Macro-F1'] = results['fscore']
        
        results = self.metrics["Class-Metric"].get_metric(reset)
        for idx, i in enumerate(OUTPUT_LABEL_ABBR):
            metrics_to_return[i+'-P'] = results['precision'][idx]
            metrics_to_return[i+'-R'] = results['recall'][idx]
            metrics_to_return[i+'-F1'] = results['fscore'][idx]
        
        results = self.ensemble_metrics["Ensemble-Metric"].get_metric(reset)
        metrics_to_return['Macro-EP'] = results['precision']
        metrics_to_return['Macro-ER'] = results['recall']
        metrics_to_return['Macro-EF1'] = results['fscore']
        
        results = self.ensemble_metrics["Ensemble-Class"].get_metric(reset)
        for idx, i in enumerate(OUTPUT_LABEL_ABBR):
            metrics_to_return[i+'-EP'] = results['precision'][idx]
            metrics_to_return[i+'-ER'] = results['recall'][idx]
            metrics_to_return[i+'-EF1'] = results['fscore'][idx]
        
        return metrics_to_return
    
    