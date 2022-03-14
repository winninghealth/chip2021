# -*- coding: utf-8 -*-
"""
@author: Yiwen Jiang @Winning Health Group
"""

from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
# The Random Seeds might be Different in my Experiments of Different Pre-trained Language Models 
prepare_environment(Params({"random_seed":1000, "numpy_seed":2000, "pytorch_seed":3000}))

import os
import torch
import argparse
from typing import Iterable
from allennlp.data import (
    Instance,
    Vocabulary,
)
from utils import init_logger
from allennlp.models import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.learning_rate_schedulers import LinearWithWarmup

from transformers import AdamW
from data_loader import DialogueDatasetReader


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary, model_path) -> Model:
    if 'CPT_base' in model_path:
        from model_cpt import ClinicalFindingsClassificationModel
    else:
        from model import ClinicalFindingsClassificationModel
    return ClinicalFindingsClassificationModel(vocab=vocab, model_path=model_path)


def build_trainer(model, train_loader, dev_loader, cuda_device, serialization_dir, config) -> Trainer:
    
    no_decay = ["bias", "layer_norm", "layernorm"]
    optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.1},
            {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0}]
    
    t_total = len(train_loader) // config.gradient_accumulation_steps * config.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.bert_lr, eps=config.adam_epsilon)
    lrschedule = LinearWithWarmup(optimizer=optimizer,
                                  num_epochs=config.num_train_epochs,
                                  num_steps_per_epoch=len(train_loader),
                                  warmup_steps=0.1*t_total)
    
    ckp = Checkpointer(serialization_dir=serialization_dir,
                       num_serialized_models_to_keep=1)
    
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     patience=config.patience,
                                     validation_metric='+Macro-F1',
                                     validation_data_loader=dev_loader,
                                     num_epochs=config.num_train_epochs,
                                     serialization_dir=serialization_dir,
                                     cuda_device=cuda_device if str(cuda_device) != 'cpu' else -1,
                                     learning_rate_scheduler=lrschedule,
                                     num_gradient_accumulation_steps=config.gradient_accumulation_steps,
                                     checkpointer=ckp,
                                     run_sanity_checks=False)
    
    return trainer


def run_training_loop(config):
    serialization_dir = config.output_model_dir
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    os.makedirs(serialization_dir, exist_ok=True)
    dataset_reader = DialogueDatasetReader(transformer_load_path=config.pretrained_model_dir)
    train_path = config.train_file
    train_data = list(dataset_reader.read(train_path))
    dev_path = config.dev_file
    dev_data = list(dataset_reader.read(dev_path))
    vocab = build_vocab(train_data + dev_data)
    vocab.save_to_files(vocabulary_dir)
    device = torch.device(config.cuda_id if torch.cuda.is_available() else 'cpu')
    model = build_model(vocab, config.pretrained_model_dir)
    model = model.to(device)
    train_loader = MultiProcessDataLoader(dataset_reader, train_path, 
                                          batch_size=config.batch_size, shuffle=True)
    dev_loader = MultiProcessDataLoader(dataset_reader, dev_path, 
                                        batch_size=config.batch_size, shuffle=False)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    trainer = build_trainer(model, train_loader, dev_loader, device, serialization_dir, config)
    trainer.train()
    return trainer


if __name__ == '__main__':
    
    init_logger()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_file", default='./data/dialog_data/train/train0.pkl', type=str)
    parser.add_argument("--dev_file", default='./data/dialog_data/valid/valid0.pkl', type=str)
    parser.add_argument("--pretrained_model_dir", default='./PLMs/Roberta_base', type=str)
    parser.add_argument("--output_model_dir", default='./save_model/Roberta_base/save_model_0/', type=str)
    parser.add_argument("--cuda_id", default='cuda:0', type=str)
    
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--patience", default=2, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument('--bert_lr', default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    
    config = parser.parse_args()
    run_training_loop(config)
    
    