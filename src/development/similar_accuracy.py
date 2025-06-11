#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.


import logging

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial

from datasets import DatasetDict, concatenate_datasets
from omegaconf import DictConfig
from src.pretrain_utils import (to_multiple_of_eight, preprocess_similar,
                                get_model, get_bert_model, preprocess_nps)
from transformers import set_seed, DataCollatorWithPadding
from string_trans.generator import SampleGenerator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from src.pretrain_utils import PretrainDataset, PretrainAugDataset, SimilarDataset
from src.modeling_program_io import ContrastiveNPSModel
from tqdm.auto import tqdm
import os
from time import time
from torchmetrics import Accuracy
import seaborn as sns

logger = logging.getLogger(__name__)


@torch.no_grad()
def similar_accuracy(cfg: DictConfig):
    trainer_args = cfg.trainer
    data_args = cfg.data
    model_args = cfg.model
    logger_args = cfg.logger

    # ipdb.set_trace()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # accelerator = Accelerator(mixed_precision=trainer_args.mixed_precision)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO)

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    # accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    
    if model_args.from_bert:
        configure_model = get_bert_model
    else:
        configure_model = get_model
    config, p_tokenizer, io_tokenizer, model = configure_model(model_args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = model.to(device)

    raw_datasets = DatasetDict.load_from_disk(data_args.dataset_dir)



    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = [col for col in raw_datasets["train"].column_names if col not in ["program", "inputs", "outputs"]]


    processed_datasets = raw_datasets.map(
        preprocess_nps,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    similar_datasets = raw_datasets.map(
        preprocess_similar,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    string_generator = SampleGenerator(data_args.num_demo, processed_datasets["train"]["program"])
    max_sample=128

    train_dataset = PretrainDataset(processed_datasets["train"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, data_args=data_args)
    train_similar_dataset = SimilarDataset(similar_datasets["train"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, data_args=data_args, max_similar=max_sample)
    eval_similar_dataset = SimilarDataset(similar_datasets["val"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, data_args=data_args, max_similar=max_sample)
    test_similar_dataset = SimilarDataset(similar_datasets["test"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, data_args=data_args, max_similar=max_sample)
    eval_dataset = PretrainDataset(processed_datasets["val"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, data_args=data_args)
    test_dataset = PretrainDataset(processed_datasets["test"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, data_args=data_args)
    whole_dataset = PretrainDataset(concatenate_datasets([processed_datasets["train"], processed_datasets["val"], processed_datasets["test"]]), program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, data_args=data_args)



    program_collator = DataCollatorWithPadding(
        p_tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )
    io_collator = DataCollatorWithPadding(
        io_tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    def collate_fn(examples, io_padding: DataCollatorWithPadding, program_padding: DataCollatorWithPadding):
        io = [{"input_ids": example["io_input_ids"], "attention_mask": example["io_attention_mask"]} for example in examples]
        io = io_padding(io)
        program = [{"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples]
        program = program_padding(program)
        return program, io

    def similar_collate_fn(examples, io_padding: DataCollatorWithPadding, program_padding: DataCollatorWithPadding):
        # ground truth is index 0 of all example
        similar_program = torch.stack([example["similar_program"] for example in examples])
        similar_io = [{"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples]
        similar_io = io_padding(similar_io)
        # similar_io in shape (batch_size, 512, io_length)
        io = [{"input_ids": example["io_input_ids"], "attention_mask": example["io_attention_mask"]} for example in examples]
        io = io_padding(io)
        program = [{"input_ids": example["program_input_ids"], "attention_mask": example["program_attention_mask"]} for example in examples]
        program = program_padding(program)
        return program, io, similar_program, similar_io



    data_collate_function = partial(collate_fn, io_padding=io_collator, program_padding=program_collator)
    similar_data_collate_function = partial(similar_collate_fn, io_padding=io_collator, program_padding=program_collator)


    whole_loader = DataLoader(
        whole_dataset, shuffle=True, collate_fn=data_collate_function, batch_size=trainer_args.per_device_train_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collate_function, batch_size=trainer_args.per_device_train_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=data_collate_function, batch_size=trainer_args.per_device_eval_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, collate_fn=data_collate_function, batch_size=trainer_args.per_device_eval_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )

    train_similar_loader = DataLoader(
        train_similar_dataset, shuffle=True, collate_fn=similar_data_collate_function, batch_size=trainer_args.per_device_eval_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )
    eval_similar_loader = DataLoader(
        eval_similar_dataset, shuffle=False, collate_fn=similar_data_collate_function, batch_size=trainer_args.per_device_eval_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )
    test_similar_loader = DataLoader(
        test_similar_dataset, shuffle=False, collate_fn=similar_data_collate_function, batch_size=trainer_args.per_device_eval_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )

    model = model.to(device)

    all_loaders = [(train_loader, train_similar_loader), (eval_loader, eval_similar_loader), (test_loader, test_similar_loader)]
    split_name = ["train", "eval", "test"]

    for split, (basic_loader, similar_loader) in zip(split_name, all_loaders):

        program_embeddings = []

        for program, io in tqdm(basic_loader):
            for k in program:
                program[k] = program[k].to(device)
            for k in io:
                io[k] = io[k].to(device)
            program_embedding = model.get_program_embeds(**program)
            program_embeddings.append(program_embedding)

        program_embeddings = torch.cat(program_embeddings, dim=0)
        accuracy_metric = Accuracy("multiclass", num_classes=max_sample).to(device=device)

        similar_program_accuracy = []
        similar_io_accuracy = []

        for (program, io, similar_program, similar_io) in tqdm(similar_loader):
            
            bs, n_sample, io_length = similar_io["input_ids"].shape

            for k in program:
                program[k] = program[k].to(device)
            for k in io:
                io[k] = io[k].to(device)
            similar_program = similar_program.to(device)
            for k in similar_io:
                similar_io[k] = similar_io[k].to(device).view(bs * n_sample, io_length)
            
            program_embedding = model.get_program_embeds(**program)
            similar_program_embedding = program_embeddings[similar_program]
            similar_io_embedding = model.get_io_embeds(**similar_io)
            similar_io_embedding = similar_io_embedding.view(bs, n_sample, -1)
            io_embedding = model.get_io_embeds(**io)

            # eval similar program
            logits = torch.bmm(io_embedding.unsqueeze(1), similar_program_embedding.transpose(1, 2)).squeeze(1) * model.logit_scale.exp()
            labels = torch.zeros(bs, dtype=torch.long).to(device)
            acc = accuracy_metric(logits, labels)
            similar_program_accuracy.append(acc.item())

            # eval similar io
            logits =  torch.bmm(program_embedding.unsqueeze(1), similar_io_embedding.transpose(1, 2)).squeeze(1) * model.logit_scale.exp()
            labels = torch.zeros(bs, dtype=torch.long).to(device)
            acc = accuracy_metric(logits, labels)
            similar_io_accuracy.append(acc.item())

        print(f"{split.capitalize()} similar program accuracy out of {max_sample} samples: {np.round(np.mean(similar_program_accuracy) * 100, 2)}")
        print(f"{split.capitalize()} similar io accuracy out of {max_sample} samples: {np.round(np.mean(similar_io_accuracy) * 100, 2)}")



