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
from functools import partial

from datasets import DatasetDict, concatenate_datasets, Dataset
from omegaconf import DictConfig
from src.pretrain_utils import get_model, get_bert_model, preprocess_nps
from transformers import set_seed, DataCollatorWithPadding
from string_trans.generator import SampleGenerator
from torch.utils.data import DataLoader
from src.pretrain_utils import PretrainDataset, PretrainAugDataset
import multiprocessing as mp
from tqdm.auto import tqdm
import faiss
import os

logger = logging.getLogger(__name__)


@torch.no_grad()
def clip_difficulty(cfg: DictConfig):
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


    preprocess_function = preprocess_nps
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    string_generator = SampleGenerator(data_args.num_demo, processed_datasets["train"]["program"])


    train_aug_dataset = PretrainAugDataset(processed_datasets["train"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, io_generator=string_generator, data_args=data_args)
    eval_aug_dataset = PretrainAugDataset(processed_datasets["val"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, io_generator=string_generator, data_args=data_args)
    test_aug_dataset = PretrainAugDataset(processed_datasets["test"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, io_generator=string_generator, data_args=data_args)
    train_dataset = PretrainDataset(processed_datasets["train"], program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer, data_args=data_args)
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

    data_collate_function = partial(collate_fn, io_padding=io_collator, program_padding=program_collator)

    whole_loader = DataLoader(
        whole_dataset, shuffle=True, collate_fn=data_collate_function, batch_size=trainer_args.per_device_train_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collate_function, batch_size=trainer_args.per_device_train_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )
    train_aug_loader = DataLoader(
        train_aug_dataset, shuffle=True, collate_fn=data_collate_function, batch_size=trainer_args.per_device_train_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=data_collate_function, batch_size=trainer_args.per_device_eval_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, collate_fn=data_collate_function, batch_size=trainer_args.per_device_eval_batch_size, num_workers=0 if cfg.debug else 16, drop_last=False,
    )
    model = model.to(device)

    programs = processed_datasets["train"]["program"]


    # top 100 matches
    all_program_embeddings = []
    all_io_embeddings = []

    for program, io in tqdm(train_loader):
        for k in program:
            program[k] = program[k].to(device)
        for k in io:
            io[k] = io[k].to(device)
        program_embedding = model.get_program_embeds(**program)
        io_embedding = model.get_io_embeds(**io)
        all_io_embeddings.append(io_embedding)
        all_program_embeddings.append(program_embedding)

    all_io_embeddings = torch.cat(all_io_embeddings, dim=0)
    all_program_embeddings = torch.cat(all_program_embeddings, dim=0)

    K = 5

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(config.projection_dim)
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(all_program_embeddings.cpu().numpy())
    _, top_k_indices = index.search(all_io_embeddings.cpu().numpy(), K)

    random_program_pairs = np.random.choice(programs, (len(all_io_embeddings) * K, 2))


    clip_program_pairs = []


    for i, indices in tqdm(enumerate(top_k_indices)):
        program = processed_datasets["train"][i]["program"]
        for index in indices:
            if index == i: # skip same program
                continue
            program_index = processed_datasets["train"][int(index)]["program"]
            clip_program_pairs.append((program, program_index))

    # print(len(random_program_pairs))
    # print(len(clip_program_pairs))

    def hit_rate(results):
        count = 0
        for input_str, _ in results:
            if len(input_str) > 0:
                count += 1
        return count / len(results)

    num_sample = data_args.max_train_sample

    with mp.Pool(20) as pool:
        all_clip_results = pool.map(string_generator.generate_consensus_sample, tqdm(clip_program_pairs[:num_sample]))
        all_random_results = pool.map(string_generator.generate_consensus_sample, tqdm(random_program_pairs[:num_sample]))
    
    print(hit_rate(all_random_results))
    print(hit_rate(all_clip_results))

    def generate_consensus_classification_dataset(results, program_pairs, output_dir, dataset_name):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, dataset_name)
        program1 = []
        program2 = []
        label = []
        for (input_str, _), (p1, p2) in zip(results, program_pairs):
            if len(input_str) > 0:
                program1.append(p1)
                program2.append(p2)
                label.append(1)
            else:
                program1.append(p1)
                program2.append(p2)
                label.append(0)
        dataset = Dataset.from_dict({
            "program1": program1,
            "program2": program2,
            "label": label,
        })
        ds = dataset.train_test_split(test_size=0.1, seed=42)
        ds = DatasetDict({
            "train": ds["train"],
            "test": ds["test"],
        })
        ds.save_to_disk(output_path)

    generate_consensus_classification_dataset(all_clip_results, clip_program_pairs[:num_sample], cfg.output_dir, "clip")
    generate_consensus_classification_dataset(all_random_results, random_program_pairs[:num_sample], cfg.output_dir, "random")
