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

from datasets import DatasetDict, concatenate_datasets
from omegaconf import DictConfig
from src.pretrain_utils import (to_multiple_of_eight,
                                get_model, get_bert_model, preprocess_nps)
from transformers import set_seed, DataCollatorWithPadding
from string_trans.generator import SampleGenerator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from src.pretrain_utils import PretrainDataset, PretrainAugDataset
from src.modeling_program_io import ContrastiveNPSModel
from tqdm.auto import tqdm
import faiss
import os
from time import time
import seaborn as sns

logger = logging.getLogger(__name__)


@torch.no_grad()
def clip_top_k(cfg: DictConfig):
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
    device_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    model = model.to(device)

    # top 100 matches
    all_program_embeddings = []
    all_io_embeddings = []

    num_train = len(train_dataset)
    num_eval = len(eval_dataset)
    num_test = len(test_dataset)

    for program, io in tqdm(whole_loader):
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

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(config.projection_dim)
    index = faiss.index_cpu_to_gpu(res, device_id, index)
    index.add(all_program_embeddings.cpu().numpy())
    _, top_k_indices = index.search(all_io_embeddings.cpu().numpy(), 100)

    top_k = []
    for i, indices in tqdm(enumerate(top_k_indices)):
        top_k.append(i in indices)
    print(np.mean(top_k[:num_train]))
    print(np.mean(top_k[num_train:num_train+num_eval]))
    print(np.mean(top_k[num_train+num_eval:]))

    # top k curve
    # model = ContrastiveNPSModel.from_pretrained(model_args.model_name_or_path).to(device)
    # all_program_embeddings = []
    # all_io_embeddings = []
    # for program, io in tqdm(train_loader):
    #     for k in program:
    #         program[k] = program[k].to(device)
    #     for k in io:
    #         io[k] = io[k].to(device)
    #     program_embedding = model.get_program_embeds(**program)
    #     io_embedding = model.get_io_embeds(**io)
    #     all_io_embeddings.append(io_embedding)
    #     all_program_embeddings.append(program_embedding)

    # all_io_embeddings = torch.cat(all_io_embeddings, dim=0)
    # all_program_embeddings = torch.cat(all_program_embeddings, dim=0)
    # top_k = np.zeros((len(all_io_embeddings), K)).astype(bool)
    # for i, embedding in enumerate(all_io_embeddings):
    #     res = torch.topk(torch.sum(all_program_embeddings * embedding, dim=-1), k=K)
    #     position = torch.where(res.indices == i)[0].tolist()
    #     if len(position) > 0:
    #         top_k[i, position[0]:] = True

    # top_k = np.mean(top_k, axis=0)
    # plt.plot(np.arange(K) + 1, top_k * 100.0)
    # plt.title("Top k of train set")
    # plt.xlabel("K")
    # plt.ylabel("Accuracy")
    # plt.savefig("images/top_k.png")

    # faiss comparison
    # model = ContrastiveNPSModel.from_pretrained(model_args.model_name_or_path).to(device)
    # all_program_embeddings = []
    # all_io_embeddings = []
    # for program, io in tqdm(train_loader):
    #     for k in program:
    #         program[k] = program[k].to(device)
    #     for k in io:
    #         io[k] = io[k].to(device)
    #     program_embedding = model.get_program_embeds(**program)
    #     io_embedding = model.get_io_embeds(**io)
    #     all_io_embeddings.append(io_embedding)
    #     all_program_embeddings.append(program_embedding)

    # all_io_embeddings = torch.cat(all_io_embeddings, dim=0)
    # all_program_embeddings = torch.cat(all_program_embeddings, dim=0)

    # K = 1000
    # brute_force_gpu = []
    # faiss_cpu = []
    # faiss_gpu = []

    # device_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

    # k = [5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000]
    # for K in tqdm(k):
    #     # brute force gpu
    #     t = time()
    #     for embedding in all_io_embeddings:
    #         res = torch.topk(torch.sum(all_program_embeddings * embedding, dim=-1), k=K)
    #     brute_force_gpu.append(time() - t)
        
    #     # faiss cpu
    #     t = time()
    #     index = faiss.IndexFlatIP(config.projection_dim)
    #     index.add(all_program_embeddings.cpu().numpy())
    #     _, top_k_indices = index.search(all_io_embeddings.cpu().numpy(), K)
    #     faiss_cpu.append(time() - t)

    #     # faiss gpu
    #     t = time()
    #     res = faiss.StandardGpuResources()
    #     index = faiss.IndexFlatIP(config.projection_dim)
    #     index = faiss.index_cpu_to_gpu(res, device_id, index)
    #     index.add(all_program_embeddings.cpu().numpy())
    #     _, top_k_indices = index.search(all_io_embeddings.cpu().numpy(), K)
    #     faiss_gpu.append(time() - t)

    # plt.plot(k, brute_force_gpu)
    # plt.plot(k, faiss_cpu)
    # plt.plot(k, faiss_gpu)
    # plt.legend(["Brute force", "Faiss", "Faiss GPU"])
    # plt.title("Top k time usage")
    # plt.xlabel("Top k")
    # plt.ylabel("seconds")
    # plt.savefig("images/top_k_time.png")

    # average position
    # all_program_embeddings = []
    # all_io_embeddings = []
    # for program, io in tqdm(train_loader):
    #     for k in program:
    #         program[k] = program[k].to(device)
    #     for k in io:
    #         io[k] = io[k].to(device)
    #     program_embedding = model.get_program_embeds(**program)
    #     io_embedding = model.get_io_embeds(**io)
    #     all_io_embeddings.append(io_embedding)
    #     all_program_embeddings.append(program_embedding)

    # all_io_embeddings = torch.cat(all_io_embeddings, dim=0)
    # all_program_embeddings = torch.cat(all_program_embeddings, dim=0)

    # positions = torch.zeros(len(all_io_embeddings), len(all_io_embeddings))
    # for i, embedding in tqdm(enumerate(all_io_embeddings)):
    #     sim = torch.sum(all_program_embeddings * embedding, dim=-1)
    #     indices = torch.argsort(sim, descending=True).cpu()
    #     positions[i, indices] = torch.arange(len(all_io_embeddings)).float()

    # average_pos = torch.mean(positions, dim=0).numpy()
    
    # sns.distplot(average_pos)
    # plt.xlabel("Average Position")
    # plt.ylabel("Density")
    # plt.show()
