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


from datasets import DatasetDict
from omegaconf import DictConfig
from src.pretrain_utils import (to_multiple_of_eight,
                                get_model, get_bert_model, preprocess_nps)
from transformers import set_seed
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@torch.no_grad()
def clip_embedding(cfg: DictConfig):
    trainer_args = cfg.trainer
    data_args = cfg.data
    model_args = cfg.model
    logger_args = cfg.logger


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
    model = model.to(device)

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


    # train_dataset = processed_datasets["train"]
    # eval_dataset = processed_datasets["val"]
    # test_dataset = processed_datasets["test"]

    io_max_length = to_multiple_of_eight(data_args.max_input_length + data_args.max_output_length + 2)

    def get_io_samples(dataset):
        all_inputs = []
        all_outputs = []
        label = []
        for i in range(data_args.max_train_sample):
            data = dataset[i]
            inputs = data["inputs"]
            outputs = data["outputs"]
            label.extend([i] * len(inputs))
            all_inputs.extend(inputs)
            all_outputs.extend(outputs)


        batch_inputs = io_tokenizer.batch_encode_plus(all_inputs, all_outputs, padding="max_length", max_length=io_max_length, return_tensors="pt")
        label = np.array(label)
        return batch_inputs, label

    for set_name in ["train", "val", "test"]:

        batch_inputs, label = get_io_samples(processed_datasets[set_name])
        for k in batch_inputs:
            batch_inputs[k] = batch_inputs[k].to(device)

        outputs = model.get_io_states(**batch_inputs).cpu().numpy()
        # features = PCA(n_components=2).fit_transform(outputs)
        features = TSNE().fit_transform(outputs)
        # ipdb.set_trace()

        for i in range(data_args.max_train_sample):
            sample_features = features[label == i]
            plt.scatter(sample_features[:, 0], sample_features[:, 1])

        plt.show()
        plt.cla()
        plt.clf()
