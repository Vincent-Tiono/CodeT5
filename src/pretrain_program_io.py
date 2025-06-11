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

import json
import logging
import math
import os
import random
from functools import partial

import ipdb
import numpy as np
import torch
import torch._dynamo
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, get_scheduler, set_seed

import wandb
from datasets import DatasetDict
from src.base_classes import BaseDataset
from src.datacollator import (karel_io_collate_fn, string_collate_fn, karel_demo_collate_fn,
                              string_mix_or_hard_collate_fn)
from src.pretrain_dataset import DATASET_MAP
from src.pretrain_utils import (get_karel_io_model, get_string_model, get_karel_demo_model,
                                preprocess_string_mix_or_hard)

logger = logging.getLogger(__name__)


def pretrain_program_io(cfg: DictConfig):
    trainer_args = cfg.trainer
    data_args = cfg.data
    model_args = cfg.model
    logger_args = cfg.logger

    if data_args.load_info_from_dataset:
        with open(os.path.join(data_args.dataset_dir, "info.json"), "r") as f:
            info = json.load(f)
            data_args.max_program_length = info["max_program_length"]
            data_args.max_input_length = info["max_input_length"]
            data_args.max_output_length = info["max_output_length"]

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(mixed_precision=trainer_args.mixed_precision)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    model_args.use_similar_dataset = data_args.use_similar_dataset
    model_args.hard_sample_only = data_args.hard_sample_only

    if cfg.task == "String":
        config, p_tokenizer, io_tokenizer, model = get_string_model(model_args)
    elif cfg.task == "KarelIO":
        config, p_tokenizer, io_tokenizer, model = get_karel_io_model(
            model_args)
    elif cfg.task == "KarelDemo":
        config, p_tokenizer, io_tokenizer, model = get_karel_demo_model(
            model_args)
    else:
        raise NotImplementedError

    # if not cfg.debug:
    #     torch._dynamo.config.log_level = logging.ERROR
    #     model = torch.compile(model)

    raw_datasets = DatasetDict.load_from_disk(data_args.dataset_dir)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = [col for col in raw_datasets["train"].column_names if col not in [
        "program", "inputs", "outputs"]]

    if cfg.task == "String" and data_args.use_similar_dataset:
        preprocess_function = partial(
            preprocess_string_mix_or_hard, program_tokenizer=p_tokenizer, io_tokenizer=io_tokenizer)
        with accelerator.main_process_first():
            # if data_args.overwrite_cache:
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # num_proc=1,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    else:
        processed_datasets = raw_datasets

    train_dataset: BaseDataset = DATASET_MAP[f"{cfg.task}Train"](
        dataset=processed_datasets["train"], tokenizer=io_tokenizer, target_tokenizer=p_tokenizer, data_args=data_args)
    data_args["preload"] = False
    train_dataset_for_eval: BaseDataset = DATASET_MAP[f"{cfg.task}Test"](
        dataset=processed_datasets["train"], tokenizer=io_tokenizer, target_tokenizer=p_tokenizer, data_args=data_args)
    eval_dataset: BaseDataset = DATASET_MAP[f"{cfg.task}Test"](
        dataset=processed_datasets["val"], tokenizer=io_tokenizer, target_tokenizer=p_tokenizer, data_args=data_args)
    test_dataset: BaseDataset = DATASET_MAP[f"{cfg.task}Test"](
        dataset=processed_datasets["test"], tokenizer=io_tokenizer, target_tokenizer=p_tokenizer, data_args=data_args)

    if data_args.max_train_sample is not None:
        train_dataset.set_max_train_sample(data_args.max_train_sample)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    program_collator = DataCollatorWithPadding(
        p_tokenizer,
        padding=True,
        pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
    )
    if cfg.task == "String":
        io_collator = DataCollatorWithPadding(
            io_tokenizer,
            padding=True,
            pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
        )

    if cfg.task == "String":
        if data_args.use_similar_dataset:
            train_data_collate_function = partial(string_mix_or_hard_collate_fn, io_padding=io_collator,
                                                  program_padding=program_collator, downsize=data_args.hard_sample_downsize)
            data_collate_function = partial(
                string_collate_fn, io_padding=io_collator, program_padding=program_collator)
        else:
            data_collate_function = partial(
                string_collate_fn, io_padding=io_collator, program_padding=program_collator)
            train_data_collate_function = data_collate_function
    elif cfg.task == "KarelIO":
        data_collate_function = partial(
            karel_io_collate_fn, program_padding=program_collator)
        train_data_collate_function = data_collate_function
    elif cfg.task == "KarelDemo":
        data_collate_function = partial(
            karel_demo_collate_fn, program_padding=program_collator)
        train_data_collate_function = data_collate_function
    else:
        raise NotImplementedError

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=train_data_collate_function, batch_size=trainer_args.per_device_train_batch_size, num_workers=16,
    )
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=train_data_collate_function, batch_size=trainer_args.per_device_train_batch_size, num_workers=16 if not cfg.debug else 0,
    # )
    train_dataloader_for_eval = DataLoader(
        train_dataset_for_eval, collate_fn=data_collate_function, batch_size=trainer_args.per_device_eval_batch_size, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collate_function,
                                 batch_size=trainer_args.per_device_eval_batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collate_function,
                                 batch_size=trainer_args.per_device_eval_batch_size, num_workers=4)

    no_decay = ["bias", "layer_norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": trainer_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Optimizer
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=trainer_args.learning_rate, betas=(0.9, 0.98))

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, train_dataloader_for_eval, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, train_dataloader_for_eval, eval_dataloader, test_dataloader,
    )

    # io_loaders = [train_io_dataloader, eval_io_dataloader, test_io_dataloader]
    # batched_io_loaders = [train_batched_io_dataloader, eval_batched_io_dataloader, test_batched_io_dataloader]

    def get_acc(logits) -> torch.FloatTensor:
        label = torch.arange(len(logits), device=logits.device)
        acc = torch.mean((torch.argmax(logits, dim=-1) == label).float())
        return acc

    def get_similar_acc(logits) -> torch.FloatTensor:
        label = torch.zeros(len(logits), device=logits.device)
        acc = torch.mean((torch.argmax(logits, dim=-1) == label).float())
        return acc

    # evaluation function
    def evaluate(loaders, split, output_dir):
        if output_dir is not None and logger_args.log_model:
            os.makedirs(output_dir, exist_ok=True)

        model.eval()
        all_result = {}

        with torch.no_grad():
            for loader, set_name in zip(loaders, split):

                progress_bar = tqdm(range(len(loader)),
                                    desc=f"Evaluating {set_name} set")
                set_loss = []
                program_acc = []
                io_acc = []

                for step, batch in enumerate(loader):
                    outputs = model(**batch)
                    set_loss.append(outputs.loss.item())
                    acc = get_acc(outputs.logits_per_program)
                    program_acc.append(acc.item())
                    acc = get_acc(outputs.logits_per_io)
                    io_acc.append(acc.item())
                    progress_bar.update(1)

                set_loss = np.mean(set_loss)
                program_acc = np.mean(program_acc)
                io_acc = np.mean(io_acc)

                result = {
                    "loss": np.round(set_loss, 4),
                    "program-acc": np.round(program_acc, 4) * 100.0,
                    "io-acc": np.round(io_acc, 4) * 100.0,
                }
                all_result[set_name] = result
                logger.info(result)

        if output_dir is not None and logger_args.log_model:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, save_function=accelerator.save)

        return all_result

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / trainer_args.gradient_accumulation_steps)
    if trainer_args.max_train_steps is None:
        trainer_args.max_train_steps = trainer_args.num_train_epochs * \
            num_update_steps_per_epoch
    else:
        trainer_args.num_train_epochs = math.ceil(
            trainer_args.max_train_steps / num_update_steps_per_epoch)

    if trainer_args.warmup_ratio is not None:
        trainer_args.num_warmup_steps = int(
            trainer_args.max_train_steps * trainer_args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=trainer_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=trainer_args.num_warmup_steps,
        num_training_steps=trainer_args.max_train_steps,
    )

    # Train!
    total_batch_size = trainer_args.per_device_train_batch_size * \
        accelerator.num_processes * trainer_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {trainer_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {trainer_args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {trainer_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {trainer_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(trainer_args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0

    if logger_args.log_to == "wandb" and not cfg.debug:
        wandb.init(project=logger_args.project, name=logger_args.logdir, config={
            "lr": trainer_args.learning_rate,
            "epoch": trainer_args.num_train_epochs,
            "warmup_steps": trainer_args.num_warmup_steps,
        })
        wandb.watch(model, log="all")

    all_loader = [train_dataloader_for_eval, eval_dataloader, test_dataloader]
    split = ["train", "eval", "test"]

    for epoch in range(trainer_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # ipdb.set_trace()
            loss = loss / trainer_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % trainer_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                accelerator.clip_grad_norm_(
                    model.parameters(), 10000, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                progress_bar.update(1)
                completed_steps += 1
                if logger_args.log_to == "wandb" and not cfg.debug and completed_steps % 100 == 0:
                    wandb.log({"loss": loss.item()}, step=completed_steps)
                    if data_args.hard_sample_only:
                        program_acc = get_similar_acc(
                            outputs.logits_per_program)
                        io_acc = get_similar_acc(outputs.logits_per_io)
                    else:
                        program_acc = get_acc(outputs.logits_per_program)
                        io_acc = get_acc(outputs.logits_per_io)
                    wandb.log({"program_acc": program_acc.item()
                              * 100.0}, step=completed_steps)
                    wandb.log({"io_acc": io_acc.item() * 100.0},
                              step=completed_steps)
                    if data_args.use_similar_dataset and not data_args.hard_sample_only:
                        similar_program_acc = get_similar_acc(
                            outputs.similar_logits_per_program)
                        wandb.log({"similar_program_acc": similar_program_acc.item(
                        ) * 100.0}, step=completed_steps)
                        similar_io_acc = get_similar_acc(
                            outputs.similar_logits_per_io)
                        wandb.log(
                            {"similar_io_acc": similar_io_acc.item() * 100.0}, step=completed_steps)
                        clip_loss = outputs.clip_loss
                        wandb.log({"clip_loss": clip_loss.item()},
                                  step=completed_steps)
                        similar_loss = outputs.similar_loss
                        wandb.log({"similar_loss": similar_loss.item()},
                                  step=completed_steps)

            if completed_steps >= trainer_args.max_train_steps:
                break

        if (epoch + 1) % cfg.logger.log_freq == 0:
            # evaluate all sets every log freq
            output_dir = os.path.join(cfg.output_dir, f"epoch-{epoch:02}")
            all_result = evaluate(all_loader, split, output_dir)
            for set_name, result in all_result.items():
                for k, v in result.items():
                    if logger_args.log_to == "wandb" and not cfg.debug:
                        wandb.log({f"{set_name}/{k}": v}, step=completed_steps)

    if logger_args.log_to == "wandb" and not cfg.debug:
        accelerator.wait_for_everyone()
        wandb.finish()

    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            cfg.output_dir, save_function=accelerator.save)
