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

import datasets
import editdistance as ed
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq, get_scheduler, set_seed
from transformers.optimization import Adafactor

import wandb
from src.seq2seq_utils import get_base_model
from src.utils import load_disk_dataset, CharNPE, GenerateSampleOnFlyNPE

logger = logging.getLogger(__name__)


def seq2seq_npe(cfg: DictConfig):
    trainer_args = cfg.trainer
    data_args = cfg.data
    model_args = cfg.model
    logger_args = cfg.logger

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
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    accelerator.wait_for_everyone()


    raw_datasets = DatasetDict.load_from_disk(data_args.dataset_dir)

    # Load pretrained model and tokenizer
    config, tokenizer, model = get_base_model(model_args)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    preprocess_function = partial(load_disk_dataset, cfg=cfg)

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.max_train_sample is not None:
        train_dataset = Dataset.from_dict(processed_datasets["train"][:data_args.max_train_sample])
    else:
        train_dataset = processed_datasets["train"]
    if data_args.generate_on_fly:
        train_dataset = GenerateSampleOnFlyNPE(train_dataset, tokenizer=tokenizer, data_args=data_args)
    else:
        train_dataset = CharNPE(train_dataset, tokenizer=tokenizer, data_args=data_args)
    eval_dataset = CharNPE(processed_datasets["val"], tokenizer=tokenizer, data_args=data_args)
    test_dataset = CharNPE(processed_datasets["test"], tokenizer=tokenizer, data_args=data_args)


    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=trainer_args.per_device_train_batch_size
    )
    train_dataloader_for_eval = DataLoader(train_dataset, collate_fn=data_collator, batch_size=trainer_args.per_device_eval_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=trainer_args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=trainer_args.per_device_eval_batch_size)

    # Optimizer
    if trainer_args.adafactor_for_t5:
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None) # codet5-large
        # lr_scheduler = AdafactorSchedule(optimizer) # codet5-large
    else:
        # codet5-small and prompt_tuning
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=trainer_args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, train_dataloader_for_eval, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, train_dataloader_for_eval, eval_dataloader, test_dataloader
    )

    # evaluation function
    @torch.no_grad()
    def evaluate(loaders, split, output_dir, base_dataset):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        model.eval()
        if data_args.val_max_target_length is None:
            data_args.val_max_target_length = data_args.max_target_length

        all_result = {}

        for loader, set_name in zip(loaders, split):

            gen_kwargs = {
                "max_length": data_args.val_max_target_length if data_args.val_max_target_length is not None else config.max_length,
                "num_beams": data_args.num_beams,
            }
            predictions = []
            references = []

            progress_bar = tqdm(range(len(loader)), desc=f"Evaluating {set_name} set")

            for step, batch in enumerate(loader):
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not data_args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if data_args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                progress_bar.update(1)
            
            split = set_name if set_name != "eval" else "val"
            raw_eval_dataset = base_dataset[split]

            exact_match = 0
            distance = 0.0
            for pred, ref in zip(predictions, references):
                if pred == ref:
                    exact_match += 1
                if len(pred) == 0:
                    distance += ed.eval(pred, ref)
                else:
                    distance += ed.eval(pred, ref) / len(ref)

            exact_match = exact_match / len(predictions)
            distance = distance / len(predictions)
            result = {
                "exact": np.round(exact_match, 4) * 100.0,
                "distance":  np.round(distance, 4) * 100.0,
            }
            all_result[set_name] = result
            logger.info(result)

            if output_dir is not None:
                with open(os.path.join(output_dir, f"{set_name}_result.json"), "w") as f:
                    json.dump(result, f, indent=4)
                output_predictions = {}
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    data = raw_eval_dataset[i // cfg.data.num_demo]
                    sample_id = i % cfg.data.num_exec
                    inputs = data[cfg.data.input_column]
                    outputs = data[cfg.data.output_column]
                    output_predictions[i] = {
                        "program": data[cfg.data.program_column],
                        "input": inputs[sample_id],
                        "output": outputs[sample_id],
                        "prediction": pred,
                    }
                with open(os.path.join(output_dir, f"{set_name}_beam_results_{data_args.num_beams}.json"), "w") as f:
                    json.dump(output_predictions, f, ensure_ascii=False, indent=4)

        if output_dir is not None and logger_args.log_model:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)

        return all_result



    if not cfg.eval_only:
        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / trainer_args.gradient_accumulation_steps)
        if trainer_args.max_train_steps is None:
            trainer_args.max_train_steps = trainer_args.num_train_epochs * num_update_steps_per_epoch
        else:
            trainer_args.num_train_epochs = math.ceil(trainer_args.max_train_steps / num_update_steps_per_epoch)

        if trainer_args.warmup_ratio is not None:
            trainer_args.num_warmup_steps = int(trainer_args.max_train_steps * trainer_args.warmup_ratio)

        if trainer_args.adafactor_for_t5:
            pass
        else:
            lr_scheduler = get_scheduler(
                name=trainer_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=trainer_args.num_warmup_steps,
                num_training_steps=trainer_args.max_train_steps,
            )

        # Train!
        total_batch_size = trainer_args.per_device_train_batch_size * accelerator.num_processes * trainer_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {trainer_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {trainer_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {trainer_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {trainer_args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(trainer_args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        writer = None
        if logger_args.log_to == "tensorboard" or logger_args.log_to == "all":
            writer = SummaryWriter(log_dir=logger_args.logdir)
        elif logger_args.log_to == "wandb" and not cfg.debug:
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
                loss = loss / trainer_args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % trainer_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    if not trainer_args.adafactor_for_t5:
                        lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    if writer is not None:
                        writer.add_scalar("loss", loss.item(), global_step=completed_steps)
                    if logger_args.log_to == "wandb" and completed_steps % 100 == 0 and not cfg.debug:
                        wandb.log({"loss": loss.item()}, step=completed_steps)

                if completed_steps >= trainer_args.max_train_steps:
                    break

            if (epoch + 1) % cfg.logger.log_freq == 0:
                # evaluate all sets every log freq
                output_dir = os.path.join(cfg.output_dir, f"epoch-{epoch:02}-{cfg.data.execution_dataset_type}")
                all_result = evaluate(all_loader, split, output_dir, raw_datasets)
                for set_name, result in all_result.items():
                    for k, v in result.items():
                        if logger_args.log_to == "wandb" and not cfg.debug:
                            wandb.log({f"{set_name}/execution-{cfg.data.execution_dataset_type}-{k}": v}, step=completed_steps)
                        if writer is not None:
                            writer.add_scalar(f"{set_name}/execution-{cfg.data.execution_dataset_type}-{k}", v, global_step=completed_steps)
        
        if logger_args.log_to == "wandb" and not cfg.debug:
            accelerator.wait_for_everyone()
            wandb.finish()
            
        if cfg.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(cfg.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(cfg.output_dir)
    
    else:
        all_loader = [train_dataloader_for_eval, eval_dataloader, test_dataloader]
        split = ["train", "eval", "test"]
        evaluate(all_loader, split, cfg.output_dir)
