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

import datetime
import json
import logging
import math
import multiprocessing as mp
import os
import random

from typing import Union

import ipdb
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq, get_scheduler, set_seed, BatchEncoding
from transformers.optimization import Adafactor

import wandb
from datasets import DatasetDict
from hprl_karel_env import KarelEvalParallel, KarelDemoEvalParallel
from src.base_classes import BaseDataset
from src.datacollator import KarelIODataCollator, StringTransFusedDataCollator, KarelDemoDataCollator, KarelDemoFusedDataCollator, StringTransMismatchDataCollator, StringFusedMismatchDataCollator, KarelIOMismatchDataCollator, KarelDemoMismatchDataCollator
from src.seq2seq_dataset import DATASET_MAP
from src.seq2seq_utils import MODEL_MAP
from string_trans import StringTransEval
from functools import partial

logger = logging.getLogger(__name__)

def evaluate_single_program(predict_program_and_index, config, evaluator: Union[StringTransEval, KarelEvalParallel, KarelDemoEvalParallel], dataset):
    predict_program, index = predict_program_and_index
    data = dataset[index]
    inputs = data[config.data.input_column]
    outputs = data[config.data.output_column]
    program_score = evaluator.eval_single_program(
        predict_program, inputs, outputs)
    output_score = None
    if config.task.startswith("KarelDemo"):
        output_score = evaluator.eval_input_output(predict_program, inputs, outputs)
    sample_score = program_score == 1
    return program_score, sample_score, output_score


def seq2seq_nps(cfg: DictConfig):
    trainer_args = cfg.trainer
    data_args = cfg.data
    model_args = cfg.model
    logger_args = cfg.logger
    tuner_args = cfg.tuner

    # setup auto-tuned hyperparameters
    if tuner_args.path is not None:
        with open(os.path.join(tuner_args.path, "autotune.json")) as f:
            hyperparameters = json.load(f)
        data_args.max_source_length = hyperparameters["max_source_length"]
        data_args.max_target_length = hyperparameters["max_target_length"]
        data_args.max_io_length = hyperparameters["max_io_length"]

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(mixed_precision=trainer_args.mixed_precision)
    # ipdb.set_trace()
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
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    accelerator.wait_for_everyone()

    raw_datasets = DatasetDict.load_from_disk(data_args.dataset_dir)

    # Load pretrained model and tokenizer
    config, target_tokenizer, tokenizer, model = MODEL_MAP[cfg.task](model_args)

    model.gradient_checkpointing_enable()

    # Preprocessing the datasets.
    # First we tokenize all the texts. Now processed dataset is raw dataset
    processed_datasets = raw_datasets

    train_dataset: BaseDataset = DATASET_MAP[f"{cfg.task}Train"](
        dataset=processed_datasets["train"], tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)
    train_dataset_for_eval: BaseDataset = DATASET_MAP[f"{cfg.task}Test"](
        dataset=processed_datasets["train"], tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)
    eval_dataset: BaseDataset = DATASET_MAP[f"{cfg.task}Test"](
        dataset=processed_datasets["val"], tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)
    test_dataset: BaseDataset = DATASET_MAP[f"{cfg.task}Test"](
        dataset=processed_datasets["test"], tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)

    if data_args.max_train_sample is not None:
        train_dataset.set_max_train_sample(data_args.max_train_sample)
        train_dataset_for_eval.set_max_train_sample(data_args.max_train_sample)

    elif cfg.autotune:
        num_eval_sample = trainer_args.per_device_eval_batch_size * tuner_args.eval_steps
        train_dataset_for_eval.set_max_train_sample(num_eval_sample)

    # Log a few random samples from the training set:
    if not cfg.autotune:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else target_tokenizer.pad_token_id
    tokenizer.truncation = True
    test_data_collator = None
    if cfg.task == "String":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )
    elif cfg.task.startswith("StringMismatch") or cfg.task.startswith("StringDissimilar"):
        data_collator = StringTransMismatchDataCollator(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )
        test_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )
    elif cfg.task == "StringFused" or cfg.task == "StringFusedAugmented":
        data_collator = StringTransFusedDataCollator(
            target_tokenizer=target_tokenizer,
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
        )
    elif cfg.task.startswith("StringFusedMismatch") or cfg.task.startswith("StringFusedDissimilar"):
        data_collator = StringFusedMismatchDataCollator(
            target_tokenizer=target_tokenizer,
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
        )
        test_data_collator = StringTransFusedDataCollator(
            target_tokenizer=target_tokenizer,
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
        )
    elif cfg.task.startswith("KarelIOMismatch") or cfg.task.startswith("KarelIODissimilar"):
        data_collator = KarelIOMismatchDataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8,
            max_length=data_args.max_target_length,
        )
        test_data_collator = KarelIODataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8,
            max_length=data_args.max_target_length,
        )
    elif cfg.task.startswith("KarelIO"):
        data_collator = KarelIODataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8,
            max_length=data_args.max_target_length,
        )
    elif cfg.task == "KarelDemo":
        data_collator = KarelDemoDataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8,
        )
    elif cfg.task == "KarelDemoMismatch" or cfg.task == "KarelDemoDissimilar":
        data_collator = KarelDemoMismatchDataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8,
        )
        test_data_collator = KarelDemoDataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8,
        )
    elif cfg.task == "KarelDemoFused":
        data_collator = KarelDemoFusedDataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
        )
    else:
        raise NotImplementedError

    if test_data_collator is None:
        test_data_collator = data_collator

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    num_workers = 16 if not cfg.debug else 0
    num_workers = 16

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=trainer_args.per_device_train_batch_size, num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    train_dataloader_for_eval = DataLoader(train_dataset_for_eval, collate_fn=test_data_collator,
                                           batch_size=trainer_args.per_device_eval_batch_size, num_workers=num_workers, drop_last=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=test_data_collator,
                                 batch_size=trainer_args.per_device_eval_batch_size, num_workers=num_workers, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=test_data_collator,
                                 batch_size=trainer_args.per_device_eval_batch_size, num_workers=num_workers, drop_last=True, pin_memory=True)

    # Optimizer
    if trainer_args.adafactor_for_t5:
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True,
                              warmup_init=True, lr=None, weight_decay=trainer_args.weight_decay)  # codet5-large
        # lr_scheduler = AdafactorSchedule(optimizer) # codet5-large
    elif trainer_args.adamw:
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
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
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=trainer_args.learning_rate)
    else:
        # codet5-small and prompt_tuning
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False,
                              lr=trainer_args.learning_rate, weight_decay=trainer_args.weight_decay)

    

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, train_dataloader_for_eval, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, train_dataloader_for_eval, eval_dataloader, test_dataloader
    )


    # evaluation function
    @torch.no_grad()
    def evaluate(loaders, split, output_dir):
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

            progress_bar = tqdm(range(len(
                loader)), desc=f"Evaluating {set_name} set", disable=not accelerator.is_local_main_process)

            for step, batch in enumerate(loader):
                if cfg.task.startswith("String"):
                    if model_args.model_type.startswith("contrastive-nps-augmented"):
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            t5_input_ids=batch["t5_input_ids"],
                            t5_attention_mask=batch["t5_attention_mask"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs,
                        )
                    else:
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs,
                        )
                elif cfg.task.startswith("KarelIO"):
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_output_states"],
                        **gen_kwargs,
                    )
                elif cfg.task == "KarelDemoFused":
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        input_states=batch["input_states"],
                        states_padding_mask=batch["states_padding_mask"],
                        **gen_kwargs,
                    )
                elif cfg.task.startswith("KarelDemo"):
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_output_states"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )
                else:
                    raise NotImplementedError

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=target_tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not data_args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(
                        batch["labels"], dim=1, pad_index=target_tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(
                    generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if data_args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels,
                                      target_tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = target_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)
                decoded_labels = target_tokenizer.batch_decode(
                    labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                progress_bar.update(1)
            all_result[set_name] = {
                "predictions": predictions,
                "references": references,
            }
        return all_result

    def interact(results, current_steps):
        if cfg.task.startswith("String"):
            evaluator = StringTransEval()
        elif cfg.task.startswith("KarelIO"):
            evaluator = KarelEvalParallel()
        elif cfg.task.startswith("KarelDemo"):
            evaluator = KarelDemoEvalParallel()
        else:
            raise NotImplementedError

        all_result = {"global_steps": current_steps}
        for set_name, data in results.items():
            predictions = data["predictions"]
            references = data["references"]
            split = set_name if set_name != "eval" else "val"
            raw_eval_dataset = raw_datasets[split]
            # all_data = []
            # for i in tqdm(range(len(raw_eval_dataset)), desc="Loading dataset"):
            #     all_data.append(raw_eval_dataset[i])
            score = []
            sample_score = []
            exact_match = []
            match_output = []
            # for i, predict_program in tqdm(enumerate(predictions), desc="Evaluating program synthesis"):
            #     data = raw_eval_dataset[i]
            #     inputs = data[cfg.data.input_column]
            #     outputs = data[cfg.data.output_column]
            #     exact_match.append(predict_program ==
            #                        data[cfg.data.program_column])
            #     program_score = evaluator.eval_single_program(
            #         predict_program, inputs, outputs)
            #     if cfg.task.startswith("KarelDemo"):
            #         output_score = evaluator.eval_input_output(predict_program, inputs, outputs)
            #         match_output.append(output_score)
            #     score.append(program_score)
            #     sample_score.append(program_score == 1)

            program_and_index = [(predict_program, i) for i, predict_program in enumerate(predictions)]
            program_evaluate_function = partial(evaluate_single_program, config=cfg, evaluator=evaluator, dataset=raw_eval_dataset)
            # results = []
            # for program, i in tqdm(program_and_index, desc="Evaluating program synthesis"):
            #     results.append(program_evaluate_function((program, i)))
            with mp.Pool(16) as pool:
                results = list(tqdm(pool.imap(program_evaluate_function, program_and_index), desc="Evaluating program synthesis", total=len(program_and_index)))

            for pred, ref in zip(predictions, references):
                exact_match.append(pred == ref)

            for program_score, sample, output_score in results:
                score.append(program_score)
                sample_score.append(sample)
                if cfg.task.startswith("KarelDemo"):
                    match_output.append(output_score)

            score = np.mean(score)
            exact_match = np.mean(exact_match)
            sample_score = np.mean(sample_score)
            result = {
                "exact": np.round(exact_match, 4) * 100.0,
                "score": np.round(score, 4) * 100.0,
                "sample": np.round(sample_score, 4) * 100.0,
            }
            if cfg.task.startswith("KarelDemo"):
                match_output = np.mean(match_output)
                result["output_exact"] = np.round(match_output, 4) * 100.0
            all_result[set_name] = result
            res = result.copy()
            res["set"] = set_name
            res["current_steps"] = current_steps
            logger.info(res)

            if output_dir is not None:
                with open(os.path.join(output_dir, f"{set_name}_result.json"), "w") as f:
                    json.dump(result, f, indent=4)
                output_predictions = {}
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    output_predictions[i] = {
                        "prediction": pred,
                        "reference": ref,
                    }
                with open(os.path.join(output_dir, f"{set_name}_beam_results_{data_args.num_beams}.json"), "w") as f:
                    json.dump(output_predictions, f,
                              ensure_ascii=False, indent=4)

        if output_dir is not None and logger_args.log_model:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                target_tokenizer.save_pretrained(output_dir)

        return all_result


    def concurrent_interact(results, current_step, result_queue: mp.Queue):
        result = interact(results, current_step)
        result_queue.put(result)

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

    if trainer_args.adafactor_for_t5:
        pass
    else:
        lr_scheduler = get_scheduler(
            name=trainer_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=trainer_args.num_warmup_steps,
            num_training_steps=trainer_args.max_train_steps,
        )

    if cfg.autotune:
        if tuner_args.padding:
            import matplotlib.pyplot as plt
            import seaborn as sns
            source_length = 0
            io_length = 0
            all_io_length = []
            all_source_length = []
            target_length = 0
            progress_bar = tqdm(range(len(train_dataloader)),
                                desc="Autotune padding")
            for batch in train_dataloader:
                if model_args.model_type.startswith("contrastive"):
                    pad_token_id = tokenizer.pad_token_id
                    batch_length = torch.sum(
                        batch["input_ids"] != pad_token_id, dim=-1).max().item()
                    # print(batch_length)
                    all_io_length.append(batch_length)
                    io_length = max(io_length, batch_length)
                    if model_args.model_type.startswith("contrastive-nps-augmented"):
                        source_length = max(source_length, torch.sum(
                            batch["t5_input_ids"] != tokenizer.pad_token_id, dim=-1).max().item())
                else:
                    batch_length = torch.sum(
                        batch["input_ids"] != tokenizer.pad_token_id, dim=-1).max().item()
                    all_source_length.append(batch_length)
                    source_length = max(source_length, torch.sum(
                        batch["input_ids"] != tokenizer.pad_token_id, dim=-1).max().item())
                target_length = max(target_length, torch.sum(
                    batch["labels"] != -100, dim=-1).max().item())
                progress_bar.update(1)

            if len(all_io_length) != 0:
                if tuner_args.cutoff_rate > 0:
                    # truncation cutoff_rate of input-output pair
                    io_length = np.sort(all_io_length)[int(
                        len(all_io_length) * (1 - tuner_args.cutoff_rate))]

                sns.histplot(all_io_length, kde=True, stat="density")
                plt.xlabel("IO pairs length")
                plt.savefig(os.path.join(cfg.output_dir, "source.png"))

            if len(all_source_length) != 0:
                if tuner_args.cutoff_rate > 0:
                    # truncation cutoff_rate of input-output pairs
                    source_length = np.sort(all_source_length)[int(
                        len(all_source_length) * (1 - tuner_args.cutoff_rate))]

                sns.histplot(all_source_length, kde=True, stat="density")
                plt.xlabel("IO pairs length")
                plt.savefig(os.path.join(cfg.output_dir, "source.png"))

            source_length = math.ceil(source_length / 8) * 8
            target_length = math.ceil(target_length / 8) * 8
            io_length = math.ceil(io_length / 8) * 8
            return {
                "source_length": source_length,
                "target_length": target_length,
                "io_length": io_length,
            }

        start = torch.cuda.Event(enable_timing=True)
        one_fourth = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        eval_start = torch.cuda.Event(enable_timing=True)
        eval_end = torch.cuda.Event(enable_timing=True)
        progress_bar = tqdm(range(tuner_args.train_steps),
                            desc="Recording training speed")
        if tuner_args.compile:
            model = torch.compile(model)
            torch._dynamo.config.log_level = logging.ERROR

        try:
            completed_steps = 0
            start.record()
            for step, batch in enumerate(train_dataloader):

                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / trainer_args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % trainer_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    accelerator.clip_grad_norm_(
                        model.parameters(), trainer_args.max_grad_norm)
                    optimizer.step()
                    if not trainer_args.adafactor_for_t5:
                        lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    if completed_steps == tuner_args.train_steps // 4:
                        one_fourth.record()

                if completed_steps >= tuner_args.train_steps:
                    break
            end.record()

            eval_start.record()
            evaluate([train_dataloader_for_eval],
                     split=["train"], output_dir=None)
            eval_end.record()
            torch.cuda.synchronize()
            total_eval_steps = (len(train_dataset) // trainer_args.per_device_eval_batch_size) + \
                len(eval_dataloader) + len(test_dataloader)
            total_eval_steps = total_eval_steps * \
                (trainer_args.num_train_epochs // logger_args.log_freq)
            estimated_eval_time = math.ceil(
                total_eval_steps * (eval_start.elapsed_time(eval_end) / 1000 / tuner_args.eval_steps))
            estimated_eval_time = datetime.timedelta(
                seconds=estimated_eval_time)
            trial_duration = start.elapsed_time(end) / 1000
            step_time = one_fourth.elapsed_time(
                end) / 1000 / (tuner_args.train_steps - tuner_args.train_steps // 4)
            estimated_training_time = math.ceil(
                (trainer_args.max_train_steps - tuner_args.train_steps) * step_time + trial_duration)
            estimated_training_time = datetime.timedelta(
                seconds=estimated_training_time)
            logger.info(f"Estimated training time: {estimated_training_time}")
            logger.info(f"Estimated evaluation time: {estimated_eval_time}")
            return {"train_time": estimated_training_time, "eval_time": estimated_eval_time}
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return {"OOM": True}
            else:
                raise e

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

    if logger_args.log_to == "wandb" and not cfg.debug and not cfg.autotune and accelerator.is_main_process:
        wandb.init(project=logger_args.project, name=logger_args.logdir, config={
            "lr": trainer_args.learning_rate,
            "epoch": trainer_args.num_train_epochs,
            "warmup_steps": trainer_args.num_warmup_steps,
        })
        wandb.watch(model, log="all")
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        wandb.define_metric("test/step")
        wandb.define_metric("test/*", step_metric="test/step")
    
    # result_queue = mp.Queue()

    model: torch.nn.Module = model

    all_loader = [train_dataloader_for_eval, eval_dataloader, test_dataloader]
    detached_gradients = None
    gradient_place_holder = None
    split = ["train", "eval", "test"]
    for epoch in range(trainer_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if trainer_args.mismatch:
                batch, mismatch_batch = batch
                if detached_gradients is not None:
                    gradient_place_holder = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
                    for batch_gradient, dg in zip(gradient_place_holder, detached_gradients):
                        dg.detach().add_(batch_gradient)
                    gradient_place_holder = None
                mismatch_outputs = model(**mismatch_batch)
                outputs = model(**batch)
                mismatch_loss = - mismatch_outputs.loss * trainer_args.mismatch_lambda / trainer_args.gradient_accumulation_steps
                accelerator.backward(mismatch_loss)
                # clip mismatch gradient this may only works with single GPU
                torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_args.mismatch_grad_norm, norm_type=trainer_args.mismatch_norm)
                if detached_gradients is None:
                    detached_gradients = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
                else:
                    for g, dg in zip([p.grad.detach().clone() for p in model.parameters() if p.grad is not None], detached_gradients):
                        dg.detach().add_(g)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.detach().zero_()
                loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss
            loss = loss / trainer_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % trainer_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                accelerator.clip_grad_norm_(
                    model.parameters(), trainer_args.max_grad_norm)
                if trainer_args.mismatch:
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    for g, dg in zip(gradients, detached_gradients):
                        g.detach().add_(dg)
                    detached_gradients = None
                optimizer.step()
                if not trainer_args.adafactor_for_t5:
                    lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if logger_args.log_to == "wandb" and completed_steps % 100 == 0 and not cfg.debug and not cfg.autotune and accelerator.is_main_process:
                    wandb.log({"loss": loss.item()}, step=completed_steps)
                    if "accuracy" in outputs:
                        wandb.log({"accuracy": outputs.accuracy.item()},
                                  step=completed_steps)
                    if trainer_args.mismatch:
                        wandb.log({"mismatch_loss": mismatch_loss.item() / trainer_args.mismatch_lambda}, step=completed_steps)

            if completed_steps >= trainer_args.max_train_steps:
                break

        # torch.cuda.empty_cache()
        if (epoch + 1) % cfg.logger.log_freq == 0:
            # evaluate all sets every log freq
            output_dir = os.path.join(cfg.output_dir, f"epoch-{epoch:02}")
            results = evaluate(all_loader, split, output_dir)
            all_result = interact(results, completed_steps)
            if logger_args.log_to == "wandb" and not cfg.debug and not cfg.autotune and accelerator.is_main_process:
                current_steps = all_result.pop("global_steps")
                for set_name, result in all_result.items():
                    step_result = {}
                    for k, v in result.items():
                        step_result[f"{set_name}/synthesis-{k}"] = v
                    step_result[f"{set_name}/step"] = current_steps
                    wandb.log(step_result)

            # if cfg.debug:
            #     interact(results, completed_steps)
            # else:
            #     concurrent_process = mp.Process(target=concurrent_interact, args=(
            #         results, completed_steps, result_queue))
            #     concurrent_process.start()

    # if not cfg.debug:
    #     concurrent_process.join()

    # if logger_args.log_to == "wandb" and not cfg.debug and not cfg.autotune and accelerator.is_main_process:
    #     while not result_queue.empty():
    #         all_result = result_queue.get()
    #         current_steps = all_result.pop("global_steps")
    #         for set_name, result in all_result.items():
    #             step_result = {}
    #             for k, v in result.items():
    #                 step_result[f"{set_name}/synthesis-{k}"] = v
    #             step_result[f"{set_name}/step"] = current_steps
    #             wandb.log(step_result)
    #     accelerator.wait_for_everyone()
    #     wandb.finish()

    if not cfg.debug and not cfg.autotune and accelerator.is_main_process:
        wandb.finish()    

    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            cfg.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            target_tokenizer.save_pretrained(cfg.output_dir)
