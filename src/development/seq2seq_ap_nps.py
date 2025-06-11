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
from transformers import get_scheduler, set_seed
from transformers.optimization import Adafactor

import wandb
from datasets import DatasetDict, Dataset
from hprl_karel_env import KarelEvalParallel, KarelDemoEvalParallel
from src.base_classes import BaseDataset
from src.datacollator import KarelDemoDataCollator
from src.seq2seq_dataset import DATASET_MAP
from src.seq2seq_utils import MODEL_MAP
from string_trans import StringTransEval
from functools import partial
from transformers import T5Config, AutoTokenizer
from src.modeling_karel import KarelDemoSeq2seqModel
from src.seq2seq_dataset import KarelDemoDataset
from src.development.modeling_so_fun import KarelSoFun
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer, T5Config
from transformers.file_utils import PaddingStrategy

logger = logging.getLogger(__name__)

@dataclass
class SoFunDataCollator:
    # Karel input sequence output sequence and paired input output share same data collator
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def pad_program(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            verbose=False,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # batch input_output_states reshaping [batch_size, [num_demo|num_states], height, width, channel]
        # -> [batch_size, [num_demo|num_states], channel, height, width]
        input_output_states = pad_sequence([example["input_output_states"] for example in examples], batch_first=True)
        attention_mask = pad_sequence([example["attention_mask"] for example in examples], batch_first=True)
        actions = pad_sequence([example["actions"] for example in examples], batch_first=True, padding_value=-100)
        perceptions = pad_sequence([example["perceptions"] for example in examples], batch_first=True)
        perception_mask = pad_sequence([example["perception_mask"] for example in examples], batch_first=True)
        program = [{"input_ids": example["program"]} for example in examples]
        program = self.pad_program(program)
        labels = program["input_ids"]
        labels[labels == 0] = -100
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
            labels=labels)
        return {
            "input_output_states": input_output_states,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "actions": actions,
            "perceptions": perceptions,
            "perception_mask": perception_mask,
        }

@dataclass
class AgentDataCollator:
    # Karel input sequence output sequence and paired input output share same data collator
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"


    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # batch input_output_states reshaping [batch_size, [num_demo|num_states], height, width, channel]
        # -> [batch_size, [num_demo|num_states], channel, height, width]
        input_output_states = pad_sequence([example["input_output_states"] for example in examples], batch_first=True)
        attention_mask = pad_sequence([example["attention_mask"] for example in examples], batch_first=True)
        actions = pad_sequence([example["actions"] for example in examples], batch_first=True, padding_value=-100)
        perceptions = pad_sequence([example["perceptions"] for example in examples], batch_first=True)
        perception_mask = pad_sequence([example["perception_mask"] for example in examples], batch_first=True)
        return {
            "input_output_states": input_output_states,
            "attention_mask": attention_mask,
            "actions": actions,
            "perceptions": perceptions,
            "perception_mask": perception_mask,
        }

def evaluate_single_program(predict_program_and_index, config, evaluator: Union[StringTransEval, KarelEvalParallel, KarelDemoEvalParallel], dataset):
    predict_program, index = predict_program_and_index
    data = dataset[index]
    inputs = data[config.data.input_column]
    outputs = data[config.data.output_column]
    program_score = evaluator.eval_single_program(
        predict_program, inputs, outputs)
    output_score = None
    output_score = evaluator.eval_input_output(predict_program, inputs, outputs)
    sample_score = program_score == 1
    return program_score, sample_score, output_score


def get_karel_io_model(model_args):
    # config
    if model_args.config_name:
        config = T5Config.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = T5Config.from_pretrained(model_args.model_name_or_path)
    else:
        config = T5Config()

    # tokenizer
    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer)
    config.num_channels = model_args.num_channels
    config.encoder_type = model_args.encoder_type
    model = KarelSoFun.from_pretrained(model_args.model_name_or_path, config=config)
    return config, tokenizer, tokenizer, model

class KarelDemoSoFunDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer=target_tokenzier
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        state = torch.tensor(self.dataset[0][self.input_column][0][0])
        self.end_state = torch.zeros_like(state).unsqueeze(0)

    def __getitem__(self, index):
        data = self.dataset[index]
        program, inputs, inputs_length = self._get_data(data)
        action = torch.tensor(data["actions"])
        perception = torch.tensor(data["perceptions"]).float()
        # action: num_demo, seq_len - 1
        # perception: num_demo, seq_len, 5
        # 

        # encode program into ids
        program = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in program]
        indices = self.get_indices(len(inputs))
        input_states = []
        actions = []
        perceptions = []
        perception_mask = []
        for i in indices:
            l = inputs_length[i]
            state_sequence = torch.tensor(inputs[i][:l])
            state_sequence = torch.cat([state_sequence, self.end_state])
            input_states.append(state_sequence)
            demo_action = torch.zeros(state_sequence.shape[0], dtype=torch.long).fill_(-100)
            demo_action[:l-1] = action[i][:l - 1]
            actions.append(demo_action)
            demo_perception = torch.zeros(state_sequence.shape[0], 5, dtype=torch.float32).fill_(0)
            demo_perception[:l] = perception[i][:l]
            perceptions.append(demo_perception)
            demo_perception_mask = torch.zeros(state_sequence.shape[0], dtype=torch.long).fill_(0)
            demo_perception_mask[:l] = 1
            perception_mask.append(demo_perception_mask)
            
        
        input_states = torch.cat(input_states, dim=0).float()
        attention_mask = torch.ones(len(input_states), dtype=torch.long)
        actions = torch.cat(actions, dim=0)
        perceptions = torch.cat(perceptions, dim=0)
        perception_mask = torch.cat(perception_mask, dim=0)

        return {
            "input_output_states": input_states,
            "attention_mask": attention_mask,
            "program": program["input_ids"],
            "actions": actions,
            "perceptions": perceptions,
            "perception_mask": perception_mask,
        }

class AgentDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer=target_tokenzier
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        state = torch.tensor(self.dataset[0][self.input_column][0][0])
        self.end_state = torch.zeros_like(state).unsqueeze(0)

    def __getitem__(self, index):
        data = self.dataset[index]
        inputs = data["inputs"]
        inputs_length = data["inputs_length"]
        # program, inputs, inputs_length = self._get_data(data)
        action = torch.tensor(data["actions"])
        perception = torch.tensor(data["perceptions"]).float()
        # action: num_demo, seq_len - 1
        # perception: num_demo, seq_len, 5
        # 

        indices = self.get_indices(len(inputs))
        input_states = []
        actions = []
        perceptions = []
        perception_mask = []
        for i in indices:
            l = inputs_length[i]
            state_sequence = torch.tensor(inputs[i][:l])
            state_sequence = torch.cat([state_sequence, self.end_state])
            input_states.append(state_sequence)
            demo_action = torch.zeros(state_sequence.shape[0], dtype=torch.long).fill_(-100)
            demo_action[:l-1] = action[i][:l - 1]
            actions.append(demo_action)
            demo_perception = torch.zeros(state_sequence.shape[0], 5, dtype=torch.float32).fill_(0)
            demo_perception[:l] = perception[i][:l]
            perceptions.append(demo_perception)
            demo_perception_mask = torch.zeros(state_sequence.shape[0], dtype=torch.long).fill_(0)
            demo_perception_mask[:l] = 1
            perception_mask.append(demo_perception_mask)
            
        
        input_states = torch.cat(input_states, dim=0).float()
        attention_mask = torch.ones(len(input_states), dtype=torch.long)
        actions = torch.cat(actions, dim=0)
        perceptions = torch.cat(perceptions, dim=0)
        perception_mask = torch.cat(perception_mask, dim=0)

        return {
            "input_output_states": input_states,
            "attention_mask": attention_mask,
            "actions": actions,
            "perceptions": perceptions,
            "perception_mask": perception_mask,
        }

def seq2seq_nps(cfg: DictConfig):
    trainer_args = cfg.trainer
    data_args = cfg.data
    model_args = cfg.model
    logger_args = cfg.logger
    tuner_args = cfg.tuner



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


    if cfg.task == "agent":
        dataset = Dataset.load_from_disk(data_args.dataset_dir)
        config, target_tokenizer, tokenizer, model = get_karel_io_model(model_args)
        train_dataset: BaseDataset = AgentDataset(
            dataset=dataset, tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)
        data_collator = AgentDataCollator(
            tokenizer=tokenizer,
            model=model,
        )
        def postprocess_text(preds):
            preds = [pred.strip() for pred in preds]
            return preds

        train_dataloader = DataLoader(
            train_dataset, shuffle=False, collate_fn=data_collator, batch_size=1, num_workers=0, drop_last=False, pin_memory=True,
        )

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

        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )



    else:
        raw_datasets = DatasetDict.load_from_disk(data_args.dataset_dir)

        # Load pretrained model and tokenizer
        config, target_tokenizer, tokenizer, model = get_karel_io_model(model_args)

        model.gradient_checkpointing_enable()

        # Preprocessing the datasets.
        # First we tokenize all the texts. Now processed dataset is raw dataset
        processed_datasets = raw_datasets

        data_class = KarelDemoDataset

        train_dataset: BaseDataset = KarelDemoSoFunDataset(
            dataset=processed_datasets["train"], tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)
        train_dataset_for_eval: BaseDataset = data_class(
            dataset=processed_datasets["train"], tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)
        eval_dataset: BaseDataset = data_class(
            dataset=processed_datasets["val"], tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)
        test_dataset: BaseDataset = data_class(
            dataset=processed_datasets["test"], tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)

        if data_args.max_train_sample is not None:
            train_dataset.set_max_train_sample(data_args.max_train_sample)
            
        train_dataset_for_eval.set_max_train_sample(5000)

        # Log a few random samples from the training set:
        if not cfg.autotune:
            for index in random.sample(range(len(train_dataset)), 1):
                logger.info(
                    f"Sample {index} of the training set: {train_dataset[index]}.")

        tokenizer.truncation = True
        test_data_collator = KarelDemoDataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8,
        )
        data_collator = SoFunDataCollator(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8,
        )



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
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_output_states"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

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
            evaluator = KarelDemoEvalParallel()

            all_result = {"global_steps": current_steps}
            for set_name, data in results.items():
                predictions = data["predictions"]
                references = data["references"]
                split = set_name if set_name != "eval" else "val"
                raw_eval_dataset = raw_datasets[split]
                score = []
                sample_score = []
                exact_match = []
                match_output = []


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

    if not cfg.task == "agent":
        all_loader = [train_dataloader_for_eval, eval_dataloader, test_dataloader]
        split = ["train", "eval", "test"]
    for epoch in range(trainer_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if cfg.task == "agent":
                outputs = model.forward_so_fun(**batch)
                action_loss = outputs.action_loss
                perception_loss = outputs.perception_loss
                loss = action_loss
                accelerator.backward(loss)
                # ipdb.set_trace()
                # accelerator.clip_grad_norm_(
                #     model.parameters(), trainer_args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                action_acc = outputs.action_acc
                perception_acc = outputs.perception_acc
                print(action_acc.item(), perception_acc)
                ipdb.set_trace()
                if logger_args.log_to == "wandb" and not cfg.debug and not cfg.autotune and accelerator.is_main_process:
                    wandb.log({"action_loss": action_loss.item()}, step=completed_steps)
                    wandb.log({"perception_loss": perception_loss.item()}, step=completed_steps)

                    wandb.log({"action_acc": action_acc.item()}, step=completed_steps)
                    wandb.log({"perception_acc": perception_acc.item()}, step=completed_steps)
            else:
                outputs = model(**batch)

                loss = outputs.loss
                action_loss = outputs.action_loss
                perception_loss = outputs.perception_loss
                total_loss = loss + action_loss + perception_loss
                loss = total_loss / trainer_args.gradient_accumulation_steps
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
                    if logger_args.log_to == "wandb" and (completed_steps % 100 == 0 or cfg.task == "agent" ) and not cfg.debug and not cfg.autotune and accelerator.is_main_process:
                        wandb.log({"loss": loss.item()}, step=completed_steps)
                        wandb.log({"action_loss": action_loss.item()}, step=completed_steps)
                        wandb.log({"perception_loss": perception_loss.item()}, step=completed_steps)
                        action_acc = outputs.action_acc
                        perception_acc = outputs.perception_acc
                        wandb.log({"action_acc": action_acc.item()}, step=completed_steps)
                        wandb.log({"perception_acc": perception_acc.item()}, step=completed_steps)

            if completed_steps >= trainer_args.max_train_steps:
                break

        # torch.cuda.empty_cache()
        if (epoch + 1) % cfg.logger.log_freq == 0 and not cfg.task == "agent":
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

    if not cfg.debug and not cfg.autotune and accelerator.is_main_process:
        wandb.finish()    

    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            cfg.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            target_tokenizer.save_pretrained(cfg.output_dir)
