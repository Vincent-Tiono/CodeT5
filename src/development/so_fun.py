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
import os

from typing import Union

import ipdb
import torch
import transformers
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed
from transformers.file_utils import PaddingStrategy
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from src.base_classes import BaseDataset

from transformers import T5Config, AutoTokenizer
from src.modeling_karel import KarelDemoSeq2seqModel

from PIL import Image
from pygifsicle import optimize

from hprl_karel_env.dsl import get_DSL_option_v2
from hprl_karel_env.karel_option import Karel_world

logger = logging.getLogger(__name__)

from hprl_karel_env.generator_option import KarelStateGenerator

task = "randomMaze"
state_generator = KarelStateGenerator()
state, x, y, num_wall, metadata = state_generator.generate_single_state_stair_climber()
metadata
# print(metadata)

class KarelDemoEvalParallel:
    def __init__(self, output_dir, seed=123, karel = Karel_world()) -> None:
        self.task_karel = Karel_world(env_task=task, task_definition=task, reward_diff=True)
        self.output_dir = output_dir
        self.dsl = get_DSL_option_v2(seed=seed)
        self.karel = karel

    def eval_single_program(self, program, inputs, inputs_length, actions, actions_length):
        try:
            exe = self.dsl.parse(program)
        except RuntimeError:
            print(f"Cannot parse program: {program}")
            return [], []
        score = []
        reward = []
        for traj, (inp, l, a, al) in enumerate(zip(inputs[20:], inputs_length[20:], actions[20:], actions_length[20:])):
            self.karel.set_new_state(np.array(inp[0]), metadata=metadata)
            try:
                exe(self.karel)
                correct_actions = 0
                a = a[:al]
                for i in range(max(len(a), len(self.karel.a_h))):
                    if i >= len(a) or i >= len(self.karel.a_h):
                        # correct_actions -= (max(len(actions), len(self.karel.a_h)) - i)
                        break
                    if a[i] == self.karel.a_h[i]:
                        correct_actions += 1
                    else:
                        break
                score.append(correct_actions / len(actions))
            except RuntimeError as e:
                pass
            self.karel.clear_history()

        for traj, (inp, l, a, al) in enumerate(zip(inputs[20:], inputs_length[20:], actions[20:], actions_length[20:])):
            self.task_karel.set_new_state(np.array(inp[0]), metadata=metadata)
            try:
                exe(self.task_karel)
                total_reward = np.sum(self.task_karel.r_h)
                reward.append(total_reward)
            except RuntimeError as e:
                pass
            self.karel.clear_history()

        return score, reward

    def save_gif(self, path):
        pass

class KarelDemoDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer=target_tokenzier
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        state = torch.tensor(self.dataset[0][self.input_column][0][0])
        self.end_state = torch.zeros_like(state).unsqueeze(0)

    def __getitem__(self, index):
        data = self.dataset[index]
        inputs = data[self.input_column]
        inputs_length = data[self.output_column]

        indices = self.get_indices(len(inputs))
        input_states = []
        for i in indices:
            l = inputs_length[i]
            state_sequence = torch.tensor(inputs[i][:l])
            input_states.append(torch.cat([state_sequence, self.end_state]))
        input_states = torch.cat(input_states, dim=0).float()
        attention_mask = torch.ones(len(input_states), dtype=torch.long)
        return {
            "input_output_states": input_states,
            "attention_mask": attention_mask,
        }

@dataclass
class KarelDemoDataCollator:
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
        return {
            "input_output_states": input_output_states,
            "attention_mask": attention_mask,
        }


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
    model = KarelDemoSeq2seqModel.from_pretrained(model_args.model_name_or_path, config=config)
    return config, tokenizer, tokenizer, model


def so_fun(cfg: DictConfig):
    trainer_args = cfg.trainer
    data_args = cfg.data
    model_args = cfg.model


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



    dataset = Dataset.load_from_disk(data_args.dataset_dir)

    # Load pretrained model and tokenizer
    config, target_tokenizer, tokenizer, model = get_karel_io_model(model_args)

    model = torch.compile(model)

    eval_dataset: BaseDataset = KarelDemoDataset(
        dataset=dataset, tokenizer=tokenizer, target_tokenizer=target_tokenizer, data_args=data_args)

    tokenizer.truncation = True
    test_data_collator = None

    data_collator = KarelDemoDataCollator(
        tokenizer=tokenizer,
        model=model,
    )
    if test_data_collator is None:
        test_data_collator = data_collator

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]
        return preds

    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=1, num_workers=0, drop_last=False, pin_memory=True,
    )

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    # evaluation function
    @torch.no_grad()
    def evaluate(loader, output_dir):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        model.eval()
        if data_args.val_max_target_length is None:
            data_args.val_max_target_length = data_args.max_target_length

        all_result = {}
        gen_kwargs = {
            "max_length": data_args.val_max_target_length if data_args.val_max_target_length is not None else config.max_length,
            "num_beams": data_args.num_beams,
            "num_beam_groups": data_args.num_beam_groups,
            "num_return_sequences": data_args.num_beams,
            "do_sample": data_args.do_sample,
            "top_p": data_args.top_p,
            "diversity_penalty": data_args.diversity_penalty,
            "temperature": data_args.temperature,
        }
        predictions = []
        for step, batch in enumerate(loader):
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_output_states"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=target_tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(
                generated_tokens).cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = target_tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True)
            decoded_preds = postprocess_text(decoded_preds)
            predictions.extend(decoded_preds)
        all_result["predictions"] = predictions
        return all_result

    def interact(results):
        evaluator = KarelDemoEvalParallel(cfg.output_dir)

        predictions = results["predictions"]
        data = dataset[0]
        inputs = data["inputs"]
        inputs_length = data["inputs_length"]
        actions = data["actions"]
        actions_length = data["actions_length"]
        max_reward = 0
        max_score = 0
        best_program = predictions[0]
        for program in predictions:
            # print(f"Program: {program}")
            score, reward = evaluator.eval_single_program(
                program, inputs, inputs_length, actions, actions_length)
            if len(score) > 0 or len(reward) > 0:
                # print(f"Program: {program}")

                if len(score) > 0:
                    s = np.sum(score) / 5
                    r = np.sum(reward) / 5
                    if r > max_reward:
                        max_reward = r
                        best_program = program
                        print(best_program)
                        print(f"Score: {score}")
                        print(f"Reward: {reward}")
                    if s > max_score:
                        max_score = s
                        # best_program = program
                        
                        # print(f"Score: {score}")
                        # print(f"Reward: {reward}")

        print(f"Best Program: {best_program}")
        # print(f"Max Reward: {max_reward}")
        # print(f"Max Score: {max_score}")

        return score, reward

    # evaluate all sets every log freq
    output_dir = os.path.join(cfg.output_dir)
    results = None
    # results = evaluate(eval_dataloader, output_dir)
    # all_result = interact(results)

    program = "DEF run m( WHILE c( rightIsClear c) w( move w) turnLeft move WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( move i) ELSE e( move e) w) m)"
    task_karel = Karel_world(env_task=task, task_definition=task, reward_diff=True)

    def save_gif(path, s_h):
        # create video
        frames = []
        for s in s_h:
            frames.append(Image.fromarray(np.uint8(task_karel.state2image(s=s).squeeze())))
        frames[0].save(path, save_all=True, append_images=frames[1:], loop=0)

        optimize(path)

        return

    dsl = get_DSL_option_v2(seed=123)

    data = dataset[0]
    inputs = data["inputs"]
    inputs_length = data["inputs_length"]
    actions = data["actions"]
    actions_length = data["actions_length"]

    exe = dsl.parse(program)

    reward = []
    sample_idx = 0
    for traj, (inp, l, a, al) in enumerate(zip(inputs[20:], inputs_length[20:], actions[20:], actions_length[20:])):
        task_karel.set_new_state(np.array(inp[0]), metadata=metadata)
        try:
            exe(task_karel)
            total_reward = np.sum(task_karel.r_h)
            reward.append(total_reward)
            save_gif(os.path.join(cfg.output_dir, f"sample_{sample_idx}.gif"), task_karel.s_h)
            sample_idx += 1
        except RuntimeError as e:
            pass
        task_karel.clear_history()
