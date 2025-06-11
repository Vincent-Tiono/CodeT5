from src.base_classes import BaseDataset, to_multiple_of_eight
import numpy as np
import torch
from tqdm.auto import tqdm
from functools import partial
from src.pretrain_utils import preprocess_dissimilar_program
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding
import ipdb
from time import time

class StringTransDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        self.program_max_length = to_multiple_of_eight(
            data_args.max_program_length)
        self.io_max_length = to_multiple_of_eight(
            data_args.max_input_length + data_args.max_output_length - 1)

    def __getitem__(self, index):
        data = self.dataset[index]
        program, inputs, outputs = self._get_data(data)
        io_pairs = list(zip(inputs, outputs))
        input_str, output_str = io_pairs[np.random.choice(
            np.arange(len(io_pairs)))]
        batch_inputs = self.target_tokenizer(
            program, padding=self.padding, max_length=self.program_max_length)
        batch_inputs["input_ids"] = batch_inputs["input_ids"]
        batch_inputs["attention_mask"] = batch_inputs["attention_mask"]
        io_inputs = self.tokenizer.encode_plus(
            input_str, output_str, padding=self.padding, max_length=self.io_max_length)
        batch_inputs["io_input_ids"] = io_inputs["input_ids"]
        batch_inputs["io_attention_mask"] = io_inputs["attention_mask"]
        if "token_type_ids" in io_inputs:
            batch_inputs["io_token_type_ids"] = io_inputs["token_type_ids"]
        return batch_inputs


class StringTransDissimilarProgramDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        self.program_max_length = to_multiple_of_eight(
            data_args.max_program_length)
        self.io_max_length = to_multiple_of_eight(
            data_args.max_input_length + data_args.max_output_length - 1)
        train_program = [{"input_ids": input_ids, "attention_mask": attention_mask}
                         for input_ids, attention_mask in zip(dataset["input_ids"], dataset["attention_mask"])]
        train_preprocess_function = partial(
            preprocess_dissimilar_program, tokenized_program=train_program)
        dissimilar_program = dataset["dissimilar_program"]
        self.dissimilar_program = [train_preprocess_function(
            dissimilar_program) for dissimilar_program in tqdm(dissimilar_program)]
        self.max_dissimilar = min(512, data_args.num_demo)
        self.preload = data_args.preload

        # preload data, if indexing one item takes lots of time, it will be better to preload data from disk
        if self.preload:
            self.inputs_outputs = self.dataset["inputs_outputs"]
            self.dissimilar_io = self.dataset["dissimilar_io"]
            self.input_ids = self.dataset["input_ids"]
            self.attention_mask = self.dataset["attention_mask"]

        self.program_padding = DataCollatorWithPadding(
            target_tokenizer,
            padding="max_length",
            max_length=self.program_max_length,
        )
        self.io_padding = DataCollatorWithPadding(
            tokenizer,
            padding="max_length",
            max_length=self.io_max_length,
        )

    def __getitem__(self, index):
        # get all inputs
        if self.preload:
            inputs_outputs = self.inputs_outputs[index]
            dissimilar_io = self.dissimilar_io[index]
            input_ids = self.input_ids[index]
            attention_mask = self.attention_mask[index]
        else:
            data = self.dataset[index]
            inputs_outputs = data["inputs_outputs"]
            dissimilar_io = data["dissimilar_io"]
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]

        batch_inputs = BatchEncoding()
        batch_inputs["input_ids"] = input_ids
        batch_inputs["attention_mask"] = attention_mask
        input_output = inputs_outputs[np.random.choice(len(inputs_outputs))]
        batch_inputs["io_input_ids"] = torch.tensor(
            input_output, dtype=torch.long)
        batch_inputs["io_attention_mask"] = torch.ones_like(
            batch_inputs["io_input_ids"])

        # get dissimilar inputs, outputs and programs
        dissimilar_program = self.dissimilar_program[index]
        dissimilar_program: list = np.random.choice(
            dissimilar_program, self.max_dissimilar - 1, replace=False).tolist()
        ground_truth_program = {"input_ids": input_ids,
                                "attention_mask": attention_mask}
        dissimilar_program.insert(0, ground_truth_program)
        dissimilar_program = self.program_padding(dissimilar_program)

        # get dissimilar IO pairs
        np.random.shuffle(dissimilar_io)
        dissimilar_io = dissimilar_io[:self.max_dissimilar-1]
        dissimilar_io.insert(0, batch_inputs["io_input_ids"])
        dissimilar_io = [{"input_ids": tokenized_io}
                         for tokenized_io in dissimilar_io]
        dissimilar_io = self.io_padding(dissimilar_io)
        # num_negative_sample * batch_size
        batch_inputs["dissimilar_io_input_ids"] = dissimilar_io["input_ids"]
        batch_inputs["dissimilar_io_attention_mask"] = dissimilar_io["attention_mask"]
        batch_inputs["dissimilar_program_input_ids"] = dissimilar_program["input_ids"]
        batch_inputs["dissimilar_program_attention_mask"] = dissimilar_program["attention_mask"]
        return batch_inputs


class KarelIODataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        self.program_max_length = to_multiple_of_eight(
            data_args.max_program_length)

    def __getitem__(self, index):
        # t0 = time()
        if self.preload:
            program = self.program[index]
            inputs = self.inputs[index]
            outputs = self.outputs[index]
        else:
            data = self.dataset[index]
            program, inputs, outputs = self._get_data(data)
        # t1 = time()
        # print(f"get data time: {t1-t0}")
        index = np.random.choice(len(inputs))
        # io_pairs = list(zip(inputs, outputs))
        input_state, output_state = inputs[index], outputs[index]
        batch_inputs = self.tokenizer(
            program, padding=self.padding, max_length=self.program_max_length)
        input_state = torch.tensor(input_state)
        output_state = torch.tensor(output_state)
        input_output_state = torch.cat([input_state, output_state], dim=-1)
        batch_inputs["input_output_states"] = input_output_state
        return batch_inputs

class KarelDemoDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)
        self.program_max_length = to_multiple_of_eight(
            data_args.max_program_length)

    def __getitem__(self, index):
        if self.preload:
            program = self.program[index]
            inputs = self.inputs[index]
            inputs_length = self.outputs[index]
        else:
            data = self.dataset[index]
            program, inputs, inputs_length = self._get_data(data)
        index = np.random.choice(len(inputs))
        input_state_sequence = torch.tensor(inputs[index][:inputs_length[index]])
        batch_inputs = self.tokenizer(
            program, padding=self.padding, max_length=self.program_max_length)
        batch_inputs["input_states"] = input_state_sequence
        batch_inputs["states_padding_mask"] = torch.ones(len(input_state_sequence))
        # t2 = time()
        # print(f"tokenize time: {t2-t1}")
        return batch_inputs


# TODO: add more datasets
DATASET_MAP = {
    "StringTrain": StringTransDataset,
    "StringMixTrain": StringTransDissimilarProgramDataset,
    "StringdissimilarTrain": StringTransDissimilarProgramDataset,
    "StringTest": StringTransDataset,
    "StringMixTest": StringTransDataset,
    "StringdissimilarTest": StringTransDataset,
    "KarelIOTrain": KarelIODataset,
    "KarelIOTest": KarelIODataset,
    "KarelDemoTrain": KarelDemoDataset,
    "KarelDemoTest": KarelDemoDataset,
}
