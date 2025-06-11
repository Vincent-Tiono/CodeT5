import string

import numpy as np
import torch

import ipdb
from src.base_classes import BaseDataset, to_multiple_of_eight
from string_trans.consts import ALL_CHAR
from string_trans.generator import SampleGenerator
from torch.nn.utils.rnn import pad_sequence

class StringTransDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        self.delimiter: str = data_args.delimiter
        self.synthesis_prefix: str = data_args.synthesis_prefix
        self.char2int = {}
        for c in ALL_CHAR:
            cid = self.tokenizer.encode(c)[1]
            self.char2int[c] = cid
        for c in string.punctuation:
            cid = self.tokenizer.encode(c)[1]
            self.char2int[c] = cid
        self.char2int[self.delimiter] = self.tokenizer.encode(self.delimiter)[1]
        self.sample_generator = SampleGenerator(
            num_demo=data_args.num_demo,
            programs=dataset[data_args.program_column],
            min_str_len=data_args.min_str_len,
            max_str_len=data_args.max_str_len
        )

    def _encode_string(self, inp):
        encode_sequence = []
        for c in inp:
            encode_sequence.append(self.char2int[c])
        return encode_sequence

    def warp_input_ids(self, encode_sequence):
        model_input = {
            "input_ids": encode_sequence[:self.max_source_length]
        }
        model_input = self.tokenizer.pad(
            model_input, padding=self.padding, max_length=self.max_source_length)
        return model_input

    def _encode_instr_inp(self, instruction, inp):
        input_ids = self.tokenizer.encode(instruction)
        input_ids.extend(self._encode_string(inp))
        input_ids.append(self.tokenizer.eos_token_id)
        return self.warp_input_ids(input_ids)

    def get_synthesis(self, data):
        program, inputs, outputs = self._get_data(data)
        input_str = []
        output_str = []
        indices = self.get_indices(self.num_demo)
        for index in indices:
            input_str.append(inputs[index])
            output_str.append(outputs[index])
        inp = self.delimiter.join(
            [f"{i}{self.delimiter}{o}" for i, o in zip(input_str, output_str)])
        instruction = self.synthesis_prefix
        label = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        return self._encode_instr_inp(instruction, inp), label

    def get_aug_synthesis(self, data):
        return self.get_synthesis(data)

    def __getitem__(self, index):
        data = self.dataset[index]
        model_input, label = self.get_synthesis(data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            label["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in label["input_ids"]]

        model_input["labels"] = label["input_ids"]
        return model_input


class StringMismatchDataset(StringTransDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        mismatch_index = np.random.choice(self.__len__())
        while index == mismatch_index:
            mismatch_index = np.random.choice(self.__len__())
        data = self.dataset[index]
        mismatch_data = self.dataset[mismatch_index]
        model_input, label = self.get_aug_synthesis(data)
        mismatch_input, mismatch_label = self.get_synthesis(mismatch_data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            label["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in label["input_ids"]]
            mismatch_label["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in mismatch_label["input_ids"]]

        model_input["labels"] = label["input_ids"]
        model_input["false_labels"] = mismatch_label["input_ids"]
        model_input["false_input_ids"] = mismatch_input["input_ids"]
        model_input["false_attention_mask"] = mismatch_input["attention_mask"]
        return model_input

class StringDissimilarDataset(StringTransDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        mismatch_index = np.random.choice(self.__len__())
        while index == mismatch_index:
            mismatch_index = np.random.choice(self.__len__())
        data = self.dataset[index]
        mismatch_data = {}
        program_id = int(np.random.choice(data["similar_program"]))
        mismatch_data[self.program_column] = self.dataset[program_id][self.program_column]
        mismatch_data[self.input_column] = data["similar_input"]
        mismatch_data[self.output_column] = data["similar_output"]
        model_input, label = self.get_aug_synthesis(data)
        mismatch_input, mismatch_label = self.get_synthesis(mismatch_data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            label["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in label["input_ids"]]
            mismatch_label["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in mismatch_label["input_ids"]]

        model_input["labels"] = label["input_ids"]
        model_input["false_labels"] = mismatch_label["input_ids"]
        model_input["false_input_ids"] = mismatch_input["input_ids"]
        model_input["false_attention_mask"] = mismatch_input["attention_mask"]
        return model_input

class StringAugmentation(StringTransDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def get_aug_synthesis(self, data):
        program, inputs, outputs = self._get_data(data)
        r = np.random.random()
        if r < 0.5:
            input_str, output_str = self.sample_generator.generate_synthesis_sample(
                program)
            while len(input_str) == 0:
                input_str, output_str = self.sample_generator.generate_synthesis_sample(
                    program)
        else:
            input_str = []
            output_str = []
            indices = self.get_indices(len(inputs))
            for index in indices:
                input_str.append(inputs[index])
                output_str.append(outputs[index])
        inp = self.delimiter.join(
            [f"{i}{self.delimiter}{o}" for i, o in zip(input_str, output_str)])
        instruction = self.synthesis_prefix
        label = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        return self._encode_instr_inp(instruction, inp), label

class StringMismatchAugmentedDataset(StringMismatchDataset, StringAugmentation):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)


class StringDissimilarAugmentedDataset(StringDissimilarDataset, StringAugmentation):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)


class StringTransGenerateOnFlyRandomDemoDataset(StringTransDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)

    def get_synthesis(self, data):
        program, inputs, outputs = self._get_data(data)
        r = np.random.random()
        if r < 0.5:
            input_str, output_str = self.sample_generator.generate_synthesis_sample(
                program)
            while len(input_str) == 0:
                input_str, output_str = self.sample_generator.generate_synthesis_sample(
                    program)
        else:
            input_str = []
            output_str = []
            indices = self.get_indices(len(inputs))
            for index in indices:
                input_str.append(inputs[index])
                output_str.append(outputs[index])
        inp = self.delimiter.join(
            [f"{i}{self.delimiter}{o}" for i, o in zip(input_str, output_str)])
        instruction = self.synthesis_prefix
        label = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        return self._encode_instr_inp(instruction, inp), label


class StringTransFusedDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer is not the same as target_tokenizer
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        self.max_target_length = to_multiple_of_eight(
            data_args.max_target_length)
        self.max_io_length = to_multiple_of_eight(data_args.max_io_length)
        self.sample_generator = SampleGenerator(
            num_demo=data_args.num_demo,
            programs=dataset[data_args.program_column],
            min_str_len=data_args.min_str_len,
            max_str_len=data_args.max_str_len
        )

    def get_synthesis(self, data):
        program, inputs, outputs = self._get_data(data)
        indices = self.get_indices(len(inputs))
        input_str = [inputs[i] for i in indices]
        output_str = [outputs[i] for i in indices]
        batch_inputs = self.tokenizer(
            input_str, output_str, padding=self.padding, max_length=self.max_io_length)
        labels = self.target_tokenizer(
            program, padding=self.padding, max_length=self.max_target_length)
        return batch_inputs, labels


    def get_aug_synthesis(self, data):
        program, inputs, outputs = self._get_data(data)
        r = np.random.random()
        if r < 0.5:
            input_str, output_str = self.sample_generator.generate_synthesis_sample(
                program)
            while len(input_str) == 0:
                input_str, output_str = self.sample_generator.generate_synthesis_sample(
                    program)
        else:
            input_str = []
            output_str = []
            indices = self.get_indices(len(inputs))
            for index in indices:
                input_str.append(inputs[index])
                output_str.append(outputs[index])
        batch_inputs = self.tokenizer(
            input_str, output_str, padding=self.padding, max_length=self.max_io_length)
        label = self.target_tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        return batch_inputs, label


    def __getitem__(self, index):
        data = self.dataset[index]
        batch_inputs, labels = self.get_synthesis(data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                (l if l != self.target_tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        return {
            "inputs": batch_inputs,
            "labels": labels
        }


class StringTransFusedAugmentedDataset(StringTransFusedDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        data = self.dataset[index]
        batch_inputs, labels = self.get_aug_synthesis(data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                (l if l != self.target_tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        return {
            "inputs": batch_inputs,
            "labels": labels
        }

class StringTransFusedMismatchDataset(StringTransFusedDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        data = self.dataset[index]
        mismatch_index = np.random.choice(self.__len__())
        while index == mismatch_index:
            mismatch_index = np.random.choice(self.__len__())
        mismatch_data = self.dataset[mismatch_index]
        batch_inputs, labels = self.get_synthesis(data)
        mismatch_batch_inputs, mismatch_labels = self.get_synthesis(mismatch_data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                (l if l != self.target_tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        return {
            "inputs": batch_inputs,
            "labels": labels,
            "false_inputs": mismatch_batch_inputs,
            "false_labels": mismatch_labels,
        }

class StringTransFusedMismatchAugmentedDataset(StringTransFusedDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        data = self.dataset[index]
        mismatch_index = np.random.choice(self.__len__())
        while index == mismatch_index:
            mismatch_index = np.random.choice(self.__len__())
        mismatch_data = self.dataset[mismatch_index]
        batch_inputs, labels = self.get_aug_synthesis(data)
        mismatch_batch_inputs, mismatch_labels = self.get_synthesis(mismatch_data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                (l if l != self.target_tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        return {
            "inputs": batch_inputs,
            "labels": labels,
            "false_inputs": mismatch_batch_inputs,
            "false_labels": mismatch_labels,
        }

class StringTransFusedDissimilarDataset(StringTransFusedDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        data = self.dataset[index]
        mismatch_index = np.random.choice(self.__len__())
        while index == mismatch_index:
            mismatch_index = np.random.choice(self.__len__())
        mismatch_data = {}
        program_id = int(np.random.choice(data["similar_program"]))
        mismatch_data[self.program_column] = self.dataset[program_id][self.program_column]
        mismatch_data[self.input_column] = data["similar_input"]
        mismatch_data[self.output_column] = data["similar_output"]
        batch_inputs, labels = self.get_synthesis(data)
        mismatch_batch_inputs, mismatch_labels = self.get_synthesis(mismatch_data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                (l if l != self.target_tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        return {
            "inputs": batch_inputs,
            "labels": labels,
            "false_inputs": mismatch_batch_inputs,
            "false_labels": mismatch_labels,
        }

class StringTransFusedDissimilarAugmentedDataset(StringTransFusedDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        data = self.dataset[index]
        mismatch_index = np.random.choice(self.__len__())
        while index == mismatch_index:
            mismatch_index = np.random.choice(self.__len__())
        mismatch_data = {}
        program_id = int(np.random.choice(data["similar_program"]))
        mismatch_data[self.program_column] = self.dataset[program_id][self.program_column]
        mismatch_data[self.input_column] = data["similar_input"]
        mismatch_data[self.output_column] = data["similar_output"]
        batch_inputs, labels = self.get_aug_synthesis(data)
        mismatch_batch_inputs, mismatch_labels = self.get_synthesis(mismatch_data)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                (l if l != self.target_tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        return {
            "inputs": batch_inputs,
            "labels": labels,
            "false_inputs": mismatch_batch_inputs,
            "false_labels": mismatch_labels,
        }


class KarelIODataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer=target_tokenzier
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)

    def __getitem__(self, index):
        program, inputs, outputs = self._get_data(self.dataset[index])
        # encode program into ids
        program = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in program]
        indices = self.get_indices(len(inputs))
        input_states = [torch.tensor(inputs[i]) for i in indices]
        output_states = [torch.tensor(outputs[i]) for i in indices]
        # concatenate input and output states, hope the transformer can learn the relation of input output states
        input_output_states = torch.stack(input_states + output_states).float()
        return {
            "input_output_states": input_output_states,
            "program": program["input_ids"],
        }

class KarelIOPairedDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer=target_tokenzier
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)

    def __getitem__(self, index):
        program, inputs, outputs = self._get_data(self.dataset[index])
        # encode program into ids
        program = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in program]
        indices = self.get_indices(len(inputs))
        # pair input and output states
        input_output_states = torch.stack([torch.cat([torch.tensor(inputs[i]), torch.tensor(outputs[i])], dim=-1) for i in indices]).float()
        return {
            "input_output_states": input_output_states,
            "program": program["input_ids"],
        }

class karelIOPairedMismatchDataset(KarelIOPairedDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        mismatch_index = np.random.choice(self.__len__())
        while index == mismatch_index:
            mismatch_index = np.random.choice(self.__len__())
        mismatch_data = self.dataset[mismatch_index]
        program, inputs, outputs = self._get_data(self.dataset[index])
        mismatch_program, mismatch_inputs, mismatch_outputs = self._get_data(mismatch_data)
        # encode program into ids
        program = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        mismatch_program = self.tokenizer(
            mismatch_program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in program]
            mismatch_program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in mismatch_program]
        indices = self.get_indices(len(inputs))
        # pair input and output states
        input_output_states = torch.stack([torch.cat([torch.tensor(inputs[i]), torch.tensor(outputs[i])], dim=-1) for i in indices]).float()
        mismatch_input_output_states = torch.stack([torch.cat([torch.tensor(mismatch_inputs[i]), torch.tensor(mismatch_outputs[i])], dim=-1) for i in indices]).float()
        return {
            "input_output_states": input_output_states,
            "program": program["input_ids"],
            "mismatch_input_output_states": mismatch_input_output_states,
            "mismatch_program": mismatch_program["input_ids"],
        }

class karelIOFusedDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer=target_tokenzier
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)

    def __getitem__(self, index: int):
        program, inputs, outputs = self._get_data(self.dataset[index])
        # encode program into ids
        program = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in program]
        indices = self.get_indices(len(inputs))
        input_output_states = torch.stack([torch.cat([torch.tensor(inputs[i]), torch.tensor(outputs[i])], dim=-1) for i in indices]).float()
        # input_output_states in shape [num_demo, height, width, channel]
        return {
            "input_output_states": input_output_states,
            "program": program["input_ids"],
        }


class KarelDemoDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer=target_tokenzier
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)
        state = torch.tensor(self.dataset[0][self.input_column][0][0])
        self.end_state = torch.zeros_like(state).unsqueeze(0)

    def __getitem__(self, index):
        program, inputs, inputs_length = self._get_data(self.dataset[index])
        # encode program into ids
        program = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in program]
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
            "program": program["input_ids"],
        }


class KarelDemoMismatchDataset(KarelDemoDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        super().__init__(dataset, tokenizer, target_tokenizer, data_args)

    def __getitem__(self, index):
        mismatch_index = np.random.choice(self.__len__())
        while index == mismatch_index:
            mismatch_index = np.random.choice(self.__len__())
        mismatch_data = self.dataset[mismatch_index]
        program, inputs, inputs_length = self._get_data(self.dataset[index])
        mismatch_program, mismatch_inputs, mismatch_inputs_length = self._get_data(mismatch_data)
        # encode program into ids
        program = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        mismatch_program = self.tokenizer(
            mismatch_program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in program]
            mismatch_program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in mismatch_program]
        indices = self.get_indices(len(inputs))
        input_states = []
        for i in indices:
            l = inputs_length[i]
            state_sequence = torch.tensor(inputs[i][:l])
            input_states.append(torch.cat([state_sequence, self.end_state]))
        input_states = torch.cat(input_states, dim=0).float()
        attention_mask = torch.ones(len(input_states), dtype=torch.long)
        mismatch_input_states = []
        for i in indices:
            l = mismatch_inputs_length[i]
            state_sequence = torch.tensor(mismatch_inputs[i][:l])
            mismatch_input_states.append(torch.cat([state_sequence, self.end_state]))
        mismatch_input_states = torch.cat(mismatch_input_states, dim=0).float()
        mismatch_attention_mask = torch.ones(len(mismatch_input_states), dtype=torch.long)
        return {
            "input_output_states": input_states,
            "attention_mask": attention_mask,
            "program": program["input_ids"],
            "mismatch_input_output_states": mismatch_input_states,
            "mismatch_attention_mask": mismatch_attention_mask,
            "mismatch_program": mismatch_program["input_ids"],
        }

class KarelDemoFusedDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, target_tokenizer, data_args) -> None:
        # tokenizer=target_tokenzier
        super().__init__(dataset=dataset, tokenizer=tokenizer,
                         target_tokenizer=target_tokenizer, data_args=data_args)


    def __getitem__(self, index: int):
        program, inputs, inputs_length = self._get_data(self.dataset[index])
        # encode program into ids
        program = self.tokenizer(
            program, max_length=self.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            program["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in program]
        indices = self.get_indices(len(inputs))
        input_states = []
        attention_mask = []
        for i in indices:
            l = inputs_length[i]
            state_sequence = torch.tensor(inputs[i][:l])
            input_states.append(state_sequence)
            attention_mask.append(torch.ones(l, dtype=torch.long))
        attention_mask = pad_sequence(attention_mask)
        input_states = pad_sequence(input_states)
        # input_states in (seq_length, num_demo, height, width, channel)
        # attention_mask in (seq_length, num_demo)
        # the order remain the same for data collator padding with pad_sequence
        return {
            "input_states": input_states,
            "states_padding_mask": attention_mask,
            "program": program["input_ids"],
        }


DATASET_MAP = {
    "StringTrain": StringTransGenerateOnFlyRandomDemoDataset,
    "StringTest": StringTransDataset,
    "StringMismatchTrain": StringMismatchDataset,
    "StringMismatchTest": StringTransDataset,
    "StringMismatchAugmentedTrain": StringMismatchAugmentedDataset,
    "StringMismatchAugmentedTest": StringTransDataset,
    "StringDissimilarTrain": StringDissimilarDataset,
    "StringDissimilarTest": StringTransDataset,
    "StringDissimilarAugmentedTrain": StringDissimilarAugmentedDataset,
    "StringDissimilarAugmentedTest": StringTransDataset,
    "StringFusedTrain": StringTransFusedDataset,
    "StringFusedTest": StringTransFusedDataset,
    "StringFusedAugmentedTrain": StringTransFusedAugmentedDataset,
    "StringFusedAugmentedTest": StringTransFusedDataset,
    "StringFusedMismatchTrain": StringTransFusedMismatchDataset,
    "StringFusedMismatchTest": StringTransFusedDataset,
    "StringFusedMismatchAugmentedTrain": StringTransFusedMismatchAugmentedDataset,
    "StringFusedMismatchAugmentedTest": StringTransFusedDataset,
    "StringFusedDissimilarTrain": StringTransFusedDissimilarDataset,
    "StringFusedDissimilarTest": StringTransFusedDataset,
    "StringFusedDissimilarAugmentedTrain": StringTransFusedDissimilarAugmentedDataset,
    "StringFusedDissimilarAugmentedTest": StringTransFusedDataset,
    "KarelIOTrain": KarelIODataset,
    "KarelIOTest": KarelIODataset,
    "KarelIOPairedTrain": KarelIOPairedDataset,
    "KarelIOPairedTest": KarelIOPairedDataset,
    "KarelIOMismatchTrain": karelIOPairedMismatchDataset,
    "KarelIOMismatchTest": KarelIOPairedDataset,
    "KarelIOFusedTrain": karelIOFusedDataset,
    "KarelIOFusedTest": karelIOFusedDataset,
    "KarelDemoTrain": KarelDemoDataset,
    "KarelDemoTest": KarelDemoDataset,
    "KarelDemoMismatchTrain": KarelDemoMismatchDataset,
    "KarelDemoMismatchTest": KarelDemoDataset,
    "KarelDemoFusedTrain": KarelDemoFusedDataset,
    "KarelDemoFusedTest": KarelDemoFusedDataset,
}

