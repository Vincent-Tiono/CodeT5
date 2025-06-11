from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass
from typing import Optional, Any, Union, List, Dict
from transformers.tokenization_utils_base import PaddingStrategy
from transformers import PreTrainedModel
from transformers import BatchEncoding, DataCollatorWithPadding
import torch
from torch.nn.utils.rnn import pad_sequence
from time import time
import numpy as np

import ipdb

# Section of datacollator
@dataclass
class StringTransFusedDataCollator:
    target_tokenizer: PreTrainedTokenizer
    tokenizer: PreTrainedTokenizer
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def pad_program(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.target_tokenizer.pad(
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

    def pad_io(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

    def __call__(self, examples: List[Dict]):
        bs = len(examples)
        num_demo = len(examples[0]["inputs"]["input_ids"])
        io = []
        program = []
        for example in examples:
            inputs = example["inputs"]
            for input_ids, attention_mask in zip(inputs["input_ids"], inputs["attention_mask"]):
                io.append({"input_ids": input_ids,
                          "attention_mask": attention_mask})
            labels = example["labels"]
            program.append({"input_ids": labels["input_ids"]})
        io = self.pad_io(io)
        io["input_ids"] = io["input_ids"].view(bs, num_demo, -1)
        io["attention_mask"] = io["attention_mask"].view(bs, num_demo, -1)
        program = self.pad_program(program)
        labels = program["input_ids"]
        labels[labels == 0] = -100
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
            labels=labels)
        return {
            "input_ids": io["input_ids"].long(),
            "attention_mask": io["attention_mask"].long(),
            "labels": labels.long(),
            "decoder_input_ids": decoder_input_ids.long(),
        }

@dataclass
class StringFusedMismatchDataCollator(StringTransFusedDataCollator):
    def encode_sample(self, examples: List[Dict], input_key, label_key):
        bs = len(examples)
        num_demo = len(examples[0][input_key]["input_ids"])
        io = []
        program = []
        for example in examples:
            inputs = example[input_key]
            for input_ids, attention_mask in zip(inputs["input_ids"], inputs["attention_mask"]):
                io.append({"input_ids": input_ids,
                          "attention_mask": attention_mask})
            labels = example[label_key]
            program.append({"input_ids": labels["input_ids"]})
        io = self.pad_io(io)
        io["input_ids"] = io["input_ids"].view(bs, num_demo, -1)
        io["attention_mask"] = io["attention_mask"].view(bs, num_demo, -1)
        program = self.pad_program(program)
        labels = program["input_ids"]
        labels[labels == 0] = -100
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
            labels=labels)
        return {
            "input_ids": io["input_ids"].long(),
            "attention_mask": io["attention_mask"].long(),
            "labels": labels.long(),
            "decoder_input_ids": decoder_input_ids.long(),
        }

    def __call__(self, examples: List[Dict]):
        real_features = self.encode_sample(examples, "inputs", "labels")
        if np.random.random() < 0.5:
            false_features = self.encode_sample(examples, "inputs", "false_labels")
        else:
            false_features = self.encode_sample(examples, "false_inputs", "labels")
        return real_features, false_features

# modified from transformers.data.data_collator.DataCollatorForSeq2Seq
@dataclass
class StringTransMismatchDataCollator:
    tokenizer: PreTrainedTokenizer
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"


    def __call__(self, features, return_tensors=None):
        real_features = []
        false_features = []
        for feature in features:
            real_features.append({
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
            })
            if np.random.random() < 0.5:
                false_features.append({
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "labels": feature["false_labels"],
                })
            else:
                false_features.append({
                    "input_ids": feature["false_input_ids"],
                    "attention_mask": feature["false_attention_mask"],
                    "labels": feature["labels"],
                })

        real_features = self.collate_batch(real_features)
        false_features = self.collate_batch(false_features)
        return real_features, false_features


    def collate_batch(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


@dataclass
class KarelIODataCollator:
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
        input_output_states = torch.stack(
            [example["input_output_states"] for example in examples]).permute(0, 1, 4, 2, 3)
        program = [{"input_ids": example["program"]} for example in examples]
        program = self.pad_program(program)
        labels = program["input_ids"]
        labels[labels == 0] = -100
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
            labels=labels)
        return {
            "input_output_states": input_output_states,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }

@dataclass
class KarelIOMismatchDataCollator(KarelIODataCollator):
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # ipdb.set_trace()
        real_features = self.encode_sample(examples, "input_output_states", "program")
        if np.random.random() < 0.5:
            false_features = self.encode_sample(examples, "mismatch_input_output_states", "program")
        else:
            false_features = self.encode_sample(examples, "input_output_states", "mismatch_program")
        return real_features, false_features

    def encode_sample(self, examples: List[Dict], input_key, label_key):
        input_output_states = torch.stack(
            [example[input_key] for example in examples]).permute(0, 1, 4, 2, 3)
        program = [{"input_ids": example[label_key]} for example in examples]
        program = self.pad_program(program)
        labels = program["input_ids"]
        labels[labels == 0] = -100
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
            labels=labels)
        return {
            "input_output_states": input_output_states,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
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
        }

@dataclass
class KarelDemoMismatchDataCollator(KarelDemoDataCollator):
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        real_features = self.encode_sample(examples, "input_output_states", "program", "attention_mask")
        if np.random.random() < 0.5:
            false_features = self.encode_sample(examples, "mismatch_input_output_states", "program", "mismatch_attention_mask")
        else:
            false_features = self.encode_sample(examples, "input_output_states", "mismatch_program", "attention_mask")
        return real_features, false_features

    def encode_sample(self, examples: List[Dict], input_key, label_key, attention_mask_key):
        input_output_states = pad_sequence([example[input_key] for example in examples], batch_first=True)
        attention_mask = pad_sequence([example[attention_mask_key] for example in examples], batch_first=True)
        program = [{"input_ids": example[label_key]} for example in examples]
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
        }

@dataclass
class KarelDemoFusedDataCollator:
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
        # input_states reshaping [seq_length, batch_size, num_demo, height, width, channel]
        # -> [batch_size, num_demo, seq_length, channel, height, width]
        input_states = pad_sequence([example["input_states"] for example in examples])
        input_states = input_states.permute(1, 2, 0, 5, 3, 4).contiguous()
        # states_padding_mask reshaoing [seq_length, batch_size, num_demo]
        # -> [batch_size, num_demo, seq_length]
        states_padding_mask = pad_sequence([example["states_padding_mask"] for example in examples])
        states_padding_mask = states_padding_mask.permute(1, 2, 0)
        program = [{"input_ids": example["program"]} for example in examples]
        program = self.pad_program(program)
        labels = program["input_ids"]
        labels[labels == 0] = -100
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
            labels=labels)
        return {
            "input_states": input_states,
            "states_padding_mask": states_padding_mask,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }


# Section of data collate function


def string_collate_fn(examples: List[BatchEncoding], io_padding: DataCollatorWithPadding, program_padding: DataCollatorWithPadding):
    io = [{"input_ids": example["io_input_ids"],
           "attention_mask": example["io_attention_mask"]} for example in examples]
    io = io_padding(io)
    program = [{"input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"]} for example in examples]
    program = program_padding(program)
    return {
        "io_input_ids": io["input_ids"].long(),
        "io_attention_mask": io["attention_mask"].long(),
        "input_ids": program["input_ids"].long(),
        "attention_mask": program["attention_mask"].long(),
        "return_loss": True,
    }


def string_mix_or_hard_collate_fn(examples, io_padding: DataCollatorWithPadding, program_padding: DataCollatorWithPadding, downsize: int):
    new_size = int(len(examples) / downsize)
    perm = torch.randperm(len(examples))[:new_size]
    if new_size < len(examples):
        similar_program_input_ids = torch.stack(
            [example["similar_program_input_ids"] for i, example in enumerate(examples) if i in perm])
        similar_program_attention_mask = torch.stack(
            [example["similar_program_attention_mask"] for i, example in enumerate(examples) if i in perm])
        similar_io_input_ids = torch.stack(
            [example["similar_io_input_ids"] for i, example in enumerate(examples) if i in perm])
        similar_io_attention_mask = torch.stack(
            [example["similar_io_attention_mask"] for i, example in enumerate(examples) if i in perm])
    else:
        similar_program_input_ids = torch.stack(
            [example["similar_program_input_ids"] for example in examples])
        similar_program_attention_mask = torch.stack(
            [example["similar_program_attention_mask"] for example in examples])
        similar_io_input_ids = torch.stack(
            [example["similar_io_input_ids"] for example in examples])
        similar_io_attention_mask = torch.stack(
            [example["similar_io_attention_mask"] for example in examples])

    # similar_io in shape (batch_size, 512, io_length)
    io = [{"input_ids": example["io_input_ids"],
           "attention_mask": example["io_attention_mask"]} for example in examples]
    io = io_padding(io)
    program = [{"input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"]} for example in examples]
    program = program_padding(program)
    return {
        "input_ids": program["input_ids"],
        "attention_mask": program["attention_mask"],
        "io_input_ids": io["input_ids"],
        "io_attention_mask": io["attention_mask"],
        "similar_program_input_ids": similar_program_input_ids,
        "similar_program_attention_mask": similar_program_attention_mask,
        "similar_io_input_ids": similar_io_input_ids,
        "similar_io_attention_mask": similar_io_attention_mask,
        "downsize_perm": perm,
    }


def karel_io_collate_fn(examples: List[BatchEncoding], program_padding: DataCollatorWithPadding):
    input_output_states = torch.stack(
        [example["input_output_states"] for example in examples]).permute(0, 3, 1, 2)
    program = [{"input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"]} for example in examples]
    program = program_padding(program)
    return {
        "input_output_states": input_output_states.float(),
        "input_ids": program["input_ids"].long(),
        "attention_mask": program["attention_mask"].long(),
    }


def karel_demo_collate_fn(examples: List[BatchEncoding], program_padding: DataCollatorWithPadding):
    t0 = time()
    input_states = pad_sequence(
        [example["input_states"] for example in examples], batch_first=True).permute(0, 1, 4, 2, 3)
    states_padding_mask = pad_sequence([example["states_padding_mask"] for example in examples], batch_first=True)
    t1 = time()
    # print(f"time for padding: {t1-t0}")
    program = [{"input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"]} for example in examples]
    program = program_padding(program)
    return {
        "input_states": input_states.float(),
        "states_padding_mask": states_padding_mask.long(),
        "input_ids": program["input_ids"].long(),
        "attention_mask": program["attention_mask"].long(),
    }
