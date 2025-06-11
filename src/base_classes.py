from typing import Any, Dict, Union

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from time import time
import multiprocessing as mp
from tqdm.auto import tqdm
from functools import partial

# mp.set_start_method("spawn", force=True)

logger = logging.get_logger(__name__)

from datasets import Dataset as ds


def to_multiple_of_eight(length):
    return int(np.ceil(length / 8) * 8)


def load_columns(index, dataset, columns):
    return [dataset[index][column] for column in columns]


class BaseDataset(Dataset):
    def __init__(
        self,
        dataset: ds,
        tokenizer: PreTrainedTokenizer,
        target_tokenizer: PreTrainedTokenizer,
        data_args: DictConfig,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.target_tokenizer = target_tokenizer
        self.data_args = data_args
        self.program_column = data_args.program_column
        self.input_column = data_args.input_column
        self.output_column = data_args.output_column
        self.num_demo = data_args.num_demo
        self.max_source_length: int = data_args.max_source_length
        self.max_target_length: int = data_args.max_target_length
        self.ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss
        self.padding = "max_length" if data_args.pad_to_max_length else False
        self.preload = data_args.preload
        self.dataset_size = None
        if self.preload:
            logger.info("Preloading dataset")
            with mp.Pool(mp.cpu_count()) as pool:
                columns = [self.program_column, self.input_column, self.output_column]
                all_data = pool.map(partial(load_columns, dataset=self.dataset, columns=columns), tqdm(range(len(self.dataset)), desc="Loading dataset"))
        
            self.program = []
            self.inputs = []
            self.outputs = []
            for p, i, o in all_data:
                self.program.append(p)
                self.inputs.append(i)
                self.outputs.append(o)
            # for i in tqdm(range(len(self.dataset)), desc="Loading dataset"):
            #     data = self.dataset[i]
            #     self.program.append(data[self.program_column])
            #     self.inputs.append(data[self.input_column])
            #     self.outputs.append(data[self.output_column])
            logger.info("Preloading dataset finished")

        else:
            logger.info("Not preloading dataset")

    def set_max_train_sample(self, max_train_sample):
        # self.dataset = ds.from_dict(self.dataset[:max_train_sample])
        self.dataset_size = max_train_sample

    def __len__(self) -> int:
        if self.dataset_size:
            return self.dataset_size
        return len(self.dataset)

    def get_indices(self, num_sample):
        indices = np.random.choice(num_sample, self.num_demo, replace=False)
        return indices

    def _get_data(self, data):
        program = data[self.program_column]
        inputs = data[self.input_column]
        outputs = data[self.output_column]
        return program, inputs, outputs

    def __getitem__(self, index: int) -> Union[Dict[str, Any], BatchEncoding]:
        raise NotImplementedError()
