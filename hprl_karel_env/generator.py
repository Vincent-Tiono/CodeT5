from hprl_karel_env.karel_option import Karel_world
from hprl_karel_env.dsl import get_DSL_option_v2
from hprl_karel_env.generator_option_key2door import KarelStateGenerator
from datasets import DatasetDict
from tqdm.auto import tqdm
from functools import partial
import multiprocessing as mp
from omegaconf import DictConfig
import os
import json
import argparse
import ipdb
import numpy as np
import random
import editdistance as ed

class EditDistanceGenerator:
    def __init__(self, cfg: DictConfig) -> None:
        self.dsl = get_DSL_option_v2()
        self.cfg = cfg
        self.karel = Karel_world()
        self.karel_generator = KarelStateGenerator(seed=cfg.seed)
        self.dataset = DatasetDict.load_from_disk(cfg.input_dir)
        self.train_program = self.dataset["train"]["program"]
        self.eval_program = self.dataset["val"]["program"]
        self.test_program = self.dataset["test"]["program"]
        self.output_dir = cfg.output_dir
        self.seed(cfg.seed)

    def seed(self, seed):
        self.dsl.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

