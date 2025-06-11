import ipdb
from typing import Callable
from omegaconf import DictConfig, OmegaConf
import json
import os

class AutoTuner(object):
    """
    AutoTuner is a class for tuning run time of a training script.
    """
    def __init__(self, train_script: Callable, base_args) -> None:
        self.base_args = base_args
        self.args = base_args.tuner
        self.train_script = train_script
        self.hyperparameters = {"selection": None}
        self.best_time = None

    def tune(self):
        # tune runtime for the batch size
        output_file = os.path.join( self.base_args.output_dir, "autotune.json")
        self.hyperparameters["trainer.per_device_train_batch_size"] = self.base_args.trainer.per_device_train_batch_size
        self.hyperparameters["trainer.per_device_eval_batch_size"] = self.base_args.trainer.per_device_eval_batch_size

        # tune source length and target length
        self.base_args.data.max_source_length = 1024
        self.base_args.data.max_io_length = 512
        self.base_args.data.max_target_length = 1024
        pad_to_max_length = self.base_args.data.pad_to_max_length
        self.base_args.data.pad_to_max_length = False
        self.args.padding = True
        result = self.train_script(self.base_args)
        self.args.padding = False
        self.base_args.data.pad_to_max_length = pad_to_max_length
        self.base_args.data.max_source_length = result["source_length"]
        self.base_args.data.max_target_length = result["target_length"]
        self.base_args.data.max_io_length = result["io_length"]
        self.hyperparameters["max_source_length"] = result["source_length"]
        self.hyperparameters["max_target_length"] = result["target_length"]
        self.hyperparameters["max_io_length"] = result["io_length"]

        # # test training and evaluation time in eager mode
        # self.args.compile = False
        # self.base_args.data.pad_to_max_length = False
        # result = self.train_script(self.base_args)
        # if "OOM" in result:
        #     self.hyperparameters["eager_OOM"] = True
        #     with open(output_file, "w") as f:
        #         json.dump(self.hyperparameters, f, indent=4)
        #     return
        # self.hyperparameters["eager_train_time"] = str(result["train_time"])
        # self.hyperparameters["eager_eval_time"] = str(result["eval_time"])
        # self.hyperparameters["eager_run_time"] = str(result["train_time"] + result["eval_time"])
        # self.hyperparameters["selection"] = "eager"
        # self.best_time = result["train_time"] + result["eval_time"]

        # # test training time in compiled mode
        # self.args.compile = True
        # self.base_args.data.pad_to_max_length = True
        # result = self.train_script(self.base_args)
        # if "OOM" in result:
        #     self.hyperparameters["compile_OOM"] = True
        #     with open(output_file, "w") as f:
        #         json.dump(self.hyperparameters, f, indent=4)
        #     return
        # self.hyperparameters["copmile_train_time"] = str(result["train_time"])
        # self.hyperparameters["compile_eval_time"] = str(result["eval_time"])
        # self.hyperparameters["compile_run_time"] = str(result["train_time"] + result["eval_time"])
        # if result["train_time"] + result["eval_time"] < self.best_time:
        #     self.hyperparameters["selection"] = "compile"
        #     self.best_time = result["train_time"] + result["eval_time"]

        with open(output_file, "w") as f:
            json.dump(self.hyperparameters, f, indent=4)
