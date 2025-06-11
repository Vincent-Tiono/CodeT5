from hprl_karel_env.dsl import get_DSL_option_v2
# from hprl_karel_env.karel_option import Karel_world

from hprl_karel_env.karel_option import KarelEvaluation as Karel_world
from hprl_karel_env.dsl.dsl_parse_and_trace import parse_and_trace

import numpy as np
from time import time
import ipdb

class KarelEvalParallel:
    def __init__(self, seed=123, karel = Karel_world()) -> None:
        self.karel = karel
        self.dsl = get_DSL_option_v2(seed=seed)

    def eval_single_program(self, program, inputs, outputs):
        try:
            exe = self.dsl.parse(program)
        except RuntimeError:
            return 0
        score = 0
        for i, o in zip(inputs, outputs):
            self.karel.set_new_state(np.array(i))
            try:
                # s_h = self.dsl.run(self.karel, program)
                exe(self.karel)
                s_h = self.karel.s_h
                score += (1 - np.any(np.array(o).astype(bool) != s_h[-1]))
            except RuntimeError:
                pass
        return score / len(inputs)

class KarelEval:
    def __init__(self, seed=123, karel = Karel_world()) -> None:
        self.dsl = get_DSL_option_v2(seed=seed)
        self.karel = karel

    def eval_single_program(self, program, inputs, outputs):
        score = 0
        for i, o in zip(inputs, outputs):
            self.karel.set_new_state(np.array(i))
            try:
                s_h = self.dsl.run(self.karel, program)
                score += (1 - np.any(np.array(o).astype(bool) != s_h[-1]))
            except RuntimeError:
                pass
            self.karel.clear_history()
        return score / len(inputs)

class KarelDemoEvalParallel:
    def __init__(self, seed=123, karel = Karel_world()) -> None:
        self.dsl = get_DSL_option_v2(seed=seed)
        self.karel = karel

    def eval_single_program(self, program, inputs, inputs_length):
        exe, s_exe, record_dict = parse_and_trace(program)
        if not s_exe:
            return 0
        score = 0
        for i, l in zip(inputs, inputs_length):
            ground_truth = np.array(i[:l]).astype(bool)
            self.karel.set_new_state(np.array(i[0]))
            try:
                # s_h = self.dsl.run(self.karel, program)
                _, n, s_run = exe(self.karel, 0, record_dict, exe)
                if not s_run:
                    raise RuntimeError("Program execution timeout.")
                s_h = self.karel.s_h
                score += (1 - np.any(ground_truth != np.array(s_h)))
            except RuntimeError:
                pass
            self.karel.clear_history()
        return score / len(inputs)

    def eval_input_output(self, program, inputs, inputs_length):
        exe, s_exe, record_dict = parse_and_trace(program)
        if not s_exe:
            return 0
        for i, l in zip(inputs, inputs_length):
            self.karel.set_new_state(np.array(i[0]))
            try:
                # s_h = self.dsl.run(self.karel, program)
                _, n, s_run = exe(self.karel, 0, record_dict, exe)
                if not s_run:
                    raise RuntimeError("Program execution timeout.")
                s_h = self.karel.s_h
                ground_truth = np.array(i[l-1]).astype(bool)
                if np.any(ground_truth != np.array(s_h[-1])):
                    return 0
            except RuntimeError:
                pass
            self.karel.clear_history()
        return 1


class KarelDemoEval:
    def __init__(self, seed=123) -> None:
        self.dsl = get_DSL_option_v2(seed=seed)
        self.karel = Karel_world()

    def eval_single_program(self, program, inputs, inputs_length):
        score = 0
        for i, l in zip(inputs, inputs_length):
            self.karel.set_new_state(np.array(i[0]))
            try:
                s_h = self.dsl.run(self.karel, program)
                ground_truth = np.array(i[:l]).astype(bool)
                if len(ground_truth) != len(s_h):
                    score = 0
                else:
                    score += (1 - np.any(ground_truth != np.array(s_h)))
            except RuntimeError:
                pass
            self.karel.clear_history()
        return score / len(inputs)

    def eval_input_output(self, program, inputs, inputs_length):
        for i, l in zip(inputs, inputs_length):
            self.karel.set_new_state(np.array(i[0]))
            try:
                s_h = self.dsl.run(self.karel, program)
                ground_truth = np.array(i[l-1]).astype(bool)
                if np.any(ground_truth != np.array(s_h[-1])):
                    return 0
            except RuntimeError:
                return 0
            self.karel.clear_history()
        return 1
