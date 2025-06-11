import numpy as np
from string_trans.dsl import StringTransformationDSL
from string_trans.dsl_substr import SubStrDSL
from string_trans.string_env import StringTransEnv
from typing import List, Callable

DSL_MAP = {
    "base": StringTransformationDSL,
    "substr": SubStrDSL,
}


class StringTransEval:
    def __init__(self, dsl_name="base", debug=False):
        self.dsl = DSL_MAP[dsl_name](debug=debug)
        self.env = StringTransEnv(input_str="")

    def eval(self, program: str, input_str: List[str], output_str: List[str], num_demo: int):
        seen_score = self.eval_program(program, input_str[:num_demo], output_str[:num_demo])
        unseen_score = self.eval_program(program, input_str[num_demo:], output_str[num_demo:])
        accepted = (seen_score["AC"] + unseen_score["AC"]) / len(input_str)
        runtime_error = (seen_score["RE"] + unseen_score["RE"]) / len(input_str)
        syntax_error = seen_score["CE"]
        num_seen = num_demo
        num_unseen = len(input_str) - num_demo
        return {
            "accepted": accepted,
            "runtime_error": runtime_error,
            "syntax_error": syntax_error,
            "seen_accepted": seen_score["AC"] / num_seen,
            "seen_runtime_error": seen_score["RE"] / num_seen,
            "unseen_accepted": unseen_score["AC"] / num_unseen,
            "unseen_runtime_error": unseen_score["RE"] / num_unseen,
        }

    def eval_program(self, program: str, input_str: List[str], output_str: List[str]):
        """
        AC: Accepted
        CE: Compile Error
        RE: Runtime Error
        """
        accepted = 0
        compile_error = 0
        runtime_error = 0
        for i, o in zip(input_str, output_str):
            self.env.set_new_state(i)
            try:
                self.dsl.run(self.env, program)
                accepted += self.env.output_str == o
            except RuntimeError:
                return {
                    "CE": 1.0,
                    "RE": 0.0,
                    "AC": 0.0,
                }
            except IndexError:
                runtime_error += 1
            except:
                pass
        return {
            "CE": compile_error,
            "RE": runtime_error,
            "AC": accepted,
        }
    
    def eval_single_program(self, program: str, input_str: List[str], output_str: List[str]):
        try:
            exe: Callable = self.dsl.parse(program)
        except RuntimeError:
            return 0.0
        score = 0
        for i, o in zip(input_str, output_str):
            self.env.set_new_state(i)
            try:
                exe(self.env)
                score += self.env.output_str == o
            except IndexError:
                pass
        return score / len(input_str)

    def eval_single_sample(self, program: str, input_str: List[str], output_str: List[str]):
        for i, o in zip(input_str, output_str):
            self.env.set_new_state(i)
            try:
                self.dsl.run(self.env, program)
                if self.env.output_str != o:
                    return 0.0
            except:
                return 0.0
        return 1.0

    def eval_batch_program(self, program_batch, input_str_batch, output_str_batch, STARTER):
        scores = []
        for p,input_str, output_str, in zip(program_batch, input_str_batch, output_str_batch):
            prog = [STARTER] + list(p)
            scores.append(self.eval_single_program(prog, input_str, output_str))
        
        return np.mean(scores) * 100

    def eval_sample(self, program: str, input_output: tuple):
        input_str, output_str = input_output
        return self.eval_single_program(program, input_str, output_str)
