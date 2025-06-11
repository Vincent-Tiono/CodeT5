from collections import defaultdict
from string_trans.consts import *
import numpy as np
from string_trans.string_env import StringTransEnv
from string_trans.dsl_eval import DSL_MAP
from string_trans.parser_utils import SubstrFunctionality, ProgramFunctionality
from src.tokenization_nps import ProgramTokenizer
from omegaconf import DictConfig
from tqdm.auto import tqdm
from ordered_set import OrderedSet
import multiprocessing as mp
from datasets import Dataset, DatasetDict
import editdistance as ed
import os
from functools import partial
import ipdb

FUCCTION_MAP = {
    "base": ProgramFunctionality,
    "substr": SubstrFunctionality,
}

PROGRAM = "program"
INPUTS = "inputs"
OUTPUTS = "outputs"

def load_disk_dataset(examples):
    inputs = examples[INPUTS]
    outputs = examples[OUTPUTS]
    programs = examples[PROGRAM]
    return {
        INPUTS: inputs,
        OUTPUTS: outputs,
        PROGRAM: programs,
    }


class StringGenerator:
    def __init__(self, num_demo, min_str_len=20, max_str_len=30, maximum_attempt=100, dsl_name="base"):
        self.dsl = DSL_MAP[dsl_name]()
        self.lexer = self.dsl.lexer
        self.yacc = self.dsl.yacc
        self.min_str_len = min_str_len
        self.max_str_len = max_str_len
        self.maximum_attempt = maximum_attempt
        self.num_demo = num_demo
        self.env = StringTransEnv(input_str="")

    def get_tokens(self, code):
        self.lexer.input(code)
        tokens = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            tokens.append(tok)
        return tokens

    def get_stats(self, code):
        tokens = self.get_tokens(code)
        program_regex = defaultdict(int)
        program_schar = defaultdict(int)
        for i, token in enumerate(tokens):
            if token.type.capitalize() in REGEX_DICT or token.type == "SCHAR":
                if tokens[i + 1].type == "INT":
                    repetition = tokens[i + 1].value
                    if repetition >= 0:
                        repetition += 1
                    else:
                        repetition = -repetition
                else:
                    repetition = 1
                if token.type.capitalize() in REGEX_DICT:
                    program_regex[token.type.capitalize()] += repetition
                elif token.type == "SCHAR" and tokens[i - 1].type != "SCHAR":
                    schar = token.value
                    schar = " " if schar == "\"SPACE\"" else schar[1:-1]
                    program_schar[schar] += repetition
            elif token.type == "TRIM":
                program_schar["TRIM"] += 1
        return program_regex, program_schar

    def regex_gen(self, reg):
        delimiter = np.random.choice(SCHAR_CHAR)
        return self._one_regex_gen(reg) + delimiter

    def _one_regex_gen(self, reg):
        l = int(np.ceil(np.random.exponential(1.5)))
        if reg == "Word":
            return "".join(np.random.choice(WORD_CHAR, l))
        elif reg == "Num":
            return "".join(np.random.choice(INT_CHAR, l))
        elif reg == "Alphanum":
            return "".join(np.random.choice(ALPHANUM_CHAR, l))
        elif reg == "Allcaps":
            return "".join(np.random.choice(UPPER_CHAR, l))
        elif reg == "Propcase":
            return (
                np.random.choice(UPPER_CHAR) +
                "".join(np.random.choice(LOWER_CHAR, l))
            )
        elif reg == "Lower":
            return "".join(np.random.choice(LOWER_CHAR, l))
        elif reg == "Digit":
            return "".join(np.random.choice(INT_CHAR))
        elif reg == "Char":
            return "".join(np.random.choice(ALPHANUM_CHAR))

    def schar_gen(self, schar):
        if schar == "SPACE":
            schar = " "
        l = int(np.ceil(np.random.exponential()))
        if schar == "TRIM":
            return " " * l
        return schar

    def generate_input_str(self, stats):
        program_regex, program_schar = stats
        input_str = []
        for reg, repetition in program_regex.items():
            r = np.random.randint(min(1, repetition - 5), repetition + 5)
            input_str.extend([self.regex_gen(reg) for _ in range(r)])
        for schar, repetition in program_schar.items():
            r = np.random.randint(min(1, repetition - 5), repetition + 5)
            input_str.extend([self.schar_gen(schar) for _ in range(r)])

        length = len("".join(input_str))
        size = max(5, self.max_str_len - length)

        input_str += np.random.choice(ALL_CHAR, size=(size,)).tolist()
        np.random.shuffle(input_str)
        l = np.random.randint(self.min_str_len, self.max_str_len + 1)
        return "".join(input_str)[:l]

    def generate_synthesis_sample(self, code):
        program_stats = self.get_stats(code)

        # compile code
        exec_fn = self.yacc.parse(code)

        success_input_str = []
        success_output_str = []
        attempt = 0

        while True:
            if len(success_input_str) >= self.num_demo:
                break
            if attempt > self.maximum_attempt:
                # If random sampling can not generate enough program, discard this program
                return [], []
            input_str = self.generate_input_str(program_stats)
            self.env.set_new_state(input_str)
            attempt += 1
            try:
                exec_fn(self.env)
                o = self.env.output_str
                if len(o) != 0:
                    success_input_str.append(input_str)
                    success_output_str.append(o)
            except IndexError:
                pass

        return success_input_str, success_output_str


    def generate_sample_with_code(self, code1, code2):
        reg_dict1, sch_dict1 = self.get_stats(code1)
        reg_dict2, sch_dict2 = self.get_stats(code2)
        for k, v in reg_dict2.items():
            reg_dict1[k] = max(reg_dict1[k], v)
        for k, v in sch_dict2.items():
            sch_dict1[k] = max(sch_dict1[k], v)
        program_stats = reg_dict1, sch_dict1

        # compile code
        exec_fn = self.yacc.parse(code1)

        success_input_str = []
        success_output_str = []
        attempt = 0

        while True:
            if len(success_input_str) >= self.num_demo:
                break
            if attempt > self.maximum_attempt:
                # If random sampling can not generate enough program, discard this program
                return [], []
            input_str = self.generate_input_str(program_stats)
            self.env.set_new_state(input_str)
            attempt += 1
            try:
                exec_fn(self.env)
                o = self.env.output_str
                if len(o) != 0:
                    success_input_str.append(input_str)
                    success_output_str.append(o)
            except IndexError:
                pass

        return success_input_str, success_output_str

    def generate_consensus_sample(self, code_pair, tries=400):
        code1, code2 = code_pair
        reg_dict1, sch_dict1 = self.get_stats(code1)
        reg_dict2, sch_dict2 = self.get_stats(code2)
        for k, v in reg_dict2.items():
            reg_dict1[k] = max(reg_dict1[k], v)
        for k, v in sch_dict2.items():
            sch_dict1[k] = max(sch_dict1[k], v)
        program_stats = reg_dict1, sch_dict1

        # compile code
        exec_fn1 = self.yacc.parse(code1)
        exec_fn2 = self.yacc.parse(code2)

        success_input_str = []
        success_output_str = []
        attempt = 0

        for _ in range(tries):
            if len(success_input_str) >= self.num_demo:
                break
            if attempt > self.maximum_attempt:
                # If random sampling can not generate enough program, discard this program
                return success_input_str, success_output_str
            input_str = self.generate_input_str(program_stats)
            self.env.set_new_state(input_str)
            attempt += 1
            try:
                exec_fn1(self.env)
                o1 = self.env.output_str
                self.env.set_new_state(input_str)
                exec_fn2(self.env)
                o2 = self.env.output_str
                if len(o1) != 0 and o1 == o2:
                    success_input_str.append(input_str)
                    success_output_str.append(o1)
            except IndexError:
                pass

        return success_input_str, success_output_str

    def get_string_edit_distance(self, string1, string2):
        if string2 == None:
            return len(string1)
        return ed.eval(string1, string2)
    
    def get_cer(self, string1, string2):
        if string2 == None:
            return len(string1)
        return ed.eval(string1, string2) / len(string1)
    
    def generate_similar_sample(self, code_pair):
        code1, code2 = code_pair
        reg_dict1, sch_dict1 = self.get_stats(code1)
        reg_dict2, sch_dict2 = self.get_stats(code2)
        for k, v in reg_dict2.items():
            reg_dict1[k] = max(reg_dict1[k], v)
        for k, v in sch_dict2.items():
            sch_dict1[k] = max(sch_dict1[k], v)
        program_stats = reg_dict1, sch_dict1

         # compile code
        exec_fn1 = self.yacc.parse(code1)
        exec_fn2 = self.yacc.parse(code2)

        success_input_str = []
        success_output_str1 = []
        success_output_str2 = []
        attempt = 0

        for _ in range(self.maximum_attempt):
            input_str = self.generate_input_str(program_stats)
            self.env.set_new_state(input_str)
            attempt += 1
            try:
                exec_fn1(self.env)
                o1 = self.env.output_str
                self.env.set_new_state(input_str)
                exec_fn2(self.env)
                o2 = self.env.output_str
                if len(o1) != 0 and len(o2) != 0:
                    success_input_str.append(input_str)
                    success_output_str1.append(o1)
                    success_output_str2.append(o2)
            except IndexError:
                pass

        if len(success_input_str) < self.num_demo:
            return [], [], []

        distances = [self.get_string_edit_distance(o1, o2) for o1, o2 in zip(success_output_str1, success_output_str2)]
        selected = np.argsort(distances)[:self.num_demo]
        output_str1 = np.array(success_output_str1)[selected].tolist()
        output_str2 = np.array(success_output_str2)[selected].tolist()
        input_str = np.array(success_input_str)[selected].tolist()
        return input_str, output_str1, output_str2

    def generate_similar_sample_with_programs(self, ref_code, codes, tries=10):
        reg_dict, sch_dict = self.get_stats(ref_code)
        for code in codes:
            reg_dict2, sch_dict2 = self.get_stats(code)
            for k, v in reg_dict2.items():
                reg_dict[k] = max(reg_dict[k], v)
            for k, v in sch_dict2.items():
                sch_dict[k] = max(sch_dict[k], v)
        program_stats = reg_dict, sch_dict
        base_program_stats = self.get_stats(ref_code)
        # ipdb.set_trace()

        # compile code
        ref_exec_fn = self.yacc.parse(ref_code)
        exec_fns = [self.yacc.parse(code) for code in codes]

        attempt = 0
        str_idx = 0
        output_distances = np.zeros((tries, len(exec_fns)))
        input_strs = []
        ref_output_strs = []
        output_strs = []

        while True:
            if attempt > 100:
                input_str = self.generate_input_str(base_program_stats)
            else:
                input_str = self.generate_input_str(program_stats)
            attempt += 1
            self.env.set_new_state(input_str)
            try:
                ref_exec_fn(self.env)
                o = self.env.output_str
            except:
                continue
            if len(o) == 0:
                continue
            output_str = []
            for exec_fn in exec_fns:
                self.env.set_new_state(input_str)
                try:
                    exec_fn(self.env)
                    o2 = self.env.output_str
                    output_str.append(o2)
                except:
                    output_str.append(None)

            output_strs.append(np.array(output_str))
            input_strs.append(np.array([input_str for _ in range(len(codes))]))
            ref_output_strs.append(np.array([o for _ in range(len(codes))]))
            output_distances[str_idx] = np.array([self.get_cer(o, o2) for o2 in output_str])
            str_idx += 1
            if str_idx >= tries:
                break

        output_strs = np.array(output_strs)
        input_strs = np.array(input_strs)
        ref_output_strs = np.array(ref_output_strs)
        has_different_output = (output_strs != None) & (output_strs != "") & (output_distances != 0)
        output_strs = output_strs[has_different_output]
        input_strs = input_strs[has_different_output]
        output_distances = output_distances[has_different_output]
        ref_output_strs = ref_output_strs[has_different_output]

        selected_indices = np.argsort(output_distances)[:len(codes)]
        success_input_str = input_strs[selected_indices]
        success_output_str = output_strs[selected_indices]
        success_ref_output_str = ref_output_strs[selected_indices]

        return success_input_str.tolist(), success_output_str.tolist(), success_ref_output_str.tolist()
       


class SampleGenerator(StringGenerator):
    def __init__(self, num_demo, programs, min_str_len=20, max_str_len=30, maximum_attempt=100, dsl_name="base", threshold=0.1):
        super().__init__(
            num_demo=num_demo,
            min_str_len=min_str_len,
            max_str_len=max_str_len,
            maximum_attempt=maximum_attempt,
            dsl_name=dsl_name
        )
        self.programs = programs
        self.threshold = threshold

    def get_stats(self, code):
        reg_dict, sch_dict = super().get_stats(code)
        if np.random.random() < self.threshold:
            random_reg, random_sch = self.get_stats(np.random.choice(self.programs))
            for k, v in random_reg.items():
                reg_dict[k] += v
            for k, v in random_sch.items():
                sch_dict[k] += v
        program_stats = reg_dict, sch_dict
        return program_stats

class ProgramGenerator:
    prob_prog = [1.0]
    prob_expr = [0.2, 0.2, 0.2, 0.2, 0.2]
    substr_prob_expr = [1/3, 1/3, 1/3]
    prob_nesting = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    prob_nesting_plus = [0.5, 0.5]
    prob_case = [1/3, 1/3, 1/3]
    prob_expr_expr = [1.0]
    prob_conststr = [1.0]
    prob_substr = [1.0]
    prob_pos = [0.45, 0.45, 0.1]
    prob_reg = [0.5, 0.5]
    prob_type = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    prob_index = [1.0]
    prob_position = [1.0]

    def __init__(self, seed=0, dsl_name="base", max_length_only=False) -> None:
        self.dsl_name = dsl_name
        self.dsl = DSL_MAP[dsl_name]()
        self.max_length_only = max_length_only
        self.prodnames = self.dsl.grammar.Prodnames
        self.rng = np.random.RandomState(seed=seed)

    def random_code(self, start_token="prog", max_length=6):
        if self.max_length_only:
            prod = self.prodnames["prog"][0]
            tokens = []
            for term in prod.prod:
                if term == "expr":
                    for _ in range(max_length):
                        tokens.extend(self.random_tokens("expr", max_length=1))
                else:
                    token = getattr(self.dsl, "t_{}".format(term))
                    tokens.append(str(token).replace("\\", ""))
            code = " ".join(tokens)
        else:
            code = " ".join(self.random_tokens(
                start_token, max_length=max_length))
        return code

    def random_tokens(self, start_token="prog", length=None, max_length=6):
        if length is None:
            length = [1]
        codes = []
        candidates = self.prodnames[start_token]
        if start_token == "expr" and self.dsl_name == "substr":
            sample_prob = self.substr_prob_expr
        else:
            sample_prob = getattr(self, "prob_{}".format(start_token))

        prod = candidates[self.rng.choice(
            range(len(candidates)), p=sample_prob)]
        while prod.prod[0] == "expr_expr" and length[0] >= max_length:
            prod = candidates[self.rng.choice(
                range(len(candidates)), p=sample_prob)]

        for term in prod.prod:
            if term == "index":
                token = self.random_index()
                codes.append(str(token))
            elif term == "position":
                token = self.random_position()
                codes.append(str(token))
            elif term in self.prodnames:  # need digging
                if term == "expr_expr":
                    length[0] += 1
                    codes.extend(self.random_tokens(term, length, max_length))
                else:
                    codes.extend(self.random_tokens(term, length, max_length))
            else:
                token = getattr(self.dsl, "t_{}".format(term))
                if callable(token):
                    if token == self.dsl.t_SCHAR:
                        token = self.random_schar()
                    else:
                        raise Exception(
                            " [!] Undefined token `{}`".format(token))

                codes.append(str(token).replace("\\", ""))

        return codes

    def random_index(self):
        return self.rng.randint(-MAX_INDEX, MAX_INDEX)

    def random_position(self):
        return self.rng.randint(-MAX_POSITION, MAX_POSITION)

    def random_schar(self):
        return "\"{}\"".format(self.rng.choice(SCHAR_LIST))

# Program by Example dataset generator, convert form dsl_generator.Generator to more flexible version
class PBEGenerator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.program_generator = ProgramGenerator(cfg.seed, dsl_name=cfg.dsl_name, max_length_only=cfg.max_length_only)
        self.rng = np.random.RandomState(cfg.seed)
        self.string_generator = StringGenerator(
            num_demo=cfg.num_demo,
            min_str_len=cfg.min_str_len,
            max_str_len=cfg.max_str_len,
            maximum_attempt=cfg.maximum_attempt,
            dsl_name=cfg.dsl_name,
        )
        self.dsl = self.program_generator.dsl
        self.output_dir = cfg.output_dir
        self.pf = FUCCTION_MAP[cfg.dsl_name]()
        self.all_program = cfg.all_program
        self.num_train = cfg.num_train
        self.num_val = cfg.num_val
        self.num_test = cfg.num_test
        self.tol = cfg.tol
        self.non_uniform = cfg.non_uniform
        self.max_prog_len = cfg.max_prog_len
        self.program_generation_ratio = cfg.program_generation_ratio
        self.construct_vocab()

    def generate(self) -> None:
        self.generate_synthesis_dataset()

    def construct_vocab(self):
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.token2int = []
            self.int2token = ["<pad>", "<sos>", "<eos>"]
            for term in self.dsl.tokens:
                token = getattr(self.dsl, "t_{}".format(term))
                if callable(token):
                    if token == self.dsl.t_INT:
                        for i in range(MIN_INT, MAX_INT + 1):
                            self.int2token.append("{}{}".format(INT_PREFIX, i))
                    if token == self.dsl.t_SCHAR:
                        for c in SCHAR_LIST:
                            self.int2token.append("\"{}\"".format(c))
                else:
                    self.int2token.append(str(token).replace("\\", ""))
            self.token2int = {v: i for i, v in enumerate(self.int2token)}
            with open(os.path.join(self.output_dir, "vocab.json"), "w") as f:
                json.dump(self.token2int, f, indent=4)
            self.int2char = ["<pad>", "<sos>", "<eos>"] + ALL_CHAR
            self.char2int = {v: i for i, v in enumerate(self.int2char)}
            with open(os.path.join(self.output_dir, "char.json"), "w") as f:
                json.dump(self.char2int, f, indent=4)

            self.sos_id = self.token2int["<sos>"]
            self.eos_id = self.token2int["<eos>"]
            self.pad_id = self.token2int["<pad>"]


    def generate_synthesis_dataset(self):
        num_program_to_generate = self.num_train + self.num_val + self.num_test

        non_repeat = int(self.program_generation_ratio * (num_program_to_generate))
        code_set = OrderedSet([self.program_generator.random_code(max_length=self.max_prog_len)
                       for _ in tqdm(range(non_repeat))])

        success_program = []
        success_inputs = []
        success_outputs = []

        code_set = list(code_set)
        synthesis_function = self.string_generator.generate_synthesis_sample
        with mp.Pool(12) as pool:
            results = pool.map(synthesis_function, tqdm((code_set)))

        func_map = {}
        func_count = np.zeros(len(self.pf.all_functions))
        total_seen = [0]
        for i, k in enumerate(self.pf.all_functions.keys()):
            func_map[k] = i

        if not self.all_program:
            progress_bar = tqdm(range(num_program_to_generate), desc="Sample uniform program")

        fail = []
        discard = []

        def get_importance(code):
            functionalities = self.pf.yacc.parse(code)
            functionalities = functionalities.split("-")
            p_min = 1.0
            p_curr = 1.0
            for functionality in functionalities:
                fid = func_map[functionality]
                total_seen[0] += 1
                func_count[fid] += 1
                p_min = p_min * (np.min(func_count) / total_seen[0])
                p_curr = p_curr * (func_count[fid] / total_seen[0])
            g = (p_min + self.tol) / (p_curr + self.tol)
            return g

        for i, (code, (inputs, outputs)) in enumerate(zip(code_set, results)):
            if len(success_inputs) >= num_program_to_generate and not self.all_program:
                break
            if len(inputs) > 0:
                g = get_importance(code)
                r = self.rng.random()
                if r < g or self.non_uniform or self.all_program:
                    success_program.append(code)
                    success_inputs.append(inputs)
                    success_outputs.append(outputs)
                    if not self.all_program:
                        progress_bar.update(1)
                else:
                    discard.append(i)
            else:
                fail.append(i)

        if self.all_program:
            self.write_dataset_with_other(success_program, success_inputs, success_outputs, self.output_dir)
        else:
            self.write_dataset(success_program, success_inputs, success_outputs, self.output_dir)

    def write_dataset(self, programs, inputs, outputs, output_dir=None, uniform_num=None):
        num_train = min(len(programs) - self.num_val - self.num_test, self.num_train) if uniform_num is None else min(uniform_num - self.num_val - self.num_test, self.num_train)
        num_train_val = num_train + self.num_val
        train_program = programs[:num_train] if uniform_num is None else programs[:num_train] + programs[uniform_num:]
        train_inputs = inputs[:num_train] if uniform_num is None else inputs[:num_train] + inputs[uniform_num:]
        train_outputs = outputs[:num_train] if uniform_num is None else outputs[:num_train] + outputs[uniform_num:]

        val_program = programs[num_train:num_train_val]
        val_inputs = inputs[num_train:num_train_val]
        val_outputs = outputs[num_train:num_train_val]

        test_program = programs[num_train_val:]
        test_inputs = inputs[num_train_val:]
        test_outputs = outputs[num_train_val:]

        train_set = Dataset.from_dict({
            PROGRAM: train_program,
            INPUTS: train_inputs,
            OUTPUTS: train_outputs,
        })
        val_set = Dataset.from_dict({
            PROGRAM: val_program,
            INPUTS: val_inputs,
            OUTPUTS: val_outputs,
        })
        test_set = Dataset.from_dict({
            PROGRAM: test_program,
            INPUTS: test_inputs,
            OUTPUTS: test_outputs,
        })

        dataset = DatasetDict({
            "train": train_set,
            "val": val_set,
            "test": test_set,
        })
        output_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.write_information(dataset, output_dir)
        self.write_five_sample(dataset, output_dir)
        self.write_distribution(dataset, output_dir)
        dataset.save_to_disk(output_dir)

    def write_dataset_with_other(self, programs, inputs, outputs, output_dir=None):
        num_train = min(len(programs) - self.num_val - self.num_test, self.num_train)
        num_train_val = num_train + self.num_val
        num_all_set = num_train_val + self.num_test
        train_program = programs[:num_train]
        train_inputs = inputs[:num_train]
        train_outputs = outputs[:num_train]

        val_program = programs[num_train:num_train_val]
        val_inputs = inputs[num_train:num_train_val]
        val_outputs = outputs[num_train:num_train_val]

        test_program = programs[num_train_val:num_all_set]
        test_inputs = inputs[num_train_val:num_all_set]
        test_outputs = outputs[num_train_val:num_all_set]

        train_set = Dataset.from_dict({
            PROGRAM: train_program,
            INPUTS: train_inputs,
            OUTPUTS: train_outputs,
        })
        val_set = Dataset.from_dict({
            PROGRAM: val_program,
            INPUTS: val_inputs,
            OUTPUTS: val_outputs,
        })
        test_set = Dataset.from_dict({
            PROGRAM: test_program,
            INPUTS: test_inputs,
            OUTPUTS: test_outputs,
        })

        if len(programs) > num_all_set:
            other_program = programs[num_all_set:]
            other_inputs = inputs[num_all_set:]
            other_outputs = outputs[num_all_set:]
            other_set = Dataset.from_dict({
                PROGRAM: other_program,
                INPUTS: other_inputs,
                OUTPUTS: other_outputs,
            })
            dataset = DatasetDict({
                "train": train_set,
                "val": val_set,
                "test": test_set,
                "other": other_set,
            })

        else:
            dataset = DatasetDict({
                "train": train_set,
                "val": val_set,
                "test": test_set,
            })
        output_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.write_information(dataset, output_dir)
        self.write_five_sample(dataset, output_dir)
        self.write_distribution(dataset, output_dir)
        dataset.save_to_disk(output_dir)


    def write_information(self, raw_datasets, output_dir=None, dataset_type="synthesis"):
        max_program_len = 0
        max_input_len = 0
        max_output_len = 0
        for split in raw_datasets.keys():
            program = raw_datasets[split][PROGRAM]
            inputs = raw_datasets[split][INPUTS]
            outputs = raw_datasets[split][OUTPUTS]
            if dataset_type == "synthesis":
                program_length = [self.get_program_len(p) for p in program]
                inputs_length = [self.get_string_len(i) for i in inputs]
            else:
                program_length = [self.get_program_len_multi(p) for p in program]
                inputs_length = [len(i) + 2 for i in inputs]
            outputs_length = [self.get_string_len(o) for o in outputs]
            max_program_len = max(max_program_len, max(program_length))
            max_input_len = max(max_input_len, max(inputs_length))
            max_output_len = max(max_output_len, max(outputs_length))
        num_train = len(raw_datasets["train"])
        num_val = len(raw_datasets["val"])
        num_test = len(raw_datasets["test"])
        info = {
            "max_program_length": max_program_len,
            "max_input_length": max_input_len,
            "max_output_length": max_output_len,
            "num_train": num_train,
            "num_val": num_val,
            "num_test": num_test,
        }
        if "other" in raw_datasets.keys():
            info["num_other"] = len(raw_datasets["other"])
        output_dir = output_dir if output_dir is not None else self.output_dir
        with open(os.path.join(output_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

    def write_distribution(self, raw_datasets, output_dir=None):
        dist = {}
        for split in raw_datasets.keys():
            program = raw_datasets[split][PROGRAM]
            functionalities = [self.pf.yacc.parse(p) for p in program]
            all_function = deepcopy(self.pf.all_functions)
            for functionality in functionalities:
                for f in functionality.split("-"):
                    all_function[f] += 1
            dist[split] = all_function
        output_dir = output_dir if output_dir is not None else self.output_dir
        with open(os.path.join(output_dir, "distribution.json"), "w") as f:
            json.dump(dist, f, indent=4)

    def write_five_sample(self, raw_datasets, output_dir=None):
        train = raw_datasets["train"]
        five_sample = {}
        for i in range(5):
            data = train[i]
            five_sample[i] = data
        output_dir = output_dir if output_dir is not None else self.output_dir
        with open(os.path.join(output_dir, "sample.json"), "w") as f:
            json.dump(five_sample, f, indent=4)

    def get_program_len(self, program):
        return len(program.split()) + 2
    
    def get_program_len_multi(self, program):
        return max([len(p.split()) for p in program]) + 2

    def get_string_len(self, string):
        return max([len(s) for s in string]) + 2

class EditDistanceGenerator(PBEGenerator):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        assert cfg.input_dir is not None
        self.input_dir = cfg.input_dir
        pbe_dataset = DatasetDict.load_from_disk(cfg.input_dir)
        self.pbe_dataset = pbe_dataset.map(load_disk_dataset)
        self.train_program = self.pbe_dataset["train"][PROGRAM]
        self.eval_program = self.pbe_dataset["val"][PROGRAM]
        self.test_program = self.pbe_dataset["test"][PROGRAM]
        self.tokenizer = ProgramTokenizer()

    def get_edit_distance(self, program1, program2):
        p1_ids = self.tokenizer.encode(program1)
        p2_ids = self.tokenizer.encode(program2)
        return ed.eval(p1_ids, p2_ids)


    def get_similar(self, program_idx, programs, similar_num=512, similar_type="edit_distance"):
        program = programs[program_idx]
        similar_distance = []
        if similar_type == "edit_distance":
            random_choice_indices = np.random.choice(len(programs), similar_num*2, replace=False)
            for i in random_choice_indices:
                if i == program_idx:
                    continue
                similar_distance.append(self.get_edit_distance(programs[i], program))
            similar_program = random_choice_indices[np.argsort(similar_distance)[:similar_num]]
            return similar_program
        elif similar_type == "str_edit_distance":
            random_choice_indices = np.random.choice(len(programs), similar_num+1, replace=False)
            random_choice_programs = []
            for i in random_choice_indices:
                if i == program_idx:
                    continue
                random_choice_programs.append(programs[i])
            input_str, output_str, ref_output_str = self.string_generator.generate_similar_sample_with_programs(program, random_choice_programs[:similar_num])
            return input_str, output_str, ref_output_str
        else:
            raise NotImplementedError
        

    def generate(self) -> None:
        # generate similar functions
        get_train_similar_io = partial(self.get_similar, programs=self.train_program, similar_type="str_edit_distance")
        get_val_similar_io = partial(self.get_similar, programs=self.eval_program, similar_type="str_edit_distance")
        get_test_similar_io = partial(self.get_similar, programs=self.test_program, similar_type="str_edit_distance")
        get_train_similar_program = partial(self.get_similar, programs=self.train_program)
        get_val_similar_program = partial(self.get_similar, programs=self.eval_program)
        get_test_similar_program = partial(self.get_similar, programs=self.test_program)

        # generate similar samples using multiprocessing
        with mp.Pool(16) as pool:
            train_similar_io = pool.map(get_train_similar_io, tqdm(range(len(self.train_program))))
            val_similar_io = pool.map(get_val_similar_io, tqdm(range(len(self.eval_program))))
            test_similar_io = pool.map(get_test_similar_io, tqdm(range(len(self.test_program))))
            train_similar_program = pool.map(get_train_similar_program, tqdm(range(len(self.train_program))))
            val_similar_program = pool.map(get_val_similar_program, tqdm(range(len(self.eval_program))))
            test_similar_program = pool.map(get_test_similar_program, tqdm(range(len(self.test_program))))

        # post-processing similar IOs
        def get_similar_io(similar_io):
            input_str, output_str, ref_output_str = [], [], []
            for i, o, ref in similar_io:
                input_str.append(i)
                output_str.append(o)
                ref_output_str.append(ref)
            return input_str, output_str, ref_output_str
        
        train_similar_io = get_similar_io(train_similar_io)
        val_similar_io = get_similar_io(val_similar_io)
        test_similar_io = get_similar_io(test_similar_io)

        def get_similar_dataset(dataset: Dataset, similar_io, similar_program):
            input_str, output_str, ref_output_str = similar_io
            dataset = dataset.add_column("similar_input", input_str)
            dataset = dataset.add_column("similar_output", output_str)
            dataset = dataset.add_column("reference_output", ref_output_str)
            dataset = dataset.add_column("similar_program", similar_program)
            return dataset

        dataset = DatasetDict({
            "train": get_similar_dataset(self.pbe_dataset["train"], train_similar_io, train_similar_program),
            "val": get_similar_dataset(self.pbe_dataset["val"], val_similar_io, val_similar_program),
            "test": get_similar_dataset(self.pbe_dataset["test"], test_similar_io, test_similar_program),
        })

        self.write_information(dataset, self.output_dir)
        self.write_five_sample(dataset, self.output_dir)
        self.write_distribution(dataset, self.output_dir)
        dataset.save_to_disk(self.output_dir)


class HardProgramWithReleventIOGenerator(EditDistanceGenerator):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def get_all_similar(self, split):
        programs = np.array(self.pbe_dataset[split][PROGRAM])
        # all_pairs_index = []
        # all_distance = np.zeros(len(programs) * (len(programs) - 1), dtype=np.uint8)
        L = len(programs)

        progress_bar = tqdm(range(int((L - 1) * (L - 2) / 2)))
        current_idx = 0
        for i in range(len(programs)):
            for j in range(i + 1, len(programs)):
                self.get_edit_distance(program1=programs[i], program2=programs[j])
                progress_bar.update(1)
                # current_idx += 1
        # all_pairs_index = np.array(all_pairs_index)
        # all_program_pairs = np.zeros_like(all_pairs_index, dtype=str)
        # all_program_pairs[:, 0] = programs[all_pairs_index[:, 0]]
        # all_program_pairs[:, 1] = programs[all_pairs_index[:, 1]]
        
    
    def get_edit_distance_tuple(self, program_tuple: tuple):
        program1, program2 = program_tuple
        p1_ids = self.tokenizer.encode(program1)
        p2_ids = self.tokenizer.encode(program2)
        return ed.eval(p1_ids, p2_ids)

    def generate(self) -> None:
        self.get_all_similar("train")

if __name__ == "__main__":
    # test
    pg = ProgramGenerator(seed=0)
    code = pg.random_code()
    print(code)
    pg =  ProgramGenerator(seed=0, dsl_name="substr")
    code = pg.random_code()
    print(code)

