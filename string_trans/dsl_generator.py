import argparse
import multiprocessing as mp

import numpy as np
from string_trans.consts import *
from string_trans.dsl import StringTransformationDSL
from string_trans.parser_utils import ProgramFunctionality
from string_trans.string_env import StringTransEnv
from tqdm.auto import tqdm
import json
from typing import List
import os
import torch
import editdistance as ed

from datasets import Dataset, DatasetDict

from collections import defaultdict, Counter
from ordered_set import OrderedSet

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.tokenize import TweetTokenizer
from datasets import load_dataset
from copy import deepcopy

from transformers import BertModel, BertTokenizer
from torchmetrics.functional import pairwise_cosine_similarity

import re
import ipdb

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def execution_behavior(input, trace, history):
    behavior = []
    for i, (middle_point, h) in enumerate(zip(trace, history)):
        if i == 0:
            behavior.append(input != middle_point)
        elif h != history[i-1]:
            behavior.append(input != middle_point)
        else:
            behavior.append(middle_point != trace[i-1])
    return tuple(behavior)

def sample_entropy(behaviors):
    if len(behaviors) == 0:
        return 0
    behavior_counter = Counter(behaviors)
    total_sample = sum(behavior_counter.values())
    prob = np.array([v / total_sample for v in behavior_counter.values()])
    keys = list(behavior_counter.keys())
    entropy = np.abs(np.sum(prob * np.log(prob))) / 2 ** len(keys[0])
    return entropy


class Generator(StringTransformationDSL):
    def __init__(self, args, **kw):
        super().__init__(**kw)
        self.env = StringTransEnv(input_str="")
        self.prodnames = self.grammar.Prodnames
        self.rng = np.random.RandomState(seed=args.seed)
        self.num_train = args.num_train
        self.num_val = args.num_val
        self.num_test = args.num_test
        self.num_demo = args.num_demo
        self.min_str_len = args.min_str_len
        self.max_str_len = args.max_str_len
        self.max_prog_len = args.max_prog_len
        self.current_index = 0
        self.maximum_attempt = args.maximum_attempt
        self.max_length_only = args.max_length_only
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.text_dataset_name = args.text_dataset_name
        self.text_dataset_path = args.text_dataset_path
        self.tol = args.tol
        self.num_retry = args.num_retry
        self.leakage_list = np.linspace(0, 1, self.num_retry + 1)
        self.leakage = args.leakage
        self.non_uniform = args.non_uniform
        self.with_retry= args.with_retry
        self.all_program = args.all_program
        self.unique = [self.t_CONSTSTR, self.t_REPLACE]
        self.pf = ProgramFunctionality()
        self.construct_vocab()

    def construct_vocab(self):
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.token2int = []
            self.int2token = ["<pad>", "<sos>", "<eos>"]
            for term in self.tokens:
                token = getattr(self, "t_{}".format(term))
                if callable(token):
                    if token == self.t_INT:
                        for i in range(MIN_INT, MAX_INT + 1):
                            self.int2token.append("{}{}".format(INT_PREFIX, i))
                    if token == self.t_SCHAR:
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

    def setup_text(self):
        self.split_num = {
            "train": self.num_train,
            "val": self.num_val,
            "test": self.num_test,
        }
        self.detokenizer = TreebankWordDetokenizer()
        self.tokenizer = TweetTokenizer() if self.text_dataset_path.startswith("tweet") else TreebankWordTokenizer()
        self.text_dataset = load_dataset(path=self.text_dataset_path, name=self.text_dataset_name, split="train")
        self.text_indices = np.arange(len(self.text_dataset)).tolist()
        self.rng.shuffle(self.text_indices)
        self.datasets = DatasetDict.load_from_disk(self.input_dir)
        self.programs = self.datasets["train"]["program"]
        self.generate_all_regex()

    def str2intseq(self, string):
        return [self.char2int["<sos>"]] + [self.char2int[char] for char in string] + [self.char2int["<eos>"]]

    def code2intseq(self, code):
        return [self.token2int["<sos>"]] + [self.token2int[t] for t in code.split()] + [self.token2int["<eos>"]]

    def intseq2code(self, intseq):
        tokens = []

        for i in intseq:
            if i == self.sos_id:
                continue
            elif i == self.eos_id or self.pad_id:
                break
            else:
                tokens.append(self.int2token[i])
        return " ".join(tokens)

    def intseq2str(self, intseq):
        tokens = []

        for i in intseq:
            if i == self.sos_id:
                continue
            elif i == self.eos_id or self.pad_id:
                break
            else:
                tokens.append(self.int2char[i])
        return "".join(tokens)

    def random_code(self, start_token="prog", max_length=6):
        if self.max_length_only:
            prod = self.prodnames["prog"][0]
            tokens = []
            for term in prod.prod:
                if term == "expr":
                    for _ in range(max_length):
                        tokens.extend(self.random_tokens("expr", max_length=1))
                else:
                    token = getattr(self, "t_{}".format(term))
                    tokens.append(str(token).replace("\\", ""))
            return " ".join(tokens)
        else:
            code = " ".join(self.random_tokens(
                start_token, max_length=max_length))
            return code

    def random_index(self):
        return self.rng.randint(-MAX_INDEX, MAX_INDEX)

    def random_position(self):
        return self.rng.randint(-MAX_POSITION, MAX_POSITION)

    def random_schar(self):
        return "\"{}\"".format(self.rng.choice(SCHAR_LIST))

    prob_prog = [1.0]
    prob_expr = [0.2, 0.2, 0.2, 0.2, 0.2]
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

    def random_tokens(self, start_token="prog", length=None, max_length=6):
        if length is None:
            length = [1]
        codes = []
        candidates = self.prodnames[start_token]
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
                token = getattr(self, "t_{}".format(term))
                if callable(token):
                    if token == self.t_SCHAR:
                        token = self.random_schar()
                    else:
                        raise Exception(
                            " [!] Undefined token `{}`".format(token))

                codes.append(str(token).replace("\\", ""))

        return codes

    def get_program_len(self, program: str):
        return len(program.split()) + 2
    
    def get_program_len_multi(self, program: List[str]):
        return max([len(p.split()) for p in program]) + 2

    def get_string_len(self, string: List[str]):
        return max([len(s) for s in string]) + 2

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

    def generate_input_str(self, stats, leakage=1.0):
        program_regex, program_schar = stats
        input_str = []
        for reg, repetition in program_regex.items():
            r = self.rng.randint(min(1, repetition - 5), repetition + 5)
            if self.rng.random() < leakage:
                input_str.extend([self.regex_gen(reg) for _ in range(r)])
        for schar, repetition in program_schar.items():
            r = self.rng.randint(min(1, repetition - 5), repetition + 5)
            if self.rng.random() < leakage:
                input_str.extend([self.schar_gen(schar) for _ in range(r)])

        length = len("".join(input_str))
        size = max(5, self.max_str_len - length)

        input_str += self.rng.choice(ALL_CHAR, size=(size,)).tolist()
        self.rng.shuffle(input_str)
        l = self.rng.randint(self.min_str_len, self.max_str_len + 1)
        return "".join(input_str)[:l]

    def random_gen_0(self):
        l = self.rng.randint(self.min_str_len, self.max_str_len + 1)
        return "".join(self.rng.choice(ALL_CHAR, l))

    def random_gen_1(self):
        input_str = []
        for reg in REGEX_LIST:
            for _ in range(MAX_INDEX):
                input_str.append(self.regex_gen(reg))
        for sch in SCHAR_LIST:
            for _ in range(MAX_INDEX):
                input_str.append(self.schar_gen(sch))
        self.rng.shuffle(input_str)
        l = self.rng.randint(self.min_str_len, self.max_str_len + 1)
        return "".join(input_str)[:l]

    def generate_all_regex(self):
        self.regex_dict = {}
        for k, v in REGEX_DICT.items():
            self.regex_dict[k] = re.compile(v)
        for schar in SCHAR_LIST:
            if schar == "SPACE":
                self.regex_dict[" "] = re.compile(" ")
            else:
                try:
                    self.regex_dict[schar] = re.compile(schar)
                except:
                    self.regex_dict[schar] = re.compile('\\' + schar)

    def regex_gen(self, reg):
        delimiter = self.rng.choice(SCHAR_CHAR)
        return self._one_regex_gen(reg) + delimiter

    def _one_regex_gen(self, reg):
        l = int(np.ceil(self.rng.exponential(1.5)))
        if reg == "Word":
            return "".join(self.rng.choice(WORD_CHAR, l))
        elif reg == "Num":
            return "".join(self.rng.choice(INT_CHAR, l))
        elif reg == "Alphanum":
            return "".join(self.rng.choice(ALPHANUM_CHAR, l))
        elif reg == "Allcaps":
            return "".join(self.rng.choice(UPPER_CHAR, l))
        elif reg == "Propcase":
            return (
                self.rng.choice(UPPER_CHAR) +
                "".join(self.rng.choice(LOWER_CHAR, l))
            )
        elif reg == "Lower":
            return "".join(self.rng.choice(LOWER_CHAR, l))
        elif reg == "Digit":
            return "".join(self.rng.choice(INT_CHAR))
        elif reg == "Char":
            return "".join(self.rng.choice(ALPHANUM_CHAR))

    def schar_gen(self, schar):
        if schar == "SPACE":
            schar = " "
        l = int(np.ceil(self.rng.exponential()))
        if schar == "TRIM":
            return " " * l
        return schar

    def generate_synthesis_sample(self, code):
        program_stats = self.get_stats(code)
        inputs = [self.generate_input_str(program_stats, self.leakage)
                  for _ in range(self.num_demo)]

        # compile code
        exec_fn = self.yacc.parse(code)

        # execute
        outputs = []
        for input_str in inputs:
            self.env.set_new_state(input_str)
            try:
                exec_fn(self.env)
                o = self.env.output_str
                if len(o) != 0:
                    outputs.append(o)
                else:
                    outputs.append(None)
            except IndexError:
                outputs.append(None)

        success = [o is not None for o in outputs]
        success_rate = np.mean(success)

        if success_rate == 0:
            # If random sampling can not generate any program, discard this program
            return [], []

        success_input_str = []
        success_output_str = []
        for i, o in zip(inputs, outputs):
            if o is not None:
                success_input_str.append(i)
                success_output_str.append(o)

        attempt = 20

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

    def generate_synthesis_dataset(self, output_dir=None):
        num_program_to_generate = self.num_train + self.num_val + self.num_test

        non_repeat = int(20 * (num_program_to_generate))
        code_set = OrderedSet([self.random_code(max_length=self.max_prog_len)
                       for _ in tqdm(range(non_repeat))])

        success_program = []
        success_inputs = []
        success_outputs = []

        code_set = list(code_set)
        synthesis_function = self.generate_synthesis_sample_with_retry if self.with_retry else self.generate_synthesis_sample
        with mp.Pool(20) as pool:
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


    def leakage_experiment(self, output_dir=None):
        num_program_to_generate = self.num_train + self.num_val + self.num_test

        non_repeat = int(20 * (num_program_to_generate))
        code_set = set([self.random_code(max_length=self.max_prog_len)
                       for _ in tqdm(range(non_repeat))])

        num_leakage = len(self.leakage_list)

        success_program = []
        success_inputs = [[] for _ in range(num_leakage)]
        success_outputs = [[] for _ in range(num_leakage)]

        code_set = list(code_set)
        leakage_results = []
        for leakage in self.leakage_list:
            self.leakage = leakage
            with mp.Pool(20) as pool:
                results = pool.map(
                    self.generate_synthesis_sample, tqdm((code_set)))
            leakage_results.append(results)

        func_map = {}
        func_count = np.zeros(len(self.pf.all_functions))
        total_seen = [0]
        for i, k in enumerate(self.pf.all_functions.keys()):
            func_map[k] = i

        progress_bar = tqdm(range(num_program_to_generate), desc="Sample uniform program")

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
        

        for i, code in enumerate(code_set):
            if len(success_inputs[0]) >= num_program_to_generate:
                break

            if all([len(leakage_results[j][i][0]) > 0 for j in range(num_leakage)]):
                g = get_importance(code)
                r = self.rng.random()
                if r < g or self.non_uniform:
                    success_program.append(code)
                    for j in range(num_leakage):
                        inputs, outputs = leakage_results[j][i]
                        success_inputs[j].append(inputs)
                        success_outputs[j].append(outputs)
                        progress_bar.update(1)

        for j in range(num_leakage):
            output_dir = os.path.join(self.output_dir, f"{j:02d}")
            self.write_dataset(success_program, success_inputs[j], success_outputs[j], output_dir)

    def dataset_entropy(self):
        # print(self.input_dir)
        raw_datasets = DatasetDict.load_from_disk(self.input_dir)
        entropies = {}
        for split in ["train", "val", "test"]:
            dataset = raw_datasets[split]
            program = dataset["program"]
            inputs = dataset["inputs"]
            outputs = dataset["outputs"]
            entropy = []
            for p, i, o in zip(program, inputs, outputs):
                exec_fn = self.yacc.parse(p)
                traces = []
                behaviors = []
                eh = None
                for input_str in i:
                    self.env.set_new_state(input_str)
                    exec_fn(self.env)
                    tr = self.env.s_h
                    eh = self.env.ex_h
                    traces.append(tr)
                    # print()
                    behaviors.append(execution_behavior(i, tr, eh))
                entropy.append(sample_entropy(behaviors))
            entropies[split] = np.mean(entropy)
        return entropies

    def generate_text_sample(self, code):
        if self.current_index >= len(self.text_dataset):
            self.current_index = 0
            self.rng.shuffle(self.text_indices)
        program_regex, program_schar = self.get_stats(code)
        regex_count = defaultdict(int)
        for regex, repetition in program_regex.items():
            regex_count[regex] = max(repetition, regex_count[regex])
        for schar, repetition in program_schar.items():
            regex_count[schar] = max(repetition, regex_count[schar])

        exe = self.yacc.parse(code)
        input_str = []
        output_str = []
        total_attempt = 0
        while total_attempt < 1000:
            if self.current_index >= len(self.text_dataset):
                self.current_index = 0
                self.rng.shuffle(self.text_indices)
            text = self.text_dataset[self.text_indices[self.current_index]]["text"]
            total_attempt += 1
            self.current_index += 1
            if len(text) == 0:
                continue
            sentences = nltk.sent_tokenize(text)
            sentence = self.rng.choice(sentences)
            sentence = self.detokenizer.detokenize(self.tokenizer.tokenize(sentence))[:self.max_str_len]
            if not is_ascii(sentence):
                continue
            
            discard = False
            for regex, count in regex_count.items():
                if regex == "TRIM":
                    continue
                r = self.regex_dict[regex]
                matches = list(r.finditer(sentence))
                if len(matches) < count:
                    discard = True
            if discard:
                continue

            self.env.set_new_state(sentence)
            try:
                exe(self.env)
                o = self.env.output_str
                if len(o) > 0 and o != sentence:
                    input_str.append(sentence)
                    output_str.append(o)
            except:
                pass
            if len(input_str) >= self.num_demo:
                break
        return input_str, output_str

    def generate_text_dataset(self, output_dir=None):
        self.setup_text()
        raw_datasets = {}
        for split in ["train", "val", "test"]:
            success_input = []
            success_program = []
            success_output = []
            programs = self.datasets[split]["program"]
            progress_bar = tqdm(range(self.split_num[split]), desc=f"Generation {split} set")
            for p in programs:
                if len(success_program) >= self.split_num[split]:
                    break
                i, o = self.generate_text_sample(p)
                if len(i) == self.num_demo:
                    success_input.append(i)
                    success_output.append(o)
                    success_program.append(p)
                    progress_bar.update(1)
            dataset = Dataset.from_dict({
                "program": success_program,
                "inputs": success_input,
                "outputs": success_output,
            })
            raw_datasets[split] = dataset
        raw_datasets = DatasetDict(raw_datasets)
        self.write_information(raw_datasets)
        self.write_five_sample(raw_datasets)
        output_dir = output_dir if output_dir is not None else self.output_dir
        raw_datasets.save_to_disk(output_dir)

    @torch.no_grad()
    def analysis(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        raw_datasets = DatasetDict.load_from_disk(self.input_dir)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        sim = {}
        for split in ["train", "val", "test"]:
            dataset = raw_datasets[split]
            program = dataset["program"]
            inputs = dataset["inputs"]
            outputs = dataset["outputs"]
            
            for i, (p, inp, out) in enumerate(zip(program, inputs, outputs)):
                embeddings = []
                for i, o in zip(inp, out):
                    io = tokenizer.encode_plus(i, o, return_tensors="pt")
                    for k in io:
                        io[k] = io[k].to(device)

                    io_emb = model(**io)["last_hidden_state"][0][0]
                    embeddings.append(io_emb)
                embeddings = torch.stack(embeddings)
                sim = pairwise_cosine_similarity(embeddings)
                for s in sim:
                    s = s.cpu().numpy().tolist()
                    row = " ".join(str(np.round(ss, 4)) for ss in s)
                    print(row)
                    

    def write_information(self, raw_datasets, output_dir=None, dataset_type="synthesis"):
        max_program_len = 0
        max_input_len = 0
        max_output_len = 0
        for split in raw_datasets.keys():
            program = raw_datasets[split]["program"]
            inputs = raw_datasets[split]["inputs"]
            outputs = raw_datasets[split]["outputs"]
            # with mp.Pool(20) as pool:
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
            program = raw_datasets[split]["program"]
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
            "program": train_program,
            "inputs": train_inputs,
            "outputs": train_outputs,
        })
        val_set = Dataset.from_dict({
            "program": val_program,
            "inputs": val_inputs,
            "outputs": val_outputs,
        })
        test_set = Dataset.from_dict({
            "program": test_program,
            "inputs": test_inputs,
            "outputs": test_outputs,
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

    def split_sets(self, examples):
        num_example = min([len(v) for v in examples.values()])
        num_train = min(num_example - self.num_val - self.num_test, self.num_train)
        num_train_val = num_train + self.num_val
        num_all_set = num_train_val + self.num_test
        train_set = {}
        val_set = {}
        test_set = {}
        other_set = {}
        for k in examples:
            train_set[k] = examples[k][:num_train]
            val_set[k] = examples[k][num_train:num_train_val]
            test_set[k] = examples[k][num_train_val:num_all_set]
            if num_train == self.num_train:
                other_set[k] = examples[k][num_all_set:]
        train_set = Dataset.from_dict(train_set)
        val_set = Dataset.from_dict(val_set)
        test_set = Dataset.from_dict(test_set)
        other_set = Dataset.from_dict(other_set)
        dataset = DatasetDict({
            "train": train_set,
            "val": val_set,
            "test": test_set,
            "other": other_set,
        })
        self.write_five_sample(dataset, self.output_dir)
        self.write_info(dataset)
        dataset.save_to_disk(self.output_dir)

    def write_info(self, raw_dataset):
        info = {}
        for k in raw_dataset.keys():
            info[k] = len(raw_dataset[k])
        with open(os.path.join(self.output_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

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
            "program": train_program,
            "inputs": train_inputs,
            "outputs": train_outputs,
        })
        val_set = Dataset.from_dict({
            "program": val_program,
            "inputs": val_inputs,
            "outputs": val_outputs,
        })
        test_set = Dataset.from_dict({
            "program": test_program,
            "inputs": test_inputs,
            "outputs": test_outputs,
        })

        if len(programs) > num_all_set:
            other_program = programs[num_all_set:]
            other_inputs = inputs[num_all_set:]
            other_outputs = outputs[num_all_set:]
            other_set = Dataset.from_dict({
                "program": other_program,
                "inputs": other_inputs,
                "outputs": other_outputs,
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

    def generate_execution_sample(self, code_pool, code_index):
        r = self.rng.random()
        input_str = self.random_gen_0() if r > 0.5 else self.random_gen_1()
        self.env.set_new_state(input_str)
        if self.current_index+self.num_demo > len(code_pool):
            self.current_index = 0
            self.rng.shuffle(code_pool)
        sample_program = [code_pool[i]
                          for i in code_index[self.current_index: self.current_index+self.num_demo]]
        self.current_index += self.num_demo
        executions = [self.yacc.parse(c) for c in sample_program]
        outputs = []
        for exe in executions:
            self.env.reset()
            try:
                exe(self.env)
                o = self.env.output_str
                if len(o) > 0 and o != input_str:
                    outputs.append(o)
                else:
                    outputs.append(None)
            except IndexError:
                outputs.append(None)

        success = [o is not None for o in outputs]
        success_rate = np.mean(success)
        if success_rate == 0:
            return input_str, [], []

        success_program = []
        success_output_str = []
        for p, o in zip(sample_program, outputs):
            if o is not None:
                success_program.append(p)
                success_output_str.append(o)

        while True:
            if len(success_program) >= self.num_demo:
                break
            if self.current_index + 1 >= len(code_pool):
                self.rng.shuffle(code_index)
                self.current_index = 0
            program = code_pool[code_index[self.current_index]]
            self.current_index += 1
            if not program in success_program:
                exe = self.yacc.parse(program)
                self.env.reset()
                try:
                    exe(self.env)
                    o = self.env.output_str
                    if len(o) > 0 and o != input_str:
                        success_program.append(program)
                        success_output_str.append(o)
                except IndexError:
                    pass

        return input_str, success_program, success_output_str

    def generate_execution_dataset(self, output_dir=None):
        num_sample_to_generate = self.num_train + self.num_val + self.num_test

        non_repeat = int(6 * (num_sample_to_generate))
        code_pool = list(set([self.random_code(
            max_length=self.max_prog_len) for _ in tqdm(range(non_repeat))]))
        code_index = np.arange(len(code_pool))
        self.rng.shuffle(code_index)
        success_input = []
        success_program = []
        success_output = []

        num_sample_to_generate = self.num_train + self.num_val + self.num_test
        progress_bar = tqdm(range(num_sample_to_generate))

        while True:
            if len(success_input) >= num_sample_to_generate:
                break
            input_str, programs, outputs = self.generate_execution_sample(
                code_pool, code_index)

            if len(programs) > 0:
                success_program.append(programs)
                success_input.append(input_str)
                success_output.append(outputs)
                progress_bar.update(1)


        num_train = min(len(success_program) - self.num_val -
                        self.num_test, self.num_train)
        num_train_val = num_train + self.num_val
        train_program = success_program[:num_train]
        train_inputs = success_input[:num_train]
        train_outputs = success_output[:num_train]

        val_program = success_program[num_train:num_train_val]
        val_inputs = success_input[num_train:num_train_val]
        val_outputs = success_output[num_train:num_train_val]

        test_program = success_program[num_train_val:]
        test_inputs = success_input[num_train_val:]
        test_outputs = success_output[num_train_val:]

        output_dir = output_dir if output_dir is not None else self.output_dir

        train_set = Dataset.from_dict({
            "program": train_program,
            "inputs": train_inputs,
            "outputs": train_outputs,
        })
        val_set = Dataset.from_dict({
            "program": val_program,
            "inputs": val_inputs,
            "outputs": val_outputs,
        })
        test_set = Dataset.from_dict({
            "program": test_program,
            "inputs": test_inputs,
            "outputs": test_outputs,
        })

        dataset = DatasetDict({
            "train": train_set,
            "val": val_set,
            "test": test_set,
        })
        self.write_information(dataset, dataset_type="execution")
        self.write_five_sample(dataset)
        dataset.save_to_disk(output_dir)


    def generate_synthesis_sample_uniform(self, code, tries=0):
        leakage = self.leakage_list[tries]
        program_stats = self.get_stats(code)
        inputs = [self.generate_input_str(program_stats, leakage=leakage)
                for _ in range(self.num_demo)]

        # compile code
        exec_fn = self.yacc.parse(code)

        # execute
        outputs = []
        traces = []
        eh = None
        for input_str in inputs:
            self.env.set_new_state(input_str)
            try:
                exec_fn(self.env)
                tr = self.env.s_h
                eh = self.env.ex_h
                o = self.env.output_str
                if len(o) != 0:
                    outputs.append(o)
                    traces.append(tr)
                else:
                    outputs.append(None)
                    traces.append(None)
            except IndexError:
                traces.append(None)
                outputs.append(None)

        success = [o is not None for o in outputs]
        success_rate = np.mean(success)

        if success_rate == 0:
            # If random sampling can not generate any program, discard this program
            return [], []

        b_count = {}
        total_seen = [0]
        def get_importance_b(b):
            p_min = 1.0
            p_curr = 1.0


            total_seen[0] += 1
            if b in b_count:
                b_count[b] += 1
            else:
                b_count[b] = 1
            
            p_min = p_min * (np.min(list(b_count.values())) / total_seen[0])
            p_curr = p_curr * (b_count[b] / total_seen[0])
            g = (p_min + self.tol) / (p_curr + self.tol)
            return g
        

        success_input_str = []
        success_output_str = []
        for i, o, tr in zip(inputs, outputs, traces):
            if len(success_input_str) == self.num_demo:
                return success_input_str, success_output_str
            if o is not None:
                behavior = execution_behavior(i, tr, eh)
                g = get_importance_b(behavior)
                r = self.rng.random()
                if r < g:
                    success_input_str.append(i)
                    success_output_str.append(o)

        attempt = 20
        max_attempt = self.maximum_attempt * (tries + 1)

        while True:
            if len(success_input_str) >= self.num_demo:
                break
            if attempt > max_attempt:
                # If random sampling can not generate enough program, discard this program
                return [], []
            input_str = self.generate_input_str(program_stats, leakage=leakage)
            self.env.set_new_state(input_str)
            attempt += 1
            try:
                exec_fn(self.env)
                tr = self.env.s_h
                eh = self.env.ex_h
                o = self.env.output_str
                if len(o) != 0:
                    behavior = execution_behavior(input_str, tr, eh)
                    g = get_importance_b(behavior)
                    r = self.rng.random()
                    if r < g:
                        success_input_str.append(input_str)
                        success_output_str.append(o)
            except IndexError:
                pass

        return success_input_str, success_output_str


    def generate_synthesis_sample_with_retry(self, code):
        tries = 0
        inputs, outputs = self.generate_synthesis_sample_uniform(code)
        while len(inputs) < self.num_demo and tries < self.num_retry:
            tries += 1
            inputs, outputs = self.generate_synthesis_sample_uniform(code, tries)
        return inputs, outputs

    def _get_similar(self, program_pair):
        [p1, p2] = program_pair
        if p1 == p2:
            return None
        f1 = self.pf.yacc.parse(p1)
        f2 = self.pf.yacc.parse(p2)
        if f1 in self.unique or f2 in self.unique:
            return None
        exec1 = self.yacc.parse(p1)
        exec2 = self.yacc.parse(p2)
        p1_reg, p1_sch = self.get_stats(p1)
        p2_reg, p2_sch = self.get_stats(p2)

        for k, v in p2_reg.items():
            p1_reg[k] += v
        for k, v in p2_sch.items():
            p1_sch[k] += v
        
        inputs = [self.generate_input_str((p1_reg, p1_sch))
            for _ in range(self.maximum_attempt)]

        output1, output2 = [], []
        same = []
        different = []
        distances = []
        for i in inputs:
            self.env.set_new_state(i)
            try:
                exec1(self.env)
                o = self.env.output_str
                if len(o) != 0:
                    output1.append(o)
                else:
                    output1.append(None)
            except IndexError:
                output1.append(None)
            self.env.set_new_state(i)
            try:
                exec2(self.env)
                o = self.env.output_str
                if len(o) != 0:
                    output2.append(o)
                else:
                    output2.append(None)
            except IndexError:
                output2.append(None)
        
        for i in range(len(inputs)):
            if output1[i] == output2[i] and output1[i] != None:
                same.append(i)
            elif output1[i] != None and output2[i] != None:
                different.append(i)
                distances.append(ed.eval(output1[i], output2[i]) / len(output1[i]))
        
        inputs = np.array(inputs)
        output1 = np.array(output1)
        output2 = np.array(output2)
        different = np.array(different)
        same = np.array(same)
        distances = np.argsort(distances)

        input_str = []
        output_str1 = []
        output_str2 = []
        
        if len(same) > 0:
            input_str.extend(inputs[same[:4]])
            output_str1.extend(output1[same[:4]])
            output_str2.extend(output2[same[:4]])
            remain = self.num_demo - len(input_str)
            selected = different[distances[:remain]]
            if len(selected) < remain:
                return None
            input_str.extend(inputs[selected])
            output_str1.extend(output1[selected])
            output_str2.extend(output2[selected])
            return (p1, p2, input_str, output_str1, output_str2)
        return None

    def generate_same_io_nps(self):
        raw_datasets = DatasetDict.load_from_disk(self.input_dir)
        programs = raw_datasets["train"]["program"]
        if "other" in raw_datasets.keys():
            programs += raw_datasets["other"]["program"]

        success_program = []
        success_inputs = []
        success_outputs = []

        pairset = set()

        num_program_to_generate = self.num_train + self.num_val + self.num_test



        all_program_pair = self.rng.choice(programs, num_program_to_generate * 20 * 2)
        all_program_pair = np.reshape(all_program_pair, (-1, 2))

        with mp.Pool(20) as pool:
            result = pool.map(self._get_similar, tqdm(all_program_pair))

        for res in result:
            if res is not None:
                p1, p2, input_str, output_str1, output_str2 = res
                if (p1, p2) in pairset:
                    continue
                else:
                    pairset.add((p1, p2))
                    pairset.add((p2, p1))

                    success_program.append(p1)
                    success_inputs.append(input_str)
                    success_outputs.append(output_str1)

                    success_program.append(p2)
                    success_inputs.append(input_str)
                    success_outputs.append(output_str2)

        self.write_dataset_with_other(success_program, success_inputs, success_outputs, self.output_dir)


    def get_similar(self, program_pair):
        [p1, p2] = program_pair
        if p1 == p2:
            return None
        exec1 = self.yacc.parse(p1)
        exec2 = self.yacc.parse(p2)
        p1_reg, p1_sch = self.get_stats(p1)
        p2_reg, p2_sch = self.get_stats(p2)

        for k, v in p2_reg.items():
            p1_reg[k] += v
        for k, v in p2_sch.items():
            p1_sch[k] += v
        
        inputs = [self.generate_input_str((p1_reg, p1_sch))
            for _ in range(self.maximum_attempt)]

        output1, output2 = [], []
        same = []
        different = []
        distances = []
        for i in inputs:
            self.env.set_new_state(i)
            try:
                exec1(self.env)
                o = self.env.output_str
                if len(o) != 0:
                    output1.append(o)
                else:
                    output1.append(None)
            except IndexError:
                output1.append(None)
            self.env.set_new_state(i)
            try:
                exec2(self.env)
                o = self.env.output_str
                if len(o) != 0:
                    output2.append(o)
                else:
                    output2.append(None)
            except IndexError:
                output2.append(None)
        
        for i in range(len(inputs)):
            if output1[i] == output2[i] and output1[i] != None:
                same.append(i)
            elif output1[i] != None and output2[i] != None:
                different.append(i)
                distances.append(ed.eval(output1[i], output2[i]) / len(output1[i]))

        if len(same) > 0 and len(different) > 0:
            if len(same) < self.num_demo:
                for _ in range(self.maximum_attempt):
                    input_str = self.generate_input_str((p1_reg, p1_sch))
                    self.env.set_new_state(input_str)
                    try:
                        exec1(self.env)
                        o1 = self.env.output_str
                    except IndexError:
                        o1 = None
                    self.env.set_new_state(input_str)
                    try:
                        exec2(self.env)
                        o2 = self.env.output_str
                    except IndexError:
                        o2 = None

                    if o1 != None and o1 == o2 and len(o1) > 0:
                        same.append(len(inputs))
                        inputs.append(input_str)
                        output1.append(o1)
                        output2.append(o2)
                    
                    elif len(different) < self.num_demo and o1 != None and o2 != None and o1 != o2 and len(o1) > 0 and len(o2) > 0:
                        different.append(len(inputs))
                        inputs.append(input_str)
                        output1.append(o1)
                        output2.append(o2)

        
            inputs = np.array(inputs)
            output1 = np.array(output1)
            output2 = np.array(output2)
            different = np.array(different)
            # print(different)
            same = np.array(same)
            distances = np.argsort(distances)

            same_input_str = inputs[same[:self.num_demo]]
            same_output_str = output1[same[:self.num_demo]]
            different_input_str = inputs[different[:self.num_demo]]
            output_str1 = output1[different[:self.num_demo]]
            output_str2 = output2[different[:self.num_demo]]
            return (p1, p2, same_input_str, same_output_str, different_input_str, output_str1, output_str2)
        return None

    def generate_same_io(self):
        raw_datasets = DatasetDict.load_from_disk(self.input_dir)
        programs = raw_datasets["train"]["program"]
        if "other" in raw_datasets.keys():
            programs += raw_datasets["other"]["program"]

        program1 = []
        program2 = []
        same_inputs = []
        same_outputs = []
        different_inputs = []
        different_outputs_1 = []
        different_outputs_2 = []


        pairset = set()

        num_program_to_generate = self.num_train + self.num_val + self.num_test

        all_program_pair = self.rng.choice(programs, num_program_to_generate * 20 * 2)
        all_program_pair = np.reshape(all_program_pair, (-1, 2))

        with mp.Pool(20) as pool:
            result = pool.map(self.get_similar, tqdm(all_program_pair))

        for res in result:
            if res is not None:
                p1, p2, same_input_str, same_output_str, different_input_str, output_str1, output_str2 = res
                if (p1, p2) in pairset:
                    continue
                else:
                    pairset.add((p1, p2))
                    pairset.add((p2, p1))

                    program1.append(p1)
                    program2.append(p2)
                    same_inputs.append(same_input_str)
                    same_outputs.append(same_output_str)
                    different_inputs.append(different_input_str)
                    different_outputs_1.append(output_str1)
                    different_outputs_2.append(output_str2)

        examples = {
            "program1": program1,
            "program2": program2,
            "same_input": same_inputs,
            "same_output": same_outputs,
            "different_input": different_inputs,
            "different_output1": different_outputs_1,
            "different_output2": different_outputs_2,
        }

        self.split_sets(examples)

    def test(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=600)
    parser.add_argument("--num_val", type=int, default=100)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--min_str_len", type=int, default=20)
    parser.add_argument("--max_str_len", type=int, default=30)
    parser.add_argument("--tol", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_demo", type=int, default=20)
    parser.add_argument("--max_prog_len", type=int, default=1)
    parser.add_argument("--max_length_only", action="store_true")
    parser.add_argument("--long_program", action="store_true")
    parser.add_argument("--synthesis", action="store_true")
    parser.add_argument("--text_dataset", action="store_true")
    parser.add_argument("--leakage_dataset", action="store_true")
    parser.add_argument("--text_dataset_name", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--text_dataset_path", type=str, default="wikitext")
    parser.add_argument("--execution", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--maximum_attempt", type=int, default=100)
    parser.add_argument("--num_retry", type=int, default=10)
    parser.add_argument("--with_retry", action="store_true")
    parser.add_argument("--leakage", type=float, default=1.0)
    parser.add_argument("--non_uniform", action="store_true")
    parser.add_argument("--all_program", action="store_true")
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--same_io", action="store_true")
    args = parser.parse_args()

    generator = Generator(args)
    if args.synthesis:
        generator.generate_synthesis_dataset()
    elif args.execution:
        generator.generate_execution_dataset()
    elif args.text_dataset:
        generator.generate_text_dataset()
    elif args.leakage_dataset:
        generator.leakage_experiment()
    elif args.entropy:
        res = generator.dataset_entropy()
        print(res)
    elif args.analysis:
        generator.analysis()
    elif args.same_io:
        generator.generate_same_io()
    else:
        generator.test()
