import re
import numpy as np
from typing import List

from string_trans.consts import REGEX_DICT

def get_matches(reg, input_str):
    if reg[0] == '"':
        try:
            if reg == '"SPACE"':
                p = re.compile(" ")
            elif reg[1] == ".":
                p = re.compile("\\.")
            elif reg[1] == "$":
                p = re.compile("\$")
            else:
                p = re.compile(reg[1:-1])
        except re.error:
            p = re.compile('\\' + reg[1:-1])
    else:
        p = re.compile(REGEX_DICT[reg])
    matches = list(p.finditer(input_str))
    return matches

def get_match_str(type, input_str):
    p = re.compile(REGEX_DICT[type])
    matches = list(p.findall(input_str))
    return matches


def sch2str(schar):
    if schar[1:-1] == "SPACE":
        return " "
    else:
        return schar[1:-1]

class StringTransEnv(object):
    def __init__(self, input_str="", seed=123):
        self._input_str = input_str
        self.nesting_mode = False
        self.top = input_str
        self._seed = seed
        self._expression_outputs: List[str] = []
        self.current_length = 0
        self.s_h = []
        self.ex_h = []

    def set_new_state(self, s):
        self._input_str = s
        self.reset()

    def const_str(self, schar):
        if schar == '"SPACE"':
            schar ='" "'
        s = schar[1:-1]
        self._expression_outputs.append(s)
        self.s_h.append("".join(self._expression_outputs))
        self.ex_h.append(self.current_length)

    def set_nesting(self):
        self.nesting_mode = True
    
    def unset_nesting(self):
        self.nesting_mode = False

    def nesting(self, nesting_function, reg=None, type=None, index=None, case=None, sch1=None, sch2=None):
        if self.nesting_mode:
            _input_str: str = self._expression_outputs[-1]
        else:
            _input_str: str = self._input_str

        if nesting_function == 'GetToken':
            s = get_match_str(type, _input_str)[index]
        elif nesting_function == 'ToCase':
            if case == "Proper":
                s = _input_str.capitalize()
            elif case == "Allcaps":
                s = _input_str.upper()
            elif case == "Lower":
                s = _input_str.lower()
        elif nesting_function == 'Replace':
            s = _input_str.replace(sch2str(sch1), sch2str(sch2))
        elif nesting_function == 'Trim':
            s = _input_str.strip()
        elif nesting_function == 'GetUpTo':
            matches = get_matches(reg, _input_str)
            if len(matches) == 0:
                s = ""
            else:
                s = _input_str[: matches[0].span()[1]]
        elif nesting_function == 'GetFrom':
            matches = get_matches(reg, _input_str)
            if len(matches) == 0:
                s = ""
            else:
                s = _input_str[matches[0].span()[1] :]
        elif nesting_function == 'GetFirst':
            matches = get_match_str(type, _input_str)
            s = ''.join(matches[:index+1])
        elif nesting_function == 'GetAll':
            matches = get_match_str(type, _input_str)
            s = ' '.join(matches)

        if self.nesting_mode:
            self._expression_outputs[-1] = s
        else:
            self._expression_outputs.append(s)

        self.s_h.append("".join(self._expression_outputs))
        self.ex_h.append(self.current_length)

    def sub_str(self, pos1, pos2):

        if len(self._input_str[pos1:pos2]) == 0:
            raise IndexError("pos1 > pos2")
        s = self._input_str[pos1: pos2]
        self._expression_outputs.append(s)

        self.s_h.append("".join(self._expression_outputs))
        self.ex_h.append(self.current_length)

    def regex(self, reg, index, boundary):
        match = get_matches(reg, self._input_str)[index]

        if boundary == "Start":
            return match.span()[0]
        else:
            return match.span()[1]

    @property
    def output_str(self):
        return "".join(self._expression_outputs)

    @property
    def input_str(self):
        return self._input_str

    def reset(self):
        self.s_h = []
        self.ex_h = []
        self.nesting_mode = False
        self._expression_outputs = []
        self._output_str = ""
        self.current_length = 0
