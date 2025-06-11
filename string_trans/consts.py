# consts.py
import string
from copy import deepcopy
import numpy as np
import json

REGEX_DICT = {
    "Word": "[A-Za-z]+",
    "Num": "\d+",
    "Alphanum": "[A-Za-z0-9]+",
    "Allcaps": "[A-Z]+",
    "Propcase": "[A-Z][a-z]+",
    "Lower": "[a-z]+",
    "Digit": "\d",
    "Char": "[A-Za-z0-9]"
}

REGEX_LIST = list(REGEX_DICT.keys())

SCHAR_CHAR = "&,.?!@()[]%{}/:;$#'"
SCHAR_LIST = list(SCHAR_CHAR)
SCHAR_LIST.append("SPACE")

# character
ALPHANUM_CHAR = []
ALPHANUM_CHAR.extend(string.ascii_letters)
ALPHANUM_CHAR.extend(string.digits)
SCHAR_CHAR = list(SCHAR_CHAR) + [" "]
WORD_CHAR = list(string.ascii_letters)
LOWER_CHAR = list(string.ascii_letters[:26])
UPPER_CHAR = list(string.ascii_letters[26:])
INT_CHAR = list(string.digits)

ALL_CHAR = deepcopy(SCHAR_CHAR)
ALL_CHAR.extend(ALPHANUM_CHAR)

INT2CHAR = ["<pad>"] + ALL_CHAR
CHAR2INT = {s: INT2CHAR.index(s) for s in INT2CHAR}


MIN_INT = -30
MAX_INT = 30
MAX_INDEX = 5
MAX_POSITION = MAX_INT

INT_PREFIX = ''

SCHAR_PROB = 0.25

def sample_prob():
    # probs for schar
    schar_prob = np.ones(len(SCHAR_CHAR))
    # schar_prob sum to all_schar_prob
    schar_prob /= schar_prob.sum()
    schar_prob *= SCHAR_PROB

    # probs for alphanum
    alphanum_probs = np.ones(len(ALPHANUM_CHAR))
    alphanum_probs /= alphanum_probs.sum()
    alphanum_probs *= (1 - SCHAR_PROB)

    return np.concatenate((schar_prob, alphanum_probs))

SAMPLE_PROB = sample_prob()
