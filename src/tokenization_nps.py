# coding=utf-8
# Copyright 2021 T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model ByT5."""


import warnings
from typing import Dict, List, Optional, Tuple

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging
from string_trans.dsl import StringTransformationDSL
from string_trans.consts import MIN_INT, MAX_INT, INT_PREFIX, SCHAR_LIST

logger = logging.get_logger(__name__)

# IOTokenizer mainly comes from ByT5Tokenizr


class IOTokenizer(PreTrainedTokenizer):
    """
    Construct a IO tokenizer. IO simply uses raw bytes utf-8 encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        inp_token (`str`, *optional*, defaults to `"<inp>"`)
            The token used for intruct input string.
        out_token (`str`, *optional*, defaults to `"<out>"`)
            The token used for intruct output string.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        inp_token="<inp>",
        out_token="<out>",
        **kwargs
    ) -> None:

        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(
            pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(
            eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(
            unk_token, str) else unk_token
        additional_special_tokens = [inp_token, out_token]

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.inp_token = inp_token
        self.out_token = out_token

        self._ascii_vocab_size = 128  # ascii 128

        # define special tokens dict, eos token must have largest id
        self.special_tokens_encoder: Dict[str, int] = {
            self.pad_token: 131,
            self.eos_token: 132,
            self.unk_token: 130,
            self.inp_token: 128,
            self.out_token: 129,
        }
        self._num_special_tokens = len(self.special_tokens_encoder)
        self.special_tokens_decoder: Dict[int, str] = {
            v: k for k, v in self.special_tokens_encoder.items()}
        self.inp_token_id = self.special_tokens_encoder[self.inp_token]
        self.out_token_id = self.special_tokens_encoder[self.out_token]

    @property
    def vocab_size(self):
        return self._ascii_vocab_size + self._num_special_tokens

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. IOTokenizer
        make use of token type ids.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        else:
            inp = [self.inp_token_id]
            out = [self.out_token_id]
            return len(inp + token_ids_0 + out) * [0] + len(token_ids_1 + eos) * [1]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `<inp> A <out> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            token_ids_0 = self._add_eos_if_not_present(token_ids_0)
            return token_ids_0
        else:
            inp = [self.inp_token_id]
            out = [self.out_token_id]
            token_ids = inp + token_ids_0 + out + token_ids_1
            token_ids = self._add_eos_if_not_present(token_ids)
            return token_ids

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.special_tokens_encoder:
            token_id = self.special_tokens_encoder[token]
        elif token in self.added_tokens_encoder:
            token_id = self.added_tokens_encoder[token]
        elif len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token)
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.special_tokens_decoder:
            token = self.special_tokens_decoder[index]
        else:
            token = chr(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self.special_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.special_tokens_encoder:
                tok_string = token.encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # IOTokenizer has no vocab file just like ByT5Tokenizer
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()


class ProgramTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self,
                 bos_token="<s>",
                 eos_token="</s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 dsl_class=StringTransformationDSL,
                 **kwargs
                 ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(
            bos_token, str) else pad_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(
            pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(
            eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(
            unk_token, str) else unk_token
        additional_special_tokens = []
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.dsl = dsl_class()
        self.program_tokens = self.dsl.construct_vocab()
        self.program_tokens_encoder: Dict[str, int] = {}
        for i, token in enumerate(self.program_tokens):
            self.program_tokens_encoder[token] = i
        self.program_tokens_decoder: Dict[int, str] = {
            v: k for k, v in self.program_tokens_encoder.items()}
        self._num_program_tokens = len(self.program_tokens_encoder)

        self.special_tokens_encoder: Dict[str, int] = {
            self.unk_token: 0 + self._num_program_tokens,
            self.pad_token: 1 + self._num_program_tokens,
            self.bos_token: 2 + self._num_program_tokens,
            self.eos_token: 3 + self._num_program_tokens,
        }
        self._num_special_tokens = len(self.special_tokens_encoder)
        self.special_tokens_decoder: Dict[int, str] = {
            v: k for k, v in self.special_tokens_encoder.items()}

    @property
    def vocab_size(self) -> int:
        return self._num_program_tokens + self._num_special_tokens

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def _add_bos_if_not_present(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) > 0 and token_ids[0] == self.bos_token_id:
            return token_ids
        else:
            return [self.bos_token_id] + token_ids

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]
        bos = [self.bos_token_id]

        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_bos_if_not_present(token_ids_0)
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = text.strip().split(" ")
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.special_tokens_encoder:
            token_id = self.special_tokens_encoder[token]
        elif token in self.added_tokens_encoder:
            token_id = self.added_tokens_encoder[token]
        else:
            token_id = self.program_tokens_encoder[token]
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.special_tokens_decoder:
            token = self.special_tokens_decoder[index]
        else:
            token = self.program_tokens_decoder[index]
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        string = []
        for token in tokens:
            if token in self.special_tokens_decoder:
                tok_string = self.special_tokens_decoder[token]
            elif token in self.added_tokens_decoder:
                tok_string = self.special_tokens_decoder[token]
            elif token in self.special_tokens_encoder:
                tok_string = token
            elif token in self.added_tokens_encoder:
                tok_string = token
            elif token in self.program_tokens_encoder:
                tok_string = token
            else:
                tok_string = self.program_tokens_decoder[token]
            string.append(tok_string)
        string = " ".join(string)
        return string

    # ProgramTokenizer has no vocab file, vocab is hard coded in dsl.py
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()


class BertIOTokenizer(IOTokenizer):
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]


class BertProgramTokenizer(ProgramTokenizer):
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
