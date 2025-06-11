from __future__ import annotations
from typing_extensions import Self
from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Self | str) -> bool:
        return str(self) == str(other)


class PretrainScript(StrEnum):
    PROGRAM_IO = "program_io"
    PROGRAM_INPUT_OUTPUT = "program_input_output"


class EvalFunctions(StrEnum):
    EVAL = "eval"
    PATTERN = "pattern"
    COPY = "copy"
    ALIAS = "alias"


class TrainScript(StrEnum):
    SEQ2SEQ_NPS = "seq2seq_nps"
    SEQ2SEQ_NPSE = "seq2seq_npse"
    SEQ2SEQ_NPE = "seq2seq_npe"
    CLASSIFICATION  = "classification"


class TestScripts(StrEnum):
    SEQ2SEQ = "seq2seq"

class VisualizationScripts(StrEnum):
    CLIP_EMBED = "clip_embedding"

class DevelopmentScripts(StrEnum):
    TOP_k = "top_k"
    DIFFICULTY = "difficulty"
    SIMILAR_ACCURACY = "similar_accuracy"
    SO_FUN = "so_fun"
    SEQ2SEQ = "seq2seq_nps"
