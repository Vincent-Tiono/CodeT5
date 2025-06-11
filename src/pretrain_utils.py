import logging
from typing import Dict, List

import ipdb
from omegaconf import DictConfig
from transformers.models.clip.configuration_clip import CLIPTextConfig

from hprl_karel_env.dsl.dsl_prob_option_v2 import DSLProb_option_v2
from src.modeling_karel import KarelIOCLIPModel, KarelDemoCLIPModel
from src.modeling_string_trans import StringCLIPModel, StringMixCLIPModel, StringDissimilarCLIPModel
from src.configuration_string_trans import StringCLIPConfig
from src.tokenization_nps import IOTokenizer, ProgramTokenizer
from string_trans.dsl import StringTransformationDSL

logger = logging.getLogger(__name__)


def get_string_model(model_args: DictConfig):
    io_tokenizer = IOTokenizer()
    p_tokenizer = ProgramTokenizer(dsl_class=StringTransformationDSL)
    if model_args.hard_sample_only:
        model_class = StringDissimilarCLIPModel
    elif model_args.use_similar_dataset:
        model_class = StringMixCLIPModel
    else:
        model_class = StringCLIPModel
    if model_args.model_name_or_path:
        nps_config = StringCLIPConfig.from_pretrained(
            model_args.model_name_or_path)
        model = model_class.from_pretrained(model_args.model_name_or_path)
    else:
        io_config = CLIPTextConfig(
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.intermediate_size,
            projection_dim=model_args.projection_dim,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            vocab_size=io_tokenizer.vocab_size,
            max_position_embeddings=512,
        )
        program_config = CLIPTextConfig(
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.intermediate_size,
            projection_dim=model_args.projection_dim,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            vocab_size=p_tokenizer.vocab_size,
            max_position_embeddings=512,
            eos_token_id=p_tokenizer.eos_token_id,
        )
        nps_config = StringCLIPConfig.from_program_io_configs(
            program_config=program_config,
            io_config=io_config,
            projection_dim=model_args.projection_dim,
        )
        model = model_class(nps_config)
    return nps_config, p_tokenizer, io_tokenizer, model


def get_karel_io_model(model_args: DictConfig):
    # config
    tokenizer = ProgramTokenizer(dsl_class=DSLProb_option_v2)
    config = CLIPTextConfig(
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        projection_dim=model_args.projection_dim,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
    )
    config.num_channels = model_args.num_channels
    model = KarelIOCLIPModel(config)
    return config, tokenizer, tokenizer, model

def get_karel_demo_model(model_args: DictConfig):
    # config
    tokenizer = ProgramTokenizer(dsl_class=DSLProb_option_v2)
    config = CLIPTextConfig(
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        projection_dim=model_args.projection_dim,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
    )
    config.num_channels = model_args.num_channels
    model = KarelDemoCLIPModel(config)
    return config, tokenizer, tokenizer, model

def preprocess_string_mix_or_hard(examples, program_tokenizer: ProgramTokenizer, io_tokenizer: IOTokenizer):
    tokenized_program = program_tokenizer(examples["program"], padding=False)
    tokenized_program["program"] = examples["program"]
    tokenized_inputs_outputs = []
    for inputs, outputs in zip(examples["inputs"], examples["outputs"]):
        result = io_tokenizer(text=inputs, text_pair=outputs)
        tokenized_inputs_outputs.append(result.input_ids)
    tokenized_program["inputs_outputs"] = tokenized_inputs_outputs
    tokenized_program["inputs"] = examples["inputs"]
    tokenized_program["outputs"] = examples["outputs"]
    tokenized_program["similar_program"] = examples["similar_program"]

    tokenized_io = []
    similar_input = examples["similar_input"]
    similar_output = examples["similar_output"]
    for i, o in zip(similar_input, similar_output):
        result = io_tokenizer(text=i, text_pair=o)
        tokenized_io.append(result.input_ids)
    tokenized_program["similar_io"] = tokenized_io
    return tokenized_program


def preprocess_dissimilar_program(dissimilar_program, tokenized_program: List[Dict]):
    tokenized_dissimilar_program = []
    for pid in dissimilar_program:
        tokenized_dissimilar_program.append(tokenized_program[pid])
    return tokenized_dissimilar_program
