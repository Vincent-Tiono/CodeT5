import logging

from transformers import (CONFIG_MAPPING, AutoConfig, AutoModelForSeq2SeqLM,
                          AutoTokenizer, T5Config)
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig
from transformers.models.clip import CLIPTextConfig
from transformers.models.t5 import T5ForConditionalGeneration

from src.configuration_string_trans import (Blip2NPSConfig, StringCLIPFusedConfig,
                                            StringCLIPFusedConfig)
from src.modeling_string_trans import (
    Blip2NPSForConditionalGeneration, StringCLIPFusedFullSequenceModel,
    StringCLIPMappingT5DecoderModel, StringCLIPModel,
    StringCLIPFusedModel, StringCLIPFusedT5DecoderModel,
    ContrastiveAugmentedFixCLIPNPSModel,
    ContrastiveAugmentedNPSModel)
from src.modeling_karel import KarelIOSeq2SeqModel, KarelIOFusedModel, KarelIOCLIPModel, KarelDemoSeq2seqModel, KarelDemoFusedModel, KarelDemoCLIPModel
from src.configuration_karel import KarelFusedConfig
from src.tokenization_nps import IOTokenizer, ProgramTokenizer
from hprl_karel_env.dsl.dsl_prob_option_v2 import DSLProb_option_v2
import ipdb

logger = logging.getLogger(__name__)


def get_string_model(model_args):
    # config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, use_fast=not model_args.use_slow_tokenizer)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # model
    if model_args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    return config, tokenizer, tokenizer, model


def get_string_fused_model(model_args):
    model_class = None
    if model_args.model_type == "contrastive-nps-full-sequence":
        model_class = StringCLIPFusedFullSequenceModel
    elif model_args.model_type == "contrastive-nps-end-token":
        model_class = StringCLIPFusedModel
    elif model_args.model_type == "contrastive-nps-t5-decoder":
        model_class = StringCLIPFusedT5DecoderModel
    elif model_args.model_type == "contrastive-nps-mapping-t5-decoder":
        model_class = StringCLIPMappingT5DecoderModel
    elif model_args.model_type == "contrastive-nps-augmented":
        model_class = ContrastiveAugmentedNPSModel
    elif model_args.model_type == "contrastive-nps-augmented-fix-clip":
        model_class = ContrastiveAugmentedFixCLIPNPSModel
    else:
        raise NotImplementedError(f"Model name {model_args.model_type} not found")
    if model_args.contrastive_path is not None:
        contrastive_config = StringCLIPFusedConfig.from_pretrained(
            model_args.contrastive_path)
        if model_args.model_name_or_path is not None:
            t5_config = T5Config.from_pretrained(model_args.model_name_or_path)
            config = StringCLIPFusedConfig.from_contrastive_t5_configs(
                contrastive_config, t5_config)
            model = model_class(config)
            model.encoder = StringCLIPModel.from_pretrained(
                model_args.contrastive_path)
            model.decoder = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer)
            io_tokenizer = IOTokenizer()
            return config, tokenizer, io_tokenizer, model
        else:
            t5_config = T5Config.from_pretrained(model_args.config_name)
            config = StringCLIPFusedConfig.from_contrastive_t5_configs(
                contrastive_config, t5_config)
            model = model_class(config)
            model.encoder = StringCLIPModel.from_pretrained(
                model_args.contrastive_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.config_name, use_fast=not model_args.use_slow_tokenizer)
            io_tokenizer = IOTokenizer()
            return config, tokenizer, io_tokenizer, model
    else:
        p_tokenizer = ProgramTokenizer()
        io_tokenizer = IOTokenizer()
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
        )
        contrastive_config = StringCLIPFusedConfig.from_program_io_configs(
            program_config=program_config,
            io_config=io_config,
        )
        if model_args.model_name_or_path is not None:
            t5_config = T5Config.from_pretrained(model_args.model_name_or_path)
            config = StringCLIPFusedConfig.from_contrastive_t5_configs(
                contrastive_config, t5_config)
            model = model_class(config)
            model.decoder = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer)
            io_tokenizer = IOTokenizer()
            return config, tokenizer, io_tokenizer, model
        else:
            t5_config = T5Config.from_pretrained(model_args.config_name)
            config = StringCLIPFusedConfig.from_contrastive_t5_configs(
                contrastive_config, t5_config)
            model = model_class(config)
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.config_name, use_fast=not model_args.use_slow_tokenizer)
            io_tokenizer = IOTokenizer()
            return config, tokenizer, io_tokenizer, model


def get_blip_2_model(model_args):
    io_config = StringCLIPFusedConfig.from_pretrained(
        model_args.contrastive_path).io_config
    qformer_config = Blip2QFormerConfig(
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
    )
    text_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    blip_2_config = Blip2NPSConfig.from_io_qformer_text_configs(
        io_config=io_config,
        qformer_config=qformer_config,
        text_config=text_config,
    )
    model = Blip2NPSForConditionalGeneration(blip_2_config)
    model.io_model = StringCLIPModel.from_pretrained(
        model_args.contrastive_path).io_model
    model.language_model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path)
    io_tokenizer = IOTokenizer()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer)
    return blip_2_config, tokenizer, io_tokenizer, model


def get_karel_io_model(model_args):
    # config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, use_fast=not model_args.use_slow_tokenizer)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    config.num_channels = model_args.num_channels
    config.encoder_type = model_args.encoder_type
    model_class = None
    if model_args.model_type.startswith("KarelIO"):
        model_class = KarelIOSeq2SeqModel
    elif model_args.model_type.startswith("KarelDemo"):
        model_class = KarelDemoSeq2seqModel
    else:
        raise ValueError(f"Model type {model_args.model_type} not supported")


    model = model_class(config)
    if model_args.model_name_or_path:
        model.decoder = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path)

    return config, tokenizer, tokenizer, model

def get_karel_io_fused_model(model_args):
    model_class = None
    contrastive_class = None
    if model_args.model_type == "KarelIOFused":
        model_class = KarelIOFusedModel
        contrastive_class = KarelIOCLIPModel
    elif model_args.model_type == "KarelDemoFused":
        model_class = KarelDemoFusedModel
        contrastive_class = KarelDemoCLIPModel
    else:
        raise ValueError(f"Model type {model_args.model_type} not supported")

    # config
    p_tokenizer = ProgramTokenizer(dsl_class=DSLProb_option_v2)
    if model_args.contrastive_path is not None:
        contrastive_config = CLIPTextConfig.from_pretrained(model_args.contrastive_path)
    else:
        contrastive_config = CLIPTextConfig(
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.intermediate_size,
            projection_dim=model_args.projection_dim,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            vocab_size=p_tokenizer.vocab_size,
            max_position_embeddings=512,
        )

    contrastive_config.num_channels = model_args.num_channels
    contrastive_config.encoder_type = model_args.encoder_type
    
    if model_args.model_name_or_path is not None:
        t5_config = T5Config.from_pretrained(model_args.model_name_or_path)
    else:
        t5_config = T5Config.from_pretrained(model_args.config_name)
    config = KarelFusedConfig.from_cliptext_t5_configs(cliptext_config=contrastive_config, t5_config=t5_config)

    # tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, use_fast=not model_args.use_slow_tokenizer)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    model = model_class(config)
    if model_args.model_name_or_path:
        model.decoder = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    if model_args.contrastive_path is not None:
        model.contrastive_model = contrastive_class.from_pretrained(model_args.contrastive_path)

    return config, tokenizer, tokenizer, model




# TODO: add more models
MODEL_MAP = {
    "String": get_string_model,
    "StringMismatch": get_string_model,
    "StringMismatchAugmented": get_string_model,
    "StringDissimilar": get_string_model,
    "StringDissimilarAugmented": get_string_model,
    "StringFused": get_string_fused_model,
    "StringFusedAugmented": get_string_fused_model,
    "StringFusedMismatch": get_string_fused_model,
    "StringFusedMismatchAugmented": get_string_fused_model,
    "StringFusedDissimilar": get_string_fused_model,
    "StringFusedDissimilarAugmented": get_string_fused_model,
    "KarelIO": get_karel_io_model,
    "KarelIOPaired": get_karel_io_model,
    "KarelIOMismatch": get_karel_io_model,
    "KarelDemo": get_karel_io_model,
    "KarelDemoMismatch": get_karel_io_model,
    "KarelIOFused": get_karel_io_fused_model,
    "KarelDemoFused": get_karel_io_fused_model,
}
