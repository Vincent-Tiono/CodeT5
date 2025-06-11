import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip import CLIPTextConfig
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig, CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.t5.configuration_t5 import T5Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class StringCLIPConfig(PretrainedConfig):
    model_type = "StringCLIP"
    is_composition = True

    def __init__(
        self, program_config=None, io_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        program_config_dict = kwargs.pop("program_config_dict", None)
        io_config_dict = kwargs.pop("io_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[program|io]_config_dict` to `[program|io]_config`, we use the values in
        # `[program|io]_config_dict` to update the values in `[program|io]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if program_config_dict is not None:
            if program_config is None:
                program_config = {}

            # This is the complete result when using `program_config_dict`.
            _program_config_dict = CLIPTextConfig(
                **program_config_dict).to_dict()

            # Give a warning if the values exist in both `_program_config_dict` and `program_config` but being different.
            for key, value in _program_config_dict.items():
                if key in program_config and value != program_config[key] and key not in ["transformers_version"]:
                    # If specified in `program_config_dict`
                    if key in program_config_dict:
                        message = (
                            f"`{key}` is found in both `program_config_dict` and `program_config` but with different values. "
                            f'The value `program_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`program_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The "
                            f'value `program_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `program_config` with the ones in `_program_config_dict`.
            program_config.update(_program_config_dict)

        if io_config_dict is not None:
            if io_config is None:
                io_config = {}

            # This is the complete result when using `io_config_dict`.
            _io_config_dict = CLIPTextConfig(**io_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _io_config_dict:
                _io_config_dict["id2label"] = {
                    str(key): value for key, value in _io_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_io_config_dict` and `io_config` but being different.
            for key, value in _io_config_dict.items():
                if key in io_config and value != io_config[key] and key not in ["transformers_version"]:
                    # If specified in `io_config_dict`
                    if key in io_config_dict:
                        message = (
                            f"`{key}` is found in both `io_config_dict` and `io_config` but with different "
                            f'values. The value `io_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`io_config_dict` is provided which will be used to initialize `CLIPTextConfig`. "
                            f'The value `io_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `io_config` with the ones in `_io_config_dict`.
            io_config.update(_io_config_dict)

        if program_config is None:
            program_config = {}
            logger.info(
                "`program_config` is `None`. Initializing the `CLIPTextConfig` with default values.")

        if io_config is None:
            io_config = {}
            logger.info(
                "`io_config` is `None`. initializing the `CLIPTextConfig` with default values.")

        self.program_config = CLIPTextConfig(**program_config)
        self.io_config = CLIPTextConfig(**io_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_program_io_configs(cls, program_config: CLIPTextConfig, io_config: CLIPTextConfig, **kwargs):
        r"""
        Instantiate a [`StringCLIPConfig`] (or a derived class) from clip text model configurations.

        Returns:
            [`StringCLIPConfig`]: An instance of a configuration object
        """

        return cls(program_config=program_config.to_dict(), io_config=io_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["program_config"] = self.program_config.to_dict()
        output["io_config"] = self.io_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class StringCLIPFusedConfig(PretrainedConfig):
    model_type = "StringCLIPFusedModel"
    is_composition = True

    def __init__(
        self, contrastive_config=None, t5_config=None, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        contrastive_config_dict = kwargs.pop("contrastive_config_dict", None)
        t5_config_dict = kwargs.pop("t5_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[contrastive|t5]_config_dict` to `[contrastive|t5]_config`, we use the values in
        # `[contrastive|t5]_config_dict` to update the values in `[contrastive|t5]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if contrastive_config_dict is not None:
            if contrastive_config is None:
                contrastive_config = {}

            # This is the complete result when using `contrastive_config_dict`.
            _contrastive_config_dict = StringCLIPConfig(
                **contrastive_config_dict).to_dict()

            # Give a warning if the values exist in both `_contrastive_config_dict` and `contrastive_config` but being different.
            for key, value in _contrastive_config_dict.items():
                if key in contrastive_config and value != contrastive_config[key] and key not in ["transformers_version"]:
                    # If specified in `contrastive_config_dict`
                    if key in contrastive_config_dict:
                        message = (
                            f"`{key}` is found in both `contrastive_config_dict` and `contrastive_config` but with different values. "
                            f'The value `contrastive_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`contrastive_config_dict` is provided which will be used to initialize `StringCLIPConfig`. The "
                            f'value `contrastive_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `contrastive_config` with the ones in `_contrastive_config_dict`.
            contrastive_config.update(_contrastive_config_dict)

        if t5_config_dict is not None:
            if t5_config is None:
                t5_config = {}

            # This is the complete result when using `t5_config_dict`.
            _t5_config_dict = T5Config(**t5_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _t5_config_dict:
                _t5_config_dict["id2label"] = {
                    str(key): value for key, value in _t5_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_t5_config_dict` and `t5_config` but being different.
            for key, value in _t5_config_dict.items():
                if key in t5_config and value != t5_config[key] and key not in ["transformers_version"]:
                    # If specified in `t5_config_dict`
                    if key in t5_config_dict:
                        message = (
                            f"`{key}` is found in both `t5_config_dict` and `t5_config` but with different "
                            f'values. The value `t5_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`t5_config_dict` is provided which will be used to initialize ` T5Config`. "
                            f'The value `t5_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `t5_config` with the ones in `_t5_config_dict`.
            t5_config.update(_t5_config_dict)

        if contrastive_config is None:
            contrastive_config = {}
            logger.info(
                "`contrastive_config` is `None`. Initializing the `StringCLIPConfig` with default values.")

        if t5_config is None:
            t5_config = {}
            logger.info(
                "`t5_config` is `None`. initializing the `T5Config` with default values.")

        self.contrastive_config = StringCLIPConfig(**contrastive_config)
        self.t5_config = T5Config(**t5_config)

    @classmethod
    def from_contrastive_t5_configs(cls, contrastive_config: StringCLIPConfig, t5_config: T5Config, **kwargs):
        r"""
        Instantiate a [`StringCLIPFusedConfig`] (or a derived class) from StringCLIPConfig config and T5Config

        Returns:
            [`StringCLIPFusedConfig`]: An instance of a configuration object
        """

        return cls(contrastive_config=contrastive_config.to_dict(), t5_config=t5_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["contrastive_config"] = self.contrastive_config.to_dict()
        output["t5_config"] = self.t5_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class Blip2NPSConfig(PretrainedConfig):

    model_type = "blip-2"
    is_composition = True

    def __init__(self, io_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        super().__init__(**kwargs)

        if io_config is None:
            io_config = {}
            logger.info(
                "io_config is None. initializing the ClipTextConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info(
                "qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info(
                "text_config is None. Initializing the text config with default values (`OPTConfig`).")

        self.io_config = CLIPTextConfig(**io_config)
        self.qformer_config = Blip2QFormerConfig(**qformer_config)
        text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.num_query_tokens = num_query_tokens
        self.qformer_config.encoder_hidden_size = self.io_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_io_qformer_text_configs(
        cls,
        io_config: CLIPTextConfig,
        qformer_config: Blip2QFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Blip2Config`] (or a derived class) from a BLIP-2 io model, Q-Former and language model
        configurations.

        Returns:
            [`Blip2Config`]: An instance of a configuration object
        """

        return cls(
            io_config=io_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["io_config"] = self.io_config.to_dict()
        output["qformer_config"] = self.qformer_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
