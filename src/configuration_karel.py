import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip import CLIPTextConfig
from transformers.models.t5 import T5Config
from transformers.utils import logging


logger = logging.get_logger(__name__)

class KarelFusedConfig(PretrainedConfig):
    model_type = "KarelFusedModel"
    is_composition = True

    def __init__(
        self, cliptext_config=None, t5_config=None, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        cliptext_config_dict = kwargs.pop("cliptext_config_dict", None)
        t5_config_dict = kwargs.pop("t5_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[cliptext|t5]_config_dict` to `[cliptext|t5]_config`, we use the values in
        # `[cliptext|t5]_config_dict` to update the values in `[cliptext|t5]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if cliptext_config_dict is not None:
            if cliptext_config is None:
                cliptext_config = {}

            # This is the complete result when using `cliptext_config_dict`.
            _cliptext_config_dict = CLIPTextConfig(
                **cliptext_config_dict).to_dict()

            # Give a warning if the values exist in both `_cliptext_config_dict` and `cliptext_config` but being different.
            for key, value in _cliptext_config_dict.items():
                if key in cliptext_config and value != cliptext_config[key] and key not in ["transformers_version"]:
                    # If specified in `cliptext_config_dict`
                    if key in cliptext_config_dict:
                        message = (
                            f"`{key}` is found in both `cliptext_config_dict` and `cliptext_config` but with different values. "
                            f'The value `cliptext_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`cliptext_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The "
                            f'value `cliptext_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `cliptext_config` with the ones in `_cliptext_config_dict`.
            cliptext_config.update(_cliptext_config_dict)

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

        if cliptext_config is None:
            cliptext_config = {}
            logger.info(
                "`cliptext_config` is `None`. Initializing the `CLIPTextConfig` with default values.")

        if t5_config is None:
            t5_config = {}
            logger.info(
                "`t5_config` is `None`. initializing the `T5Config` with default values.")

        self.cliptext_config = CLIPTextConfig(**cliptext_config)
        self.t5_config = T5Config(**t5_config)


    @classmethod
    def from_cliptext_t5_configs(cls, cliptext_config: CLIPTextConfig, t5_config: T5Config, **kwargs):
        r"""
        Instantiate a [`KarelFusedConfig`] (or a derived class) from CLIPTextConfig config and T5Config

        Returns:
            [`KarelFusedConfig`]: An instance of a configuration object
        """

        return cls(cliptext_config=cliptext_config.to_dict(), t5_config=t5_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["cliptext_config"] = self.cliptext_config.to_dict()
        output["t5_config"] = self.t5_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output

