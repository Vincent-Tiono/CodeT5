import ipdb
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from src.configuration_string_trans import StringCLIPConfig, StringCLIPFusedConfig, Blip2NPSConfig
from transformers.modeling_outputs import (BaseModelOutputWithPooling,
                                           ModelOutput)
from transformers.models.clip import CLIPPreTrainedModel, CLIPTextConfig
from transformers.models.t5 import T5ForConditionalGeneration, T5Config
# from transformers.models.clip.modeling_clip import (CLIPTextTransformer,
#                                                     _expand_mask, clip_loss)
from transformers.models.clip.modeling_clip import (CLIPTextTransformer,
                                                    clip_loss)
from transformers.models.blip_2.modeling_blip_2 import Blip2PreTrainedModel, Blip2QFormerModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


from transformers.modeling_outputs import BaseModelOutput
from transformers.activations import get_activation
from transformers.utils import logging

logger = logging.get_logger(__name__)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2NPSForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        io_outputs (`BaseModelOutputWithPooling`):
            Outputs of the io encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    io_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["io_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class StringCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for io-text similarity.
        logits_per_program:(`torch.FloatTensor` of shape `(io_batch_size, text_batch_size)`):
            The scaled dot product scores between `io_embeds` and `text_embeds`. This represents the io-text
            similarity scores.
        logits_per_io:(`torch.FloatTensor` of shape `(text_batch_size, io_batch_size)`):
            The scaled dot product scores between `text_embeds` and `io_embeds`. This represents the text-io
            similarity scores.
        program_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        io_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The io embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        program_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        io_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_program: torch.FloatTensor = None
    logits_per_io: torch.FloatTensor = None
    program_embeds: torch.FloatTensor = None
    io_embeds: torch.FloatTensor = None
    program_model_output: BaseModelOutputWithPooling = None
    io_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["program_model_output",
                                 "io_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class StringMixCLIPOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    clip_loss: Optional[torch.FloatTensor] = None
    logits_per_program: torch.FloatTensor = None
    logits_per_io: torch.FloatTensor = None
    similar_loss: Optional[torch.FloatTensor] = None
    similar_logits_per_program: torch.FloatTensor = None
    similar_logits_per_io: torch.FloatTensor = None
    program_embeds: torch.FloatTensor = None
    io_embeds: torch.FloatTensor = None
    program_model_output: BaseModelOutputWithPooling = None
    io_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["program_model_output",
                                 "io_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLIPTextTransformerWithoutCausalMasking(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, IO does not need causal mask
        causal_attention_mask = None

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(
                last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int,
                         device=last_hidden_state.device).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class StringCLIPModel(CLIPPreTrainedModel):
    config_class = StringCLIPConfig

    def __init__(self, config: StringCLIPConfig):
        super().__init__(config)

        if not isinstance(config.program_config, CLIPTextConfig):
            raise ValueError(
                "config.program_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.program_config)}."
            )

        if not isinstance(config.io_config, CLIPTextConfig):
            raise ValueError(
                "config.io_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.io_config)}."
            )

        program_config = config.program_config
        io_config = config.io_config

        self.projection_dim = config.projection_dim
        self.program_embed_dim = program_config.hidden_size
        self.io_embed_dim = io_config.hidden_size

        self.program_model = CLIPTextTransformer(program_config)
        self.io_model = CLIPTextTransformerWithoutCausalMasking(io_config)

        self.program_projection = nn.Linear(
            self.program_embed_dim, self.projection_dim, bias=False)
        self.io_projection = nn.Linear(
            self.io_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones(
            []) * self.config.logit_scale_init_value)

        # Initialize weights and apply final processing
        self.post_init()

    def get_program_embeds(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        program_outputs = self.program_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = program_outputs[1]
        program_embeds = self.program_projection(pooled_output)
        program_embeds = program_embeds / \
            program_embeds.norm(p=2, dim=-1, keepdim=True)
        return program_embeds

    def get_program_states(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        program_outputs = self.program_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size, transformer.width]
        pooler_output = program_outputs.pooler_output
        return pooler_output

    def get_io_embeds(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        io_outputs = self.io_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = io_outputs[1]  # pooled_output
        io_embeds = self.io_projection(pooled_output)
        io_embeds = io_embeds / io_embeds.norm(p=2, dim=-1, keepdim=True)
        return io_embeds

    def get_io_states(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        io_outputs = self.io_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [batch_size, transformer.width]
        last_hidden_state = io_outputs.last_hidden_state
        return last_hidden_state

    def io2io(
        self,
        input_ids_0: Optional[torch.LongTensor] = None,
        input_ids_1: Optional[torch.FloatTensor] = None,
        attention_mask_0: Optional[torch.Tensor] = None,
        attention_mask_1: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        outputs_0 = self.io_model(
            input_ids=input_ids_0,
            attention_mask=attention_mask_0,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embeds_0 = outputs_0[1]
        embeds_0 = self.io_projection(embeds_0)
        outputs_1 = self.io_model(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embeds_1 = outputs_1[1]
        embeds_1 = self.io_projection(embeds_1)

        # normalized features
        embeds_0 = embeds_0 / embeds_0.norm(p=2, dim=-1, keepdim=True)
        embeds_1 = embeds_1 / embeds_1.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_embeds_1 = torch.matmul(
            embeds_1, embeds_0.t()) * logit_scale
        logits_per_embeds_0 = logits_per_embeds_1.t()
        return StringCLIPOutput(
            logits_per_io=logits_per_embeds_0,
            logits_per_program=logits_per_embeds_1,
        )

    def encode_batch_io(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bs, ns, sl = input_ids.shape
        encoder_input_ids = input_ids.view(bs * ns, sl)
        encoder_attention_mask = attention_mask.view(bs * ns, sl)
        # [batch_size * num_sample, transformer.width]
        io_features = self.get_io_embeds(
            encoder_input_ids, encoder_attention_mask)
        # ipdb.set_trace()
        inputs_embeds = torch.mean(
            io_features.view(bs, ns, -1), dim=1).contiguous()
        return inputs_embeds

    def batch_io_to_program(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        io_input_ids: Optional[torch.Tensor] = None,
        io_attention_mask: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.encode_batch_io(io_input_ids, io_attention_mask)
        program_embeds = self.get_program_embeds(input_ids, attention_mask)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_program = torch.matmul(
            program_embeds, inputs_embeds.t()) * logit_scale
        logits_per_io = logits_per_program.t()
        return StringCLIPOutput(
            logits_per_io=logits_per_io,
            logits_per_program=logits_per_program,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        io_input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        io_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, StringCLIPOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        io_outputs = self.io_model(
            input_ids=io_input_ids,
            attention_mask=io_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        program_outputs = self.program_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        io_embeds = io_outputs[1]
        io_embeds = self.io_projection(io_embeds)

        program_embeds = program_outputs[1]
        program_embeds = self.program_projection(program_embeds)

        # normalized features
        io_embeds = io_embeds / io_embeds.norm(p=2, dim=-1, keepdim=True)
        program_embeds = program_embeds / \
            program_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_program = torch.matmul(
            program_embeds, io_embeds.t()) * logit_scale
        logits_per_io = logits_per_program.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_program)

        if not return_dict:
            output = (logits_per_io, logits_per_program,
                      program_embeds, io_embeds, program_outputs, io_outputs)
            return ((loss,) + output) if loss is not None else output

        return StringCLIPOutput(
            loss=loss,
            logits_per_io=logits_per_io,
            logits_per_program=logits_per_program,
            program_embeds=program_embeds,
            io_embeds=io_embeds,
            program_model_output=program_outputs,
            io_model_output=io_outputs,
        )


class StringMixCLIPModel(StringCLIPModel):
    def __init__(self, config: StringCLIPConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        io_input_ids: Optional[torch.LongTensor] = None,
        io_attention_mask: Optional[torch.Tensor] = None,
        similar_io_input_ids: Optional[torch.LongTensor] = None,
        similar_io_attention_mask: Optional[torch.Tensor] = None,
        similar_program_input_ids: Optional[torch.LongTensor] = None,
        similar_program_attention_mask: Optional[torch.Tensor] = None,
        downsize_perm: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, StringCLIPOutput]:

        # original CLIP forward
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        io_outputs = self.io_model(
            input_ids=io_input_ids,
            attention_mask=io_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        program_outputs = self.program_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        io_embeds = io_outputs[1]
        io_embeds = self.io_projection(io_embeds)

        program_embeds = program_outputs[1]
        program_embeds = self.program_projection(program_embeds)

        # normalized features
        io_embeds = io_embeds / io_embeds.norm(p=2, dim=-1, keepdim=True)
        program_embeds = program_embeds / \
            program_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_program = torch.matmul(
            program_embeds, io_embeds.t()) * logit_scale
        logits_per_io = logits_per_program.t()

        # compute similar program and io embeddings
        # ipdb.set_trace()
        bs, n_sample, io_length = similar_io_input_ids.shape
        bs, n_sample, program_length = similar_program_input_ids.shape
        similar_io_input_ids = similar_io_input_ids.view(
            bs * n_sample, io_length)
        similar_io_attention_mask = similar_io_attention_mask.view(
            bs * n_sample, io_length)
        similar_program_input_ids = similar_program_input_ids.view(
            bs * n_sample, program_length)
        similar_program_attention_mask = similar_program_attention_mask.view(
            bs * n_sample, program_length)

        similar_io_outputs = self.io_model(
            input_ids=similar_io_input_ids,
            attention_mask=similar_io_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        similar_io_embeds = similar_io_outputs[1]
        similar_io_embeds = self.io_projection(similar_io_embeds)
        similar_io_embeds = similar_io_embeds / \
            similar_io_embeds.norm(p=2, dim=-1, keepdim=True)
        similar_io_embeds = similar_io_embeds.view(bs, n_sample, -1)

        similar_program_outputs = self.program_model(
            input_ids=similar_program_input_ids,
            attention_mask=similar_program_attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        similar_program_embeds = similar_program_outputs[1]
        similar_program_embeds = self.program_projection(
            similar_program_embeds)
        similar_program_embeds = similar_program_embeds / \
            similar_program_embeds.norm(p=2, dim=-1, keepdim=True)
        similar_program_embeds = similar_program_embeds.view(bs, n_sample, -1)

        # compute similar loss
        loss_fct = nn.CrossEntropyLoss()
        similar_bs = similar_program_embeds.shape[0]
        # ipdb.set_trace()

        similar_program_logits = torch.bmm(io_embeds[downsize_perm].unsqueeze(
            1), similar_program_embeds.transpose(1, 2)).squeeze() * self.logit_scale.exp()
        label = torch.zeros(similar_bs, dtype=torch.long).to(
            similar_program_logits.device)
        similar_program_loss = loss_fct(similar_program_logits, label)

        similar_io_logits = torch.bmm(program_embeds[downsize_perm].unsqueeze(
            1), similar_io_embeds.transpose(1, 2)).squeeze() * self.logit_scale.exp()
        similar_io_loss = loss_fct(similar_io_logits, label)
        similar_loss = (similar_io_loss + similar_program_loss) / 2

        # compute total loss
        clip_loss_ = clip_loss(logits_per_program)
        total_loss = (clip_loss_ + similar_loss) / 2

        if not return_dict:
            output = (logits_per_program,) + program_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return StringMixCLIPOutput(
            loss=total_loss,
            clip_loss=clip_loss_,
            logits_per_io=logits_per_io,
            logits_per_program=logits_per_program,
            similar_loss=similar_loss,
            similar_logits_per_io=similar_program_logits,
            similar_logits_per_program=similar_io_logits,
        )


class StringDissimilarCLIPModel(StringMixCLIPModel):
    def __init__(self, config: StringCLIPConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        io_input_ids: Optional[torch.LongTensor] = None,
        io_attention_mask: Optional[torch.Tensor] = None,
        similar_io_input_ids: Optional[torch.LongTensor] = None,
        similar_io_attention_mask: Optional[torch.Tensor] = None,
        similar_program_input_ids: Optional[torch.LongTensor] = None,
        similar_program_attention_mask: Optional[torch.Tensor] = None,
        downsize_perm: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, StringCLIPOutput]:

        # original CLIP forward
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        io_outputs = self.io_model(
            input_ids=io_input_ids,
            attention_mask=io_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        program_outputs = self.program_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        io_embeds = io_outputs[1]
        io_embeds = self.io_projection(io_embeds)

        program_embeds = program_outputs[1]
        program_embeds = self.program_projection(program_embeds)

        # normalized features
        io_embeds = io_embeds / io_embeds.norm(p=2, dim=-1, keepdim=True)
        program_embeds = program_embeds / \
            program_embeds.norm(p=2, dim=-1, keepdim=True)

        # compute similar program and io embeddings
        # ipdb.set_trace()
        bs, n_sample, io_length = similar_io_input_ids.shape
        bs, n_sample, program_length = similar_program_input_ids.shape
        similar_io_input_ids = similar_io_input_ids.view(
            bs * n_sample, io_length)
        similar_io_attention_mask = similar_io_attention_mask.view(
            bs * n_sample, io_length)
        similar_program_input_ids = similar_program_input_ids.view(
            bs * n_sample, program_length)
        similar_program_attention_mask = similar_program_attention_mask.view(
            bs * n_sample, program_length)

        similar_io_outputs = self.io_model(
            input_ids=similar_io_input_ids,
            attention_mask=similar_io_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        similar_io_embeds = similar_io_outputs[1]
        similar_io_embeds = self.io_projection(similar_io_embeds)
        similar_io_embeds = similar_io_embeds / \
            similar_io_embeds.norm(p=2, dim=-1, keepdim=True)
        similar_io_embeds = similar_io_embeds.view(bs, n_sample, -1)

        similar_program_outputs = self.program_model(
            input_ids=similar_program_input_ids,
            attention_mask=similar_program_attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        similar_program_embeds = similar_program_outputs[1]
        similar_program_embeds = self.program_projection(
            similar_program_embeds)
        similar_program_embeds = similar_program_embeds / \
            similar_program_embeds.norm(p=2, dim=-1, keepdim=True)
        similar_program_embeds = similar_program_embeds.view(bs, n_sample, -1)

        # compute similar loss
        loss_fct = nn.CrossEntropyLoss()
        similar_bs = similar_program_embeds.shape[0]

        # cosine similarity as logits
        similar_program_logits = torch.bmm(io_embeds[downsize_perm].unsqueeze(
            1), similar_program_embeds.transpose(1, 2)).squeeze() * self.logit_scale.exp()
        label = torch.zeros(similar_bs, dtype=torch.long).to(
            similar_program_logits.device)
        similar_program_loss = loss_fct(similar_program_logits, label)

        similar_io_logits = torch.bmm(program_embeds[downsize_perm].unsqueeze(
            1), similar_io_embeds.transpose(1, 2)).squeeze() * self.logit_scale.exp()
        similar_io_loss = loss_fct(similar_io_logits, label)
        similar_loss = (similar_io_loss + similar_program_loss) / 2

        return StringCLIPOutput(
            loss=similar_loss,
            logits_per_io=similar_program_logits,
            logits_per_program=similar_io_logits,
        )


class StringCLIPFusedModel(CLIPPreTrainedModel):
    def __init__(self, config: StringCLIPFusedConfig):
        super().__init__(config)
        self.encoder = StringCLIPModel(config.contrastive_config)
        self.decoder = T5ForConditionalGeneration(config.t5_config)
        self.encoder_projection = nn.Linear(
            config.contrastive_config.projection_dim, config.t5_config.d_model, bias=False)
        self.post_init()

    def gradient_checkpointing_enable(self):
        return super().gradient_checkpointing_enable() and self.encoder.gradient_checkpointing_enable() and self.decoder.gradient_checkpointing_enable()

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.encoder_projection.apply(self.decoder._init_weights)

    def _encode_io(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bs, ns, sl = input_ids.shape
        encoder_input_ids = input_ids.view(bs * ns, sl)
        encoder_attention_mask = attention_mask.view(bs * ns, sl)
        # [batch_size * num_sample, transformer.width]
        io_hidden_states = self.encoder.get_io_embeds(
            encoder_input_ids, encoder_attention_mask)
        io_hidden_states: torch.Tensor = self.encoder_projection(io_hidden_states)
        decoder_inputs_embeds = io_hidden_states.view(bs, ns, -1).contiguous()
        return decoder_inputs_embeds, torch.ones(bs, ns, device=decoder_inputs_embeds.device, dtype=torch.long)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
            input_ids: (torch.LongTensor, in shape [batch_size, num_sample, sequence_length])
            attention_mask: (torch.LongTensor, in shape [batch_size, num_sample, sequence_length])
            labels: (torch.LongTensor, in shape [batch_size, sequence_length])
        """
        decoder_inputs_embeds, decoder_attention_mask = self._encode_io(
            input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kargs,
    ):
        decoder_inputs_embeds, decoder_attention_mask = self._encode_io(
            input_ids=input_ids, attention_mask=attention_mask)
        return self.decoder.generate(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            **kargs,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.decoder.prepare_decoder_input_ids_from_labels(labels)

    @classmethod
    def from_contrastive_t5_paths(cls, contrastive_path, t5_path):
        contrastive_config = StringCLIPConfig.from_pretrained(contrastive_path)
        t5_config = T5Config.from_pretrained(t5_path)
        contrastive_t5_config = StringCLIPFusedConfig.from_contrastive_t5_configs(
            contrastive_config, t5_config)
        model = cls(contrastive_t5_config)
        model.encoder = StringCLIPModel.from_pretrained(contrastive_path)
        model.decoder = T5ForConditionalGeneration.from_pretrained(t5_path)
        return model


class StringCLIPFusedFullSequenceModel(StringCLIPFusedModel):
    def __init__(self, config: StringCLIPFusedConfig):
        super().__init__(config)
        self.encoder_projection = nn.Linear(
            config.contrastive_config.io_config.hidden_size, config.t5_config.d_model, bias=False)

    def _encode_io(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bs, ns, sl = input_ids.shape
        encoder_input_ids = input_ids.view(bs * ns, sl)
        encoder_attention_mask = attention_mask.view(bs * ns, sl)
        # [batch_size * num_sample, sequence_length, transformer.width]
        io_hidden_states = self.encoder.get_io_states(
            encoder_input_ids, encoder_attention_mask)
        io_hidden_states = io_hidden_states.reshape(bs, ns, sl, -1)
        io_hidden_states = io_hidden_states.reshape(bs, ns * sl, -1)
        io_hidden_states: torch.Tensor = self.encoder_projection(io_hidden_states)
        io_attention_mask = attention_mask.reshape(bs, ns * sl)
        return io_hidden_states, io_attention_mask


class StringCLIPFusedT5DecoderModel(StringCLIPFusedFullSequenceModel):
    def __init__(self, config: StringCLIPFusedConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
            input_ids: (torch.LongTensor, in shape [batch_size, num_sample, sequence_length])
            attention_mask: (torch.LongTensor, in shape [batch_size, num_sample, sequence_length])
            labels: (torch.LongTensor, in shape [batch_size, sequence_length])
        """
        inputs_embeds, attention_mask = self._encode_io(
            input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=inputs_embeds,
            attentions=attention_mask,
        )
        outputs = self.decoder(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kargs,
    ):
        inputs_embeds, attention_mask = self._encode_io(
            input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=inputs_embeds,
            attentions=attention_mask,
        )
        return self.decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **kargs,
        )


class StringCLIPMappingT5DecoderModel(StringCLIPFusedT5DecoderModel):
    def __init__(self, config: StringCLIPFusedConfig):
        super().__init__(config)
        base_config: CLIPTextConfig = config.contrastive_config.io_config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_config.hidden_size,
            nhead=base_config.num_attention_heads,
            dim_feedforward=base_config.intermediate_size,
            dropout=base_config.attention_dropout,
            activation=get_activation(base_config.hidden_act),
            batch_first=True,
        )
        self.encoder_mapping = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=base_config.num_hidden_layers,
        )

    def _encode_io(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bs, ns, sl = input_ids.shape
        encoder_input_ids = input_ids.view(bs * ns, sl)
        encoder_attention_mask = attention_mask.view(bs * ns, sl)
        # [batch_size * num_sample, sequence_length, transformer.width]
        io_hidden_states = self.encoder.get_io_states(
            encoder_input_ids, encoder_attention_mask)
        io_hidden_states = io_hidden_states.view(bs, ns, sl, -1)
        io_hidden_states = io_hidden_states.view(bs, ns * sl, -1)
        io_attention_mask = attention_mask.view(bs, ns * sl).contiguous()
        io_hidden_states = self.encoder_mapping(
            io_hidden_states, src_key_padding_mask=io_attention_mask.bool())
        io_hidden_states: torch.Tensor = self.encoder_projection(io_hidden_states)
        return io_hidden_states, io_attention_mask


class ContrastiveAugmentedNPSModel(StringCLIPFusedModel):
    def __init__(self, config: StringCLIPFusedConfig):
        super().__init__(config)

    def forward(
        self,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
            input_ids: (torch.LongTensor, in shape [batch_size, num_sample, sequence_length])
            attention_mask: (torch.LongTensor, in shape [batch_size, num_sample, sequence_length])
            t5_input_ids: (torch.LongTensor, in shape [batch_size, sequence_length])
            t5_attention_mask: (torch.LongTensor, in shape [batch_size, sequence_length])
            labels: (torch.LongTensor, in shape [batch_size, sequence_length])
        """
        decoder_inputs_embeds, decoder_attention_mask = self._encode_io(
            input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.decoder.get_input_embeddings()
        t5_inputs_embeds = embeddings(t5_input_ids)
        decoder_inputs_embeds = torch.cat(
            [t5_inputs_embeds, decoder_inputs_embeds], dim=1)
        decoder_attention_mask = torch.cat(
            [t5_attention_mask, decoder_attention_mask], dim=1)
        outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kargs,
    ):
        decoder_inputs_embeds, decoder_attention_mask = self._encode_io(
            input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.decoder.get_input_embeddings()
        t5_inputs_embeds = embeddings(t5_input_ids)
        decoder_inputs_embeds = torch.cat(
            [t5_inputs_embeds, decoder_inputs_embeds], dim=1)
        decoder_attention_mask = torch.cat(
            [t5_attention_mask, decoder_attention_mask], dim=1)
        return self.decoder.generate(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            **kargs,
        )


class ContrastiveAugmentedFixCLIPNPSModel(ContrastiveAugmentedNPSModel):
    def __init__(self, config: StringCLIPFusedConfig):
        super().__init__(config)

    def _encode_io(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            return super()._encode_io(input_ids, attention_mask)

# modified from transformers.models.blip_2.modeling_blip_2.Blip2Model -> Blip2NPSModel


class Blip2NPSModel(Blip2PreTrainedModel):
    config_class = Blip2NPSConfig
    main_input_name = "io_input_ids"

    def __init__(self, config: Blip2NPSConfig):
        super().__init__(config)

        self.io_model = CLIPTextTransformerWithoutCausalMasking(
            config.io_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(
                config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(
                config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            text_outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        return text_outputs

    def get_io_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        io_outputs = self.io_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return io_outputs

    def get_qformer_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        io_outputs = self.io_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        io_embeds = io_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(
            io_embeds.size()[:-1], dtype=torch.long, device=io_embeds.device)

        query_tokens = self.query_tokens.expand(io_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=io_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return query_outputs

    def forward(
        self,
        io_input_ids: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        io_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the IO pair through the io encoder,
        # to get io embeddings of shape (batch_size, seq_len, hidden_size)
        io_outputs = self.io_model(
            input_ids=io_input_ids,
            attention_mask=io_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        io_embeds = io_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        io_attention_mask = torch.ones(
            io_embeds.size()[:-1], dtype=torch.long, device=io_embeds.device)

        query_tokens = self.query_tokens.expand(io_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=io_embeds,
            encoder_attention_mask=io_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size(
            )[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat(
            [language_model_inputs, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat(
            [language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1):, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(reduction="mean")

                loss = loss_fct(
                    shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, io_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            io_outputs=io_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

# modified from transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGeneration -> Blip2NPSForConditionalGeneration


class Blip2NPSForConditionalGeneration(Blip2PreTrainedModel):
    config_class = Blip2NPSConfig
    main_input_name = "io_input_ids"

    def __init__(self, config: Blip2NPSConfig):
        super().__init__(config)

        self.io_model = CLIPTextTransformerWithoutCausalMasking(
            config.io_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(
                config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(
                config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward(
        self,
        input_ids: torch.LongTensor,
        prompt_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]

        batch_size, num_sample, sequence_length = input_ids.shape
        encoder_input_ids = input_ids.view(
            batch_size * num_sample, sequence_length)
        encoder_attention_mask = attention_mask.view(
            batch_size * num_sample, sequence_length)

        # step 1: forward the IO pair through the io encoder,
        # to get IO pair embeddings of shape (batch_size, seq_len, hidden_size)
        io_outputs = self.io_model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        io_embeds = io_outputs[0]
        io_embeds = io_embeds.reshape(
            batch_size, num_sample, sequence_length, -1)
        io_embeds = io_embeds.reshape(
            batch_size, num_sample * sequence_length, -1)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        attention_mask = torch.ones(
            io_embeds.size()[:-1], dtype=torch.long, device=io_embeds.device)

        query_tokens = self.query_tokens.expand(io_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=io_embeds,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size(
            )[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if prompt_input_ids is None:
            prompt_input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(io_embeds.device)
            )

        inputs_embeds = self.language_model.get_input_embeddings()(prompt_input_ids)
        inputs_embeds = torch.cat(
            [language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids)
        expected_device = language_model_attention_mask.device
        prompt_attention_mask = torch.cat(
            [language_model_attention_mask, prompt_attention_mask.to(expected_device)], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prompt_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1):, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(reduction="mean")

                loss = loss_fct(
                    shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prompt_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, io_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            io_outputs=io_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.language_model.prepare_decoder_input_ids_from_labels(labels)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        prompt_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            io_input_ids (`torch.LongTensor` of shape (batch_size, num_sample, sequence_length)):
                The IO pair use to generate program
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            io_attention_mask (`torch.LongTensor` of shape (batch_size, num_sample, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size, num_sample, sequence_length = input_ids.shape
        encoder_input_ids = input_ids.view(
            batch_size * num_sample, sequence_length)
        encoder_attention_mask = attention_mask.view(
            batch_size * num_sample, sequence_length)
        io_embeds = self.io_model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state
        io_embeds = io_embeds.reshape(
            batch_size, num_sample, sequence_length, -1)
        io_embeds = io_embeds.reshape(
            batch_size, num_sample * sequence_length, -1)
        io_attention_mask = torch.ones(
            io_embeds.size()[:-1], dtype=torch.long, device=io_embeds.device)

        query_tokens = self.query_tokens.expand(io_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=io_embeds,
            encoder_attention_mask=io_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size(
            )[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if prompt_input_ids is None:
            prompt_input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(io_embeds.device)
            )
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids)

        prompt_attention_mask = torch.cat([language_attention_mask, prompt_attention_mask.to(
            language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(prompt_input_ids)
        inputs_embeds = torch.cat(
            [language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prompt_attention_mask,
            **generate_kwargs,
        )

        return outputs
