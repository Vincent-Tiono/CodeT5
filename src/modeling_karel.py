from typing import Optional, Union, Tuple

import ipdb
import torch
import torch.nn as nn
from transformers import PreTrainedModel, T5Config, T5ForConditionalGeneration
from torchvision.models.resnet import Bottleneck, BasicBlock
from transformers.models.clip import CLIPPreTrainedModel, CLIPTextConfig
# from transformers.models.clip.modeling_clip import CLIPTextTransformer, clip_loss, CLIPEncoder, _expand_mask
from transformers.models.clip.modeling_clip import CLIPTextTransformer, clip_loss, CLIPEncoder
from src.modeling_string_trans import StringCLIPOutput
from src.configuration_karel import KarelFusedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class ResNetEncoder(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        base_size = hidden_size // 4
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size, base_size,
                      kernel_size=3, stride=1, padding=1),
            BasicBlock(base_size, base_size),
            BasicBlock(base_size, base_size),
            # nn.Conv2d(base_size, base_size, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_size, base_size * 2,
                      kernel_size=3, stride=1, padding=1),
            BasicBlock(base_size * 2, base_size * 2),
            BasicBlock(base_size * 2, base_size * 2),
            # nn.Conv2d(base_size * 2, base_size * 2, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_size * 2, base_size * 4,
                      kernel_size=2, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        # input_shape (batch_size, c, h, w)
        # type casting, to avoid error when using accelrate launch
        x = x.to(self.cnn[0].weight.dtype)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return x

    def _init_weights(self, module: nn.Module):
        # intialize function from https://github.com/pytorch/vision/blob/90913fb47629de01e6369bd841fdec6c82604b48/torchvision/models/resnet.py#L208
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class KarelIOCLIPModel(CLIPPreTrainedModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.logit_scale_init_value = 2.6592
        self.state_model = ResNetEncoder(
            config.num_channels, config.hidden_size)
        self.program_model = CLIPTextTransformer(config)
        self.projection_dim = config.projection_dim
        self.program_projection = nn.Linear(
            config.hidden_size, self.projection_dim, bias=False)
        self.state_projection = nn.Linear(
            config.hidden_size, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * self.logit_scale_init_value)
        self.post_init()

    def init_weights(self):
        self.state_model.apply(self.state_model._init_weights)
        self.program_model.apply(self._init_weights)

    def get_io_states(self, input_output_states: torch.Tensor) -> torch.Tensor:
        # input_output_state (batch_size, channel, height, width)
        io_states = self.state_model(input_output_states)
        return io_states

    def forward(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, StringCLIPOutput]:
        program_outputs = self.program_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        io_states = self.state_model(input_output_states)
        io_embeds: torch.Tensor = self.state_projection(io_states)

        program_embeds: torch.Tensor = program_outputs[1]
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

        loss = clip_loss(logits_per_program)
        return StringCLIPOutput(
            loss=loss,
            logits_per_io=logits_per_io,
            logits_per_program=logits_per_program,
            program_embeds=program_embeds,
            io_embeds=io_embeds,
            program_model_output=program_outputs,
        )


class KarelIOFusedModel(CLIPPreTrainedModel):
    def __init__(self, config: KarelFusedConfig):
        super().__init__(config)
        self.encoder = KarelIOCLIPModel(config.cliptext_config)
        self.decoder = T5ForConditionalGeneration(config.t5_config)
        self.encoder_projection = nn.Linear(
            config.cliptext_config.hidden_size, config.t5_config.d_model, bias=False)
        self.post_init()

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.encoder_projection.apply(self.decoder._init_weights)

    def _encode_io(self, input_output_states: torch.Tensor) -> torch.Tensor:
        # input_output_state (batch_size, num_demos, channel, height, width)
        bs, num_demo, c, h, w = input_output_states.shape
        input_output_states = input_output_states.view(bs * num_demo, c, h, w)
        io_states = self.encoder.get_io_states(input_output_states)
        io_states = io_states.view(bs, num_demo, -1)
        return io_states

    def forward(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.encoder_projection(
            self._encode_io(input_output_states))
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        **kargs,
    ):
        inputs_embeds = self.encoder_projection(
            self._encode_io(input_output_states))
        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            **kargs,
        )

    def prepare_inputs_for_generation(self, labels: torch.Tensor):
        return self.decoder.prepare_inputs_for_generation(labels)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.decoder.prepare_decoder_input_ids_from_labels(labels)


class CNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        base_size = hidden_size // 2
        self.cnn1 = nn.Conv2d(input_size, base_size,
                              kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(base_size)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv2d(base_size, base_size * 2,
                              kernel_size=4, stride=2, padding=0)

    def forward(self, x):
        # input_shape (batch_size, c, h, w)
        # type casting, there is a bug using autocasting with accelerate launch
        x = x.to(self.cnn1.weight.dtype)

        # ipdb.set_trace()
        x = self.relu1(self.bn1(self.cnn1(x)))
        x: torch.Tensor = self.cnn2(x)
        x = x.view(x.size(0), -1)
        return x


class KarelIOSeq2SeqModel(PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        if config.encoder_type == "cnn":
            self.cnn_encoder = CNNEncoder(config.num_channels, config.d_model)
        elif config.encoder_type == "resnet":
            self.cnn_encoder = ResNetEncoder(
                config.num_channels, config.d_model)
        self.encoder_projection = nn.Linear(
            config.d_model, config.d_model, bias=False)
        self.decoder = T5ForConditionalGeneration(config)

    def gradient_checkpointing_enable(self):
        self.decoder.gradient_checkpointing_enable()

    def init_weights(self):
        self.decoder.init_weights()

    def encode_states(self, states):
        bs, num_states, c, h, w = states.shape
        states = states.view(bs * num_states, c, h, w)
        input_output_embedding = self.encoder_projection(
            self.cnn_encoder(states))
        input_output_embedding = input_output_embedding.view(
            bs, num_states, -1)
        return input_output_embedding

    def forward(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # input_output_state (batch_size, num_demo * 2, channel, height, width)
        # decoder_input_ids (batch_size, max_seq_len)
        # labels (batch_size, max_seq_len)
        # ipdb.set_trace()
        input_output_embedding = self.encode_states(input_output_states)
        outputs = self.decoder(
            inputs_embeds=input_output_embedding,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.decoder.prepare_inputs_for_generation(input_ids)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.decoder.prepare_decoder_input_ids_from_labels(labels)

    @torch.no_grad()
    def generate(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        **kargs,
    ):
        input_output_embedding = self.encode_states(input_output_states)
        return self.decoder.generate(
            inputs_embeds=input_output_embedding,
            **kargs,
        )


class KarelDemoSeq2seqModel(KarelIOSeq2SeqModel):
    def __init__(self, config: T5Config):
        super().__init__(config)

    def forward(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        input_output_embedding = self.encode_states(input_output_states)
        outputs = self.decoder(
            inputs_embeds=input_output_embedding,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kargs,
    ):
        input_output_embedding = self.encode_states(input_output_states)
        return self.decoder.generate(
            inputs_embeds=input_output_embedding,
            attention_mask=attention_mask,
            **kargs,
        )

# Merge CLIPTextEmbedding and CLIPVisionEmbedding to KarelStateEmbedding


class KarelStateEmbedding(CLIPPreTrainedModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.state_embedding = ResNetEncoder(
            config.num_channels, self.embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, c, h, w = states.shape
        states = states.view(batch_size * seq_length, c, h, w)

        if position_ids is None:
            # seq_length + 1 because the first token is the CLS embedding
            position_ids = self.position_ids[:, :seq_length+1]

        inputs_embeds = self.state_embedding(states)
        position_embeddings = self.position_embedding(position_ids)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
        embeddings = torch.cat(
            [class_embeds, inputs_embeds], dim=1) + position_embeddings
        return embeddings


# Converts transformers.model.clip.modeling_clip.CLIPVisionTransformer to KarelStateTransformer
class KarelStateTransformer(CLIPPreTrainedModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        embed_dim = config.hidden_size
        self.embeddings = KarelStateEmbedding(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        hidden_states = self.embeddings(states)
        hidden_states = self.pre_layrnorm(hidden_states)


        if attention_mask is not None:
            # We create a 4D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, seq_length + 1, seq_length + 1]
            batch_size, seq_length = attention_mask.shape
            new_attention_mask = torch.zeros(batch_size, seq_length + 1, dtype=attention_mask.dtype, device=attention_mask.device)
            new_attention_mask[:, 0] = 1
            new_attention_mask[:, 1:] = attention_mask
            attention_mask = _expand_mask(new_attention_mask, hidden_states.dtype)


        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class KarelPretrainedModel(CLIPPreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, KarelStateEmbedding):
            nn.init.normal_(module.class_embedding, mean=0.0,
                            std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.position_embedding.weight,
                            std=self.config.initializer_range * factor)
        elif isinstance(module, ResNetEncoder):
            module.apply(module._init_weights)


class KarelDemoCLIPModel(CLIPPreTrainedModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.logit_scale_init_value = 2.6592
        self.state_model = KarelStateTransformer(config)
        self.program_model = CLIPTextTransformer(config)
        self.projection_dim = config.projection_dim
        self.program_projection = nn.Linear(
            config.hidden_size, self.projection_dim, bias=False)
        self.state_projection = nn.Linear(
            config.hidden_size, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * self.logit_scale_init_value)
        self.post_init()

    def _get_states_states(
        self,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        return self.state_model(states=states, attention_mask=attention_mask)[0]

    def forward(
        self,
        input_states: Optional[torch.Tensor] = None,
        states_padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input_state for Karel Demonstation sequence
        # input_ids for Karel program, attention_mask for masking padding tokens
        program_outputs = self.program_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # ipdb.set_trace()
        io_states = self.state_model(states=input_states, attention_mask=states_padding_mask)[1]
        io_embeds: torch.Tensor = self.state_projection(io_states)

        program_embeds: torch.Tensor = program_outputs[1]
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

        loss = clip_loss(logits_per_program)
        return StringCLIPOutput(
            loss=loss,
            logits_per_io=logits_per_io,
            logits_per_program=logits_per_program,
            program_embeds=program_embeds,
            io_embeds=io_embeds,
            program_model_output=program_outputs,
        )

class KarelDemoFusedModel(CLIPPreTrainedModel):
    def __init__(self, config: KarelFusedConfig):
        super().__init__(config)
        self.encoder = KarelDemoCLIPModel(config.cliptext_config)
        self.decoder = T5ForConditionalGeneration(config.t5_config)
        self.encoder_projection =  nn.Linear(
            config.cliptext_config.projection_dim, config.t5_config.d_model, bias=False)
        self.post_init()

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.encoder_projection.apply(self.encoder._init_weights)


    def _encode_io(
        self,
        input_states: torch.Tensor,
        states_padding_mask: torch.Tensor,
    ):
        bs, num_demo, seq_length, c, h, w = input_states.shape
        input_states = input_states.view(bs*num_demo, seq_length, c, h, w)
        states_padding_mask = states_padding_mask.view(bs*num_demo, seq_length)
        io_states = self.encoder._get_states_states(states=input_states, attention_mask=states_padding_mask)
        # ipdb.set_trace()
        # sequence length with new class token
        seq_length = seq_length + 1
        io_states = io_states.view(bs, num_demo, seq_length, -1)
        io_states = io_states.view(bs, num_demo * seq_length, -1)
        # ipdb.set_trace()
        attention_mask = torch.ones(bs * num_demo, seq_length, device=states_padding_mask.device, dtype=states_padding_mask.dtype)
        attention_mask[:, :-1] = states_padding_mask
        attention_mask = attention_mask.view(bs, num_demo * seq_length)
        return io_states, attention_mask

    def forward(
        self,
        input_states: Optional[torch.Tensor] = None,
        states_padding_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds, attention_mask = self._encode_io(input_states, states_padding_mask)
        inputs_embeds = self.encoder_projection(inputs_embeds)
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_states: Optional[torch.Tensor] = None,
        states_padding_mask: Optional[torch.Tensor] = None,
        **kargs,
    ):
        inputs_embeds, attention_mask = self._encode_io(input_states, states_padding_mask)
        inputs_embeds = self.encoder_projection(inputs_embeds)
        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kargs,
        )

    def prepare_inputs_for_generation(self, labels: torch.Tensor):
        return self.decoder.prepare_inputs_for_generation(labels)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.decoder.prepare_decoder_input_ids_from_labels(labels)
