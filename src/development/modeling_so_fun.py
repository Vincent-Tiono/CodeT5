from typing import Optional, Union, Tuple

import ipdb
import torch
import torch.nn as nn
from transformers import PreTrainedModel, T5Config, T5ForConditionalGeneration
from torchvision.models.resnet import Bottleneck, BasicBlock
from transformers.models.clip import CLIPPreTrainedModel, CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPTextTransformer, clip_loss, CLIPEncoder, _expand_mask
from src.modeling_string_trans import StringCLIPOutput
from src.configuration_karel import KarelFusedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from src.modeling_karel import KarelDemoSeq2seqModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torchmetrics import Accuracy

@dataclass
class SoFunOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    action_loss: torch.FloatTensor = None
    perception_loss: torch.FloatTensor = None
    action_acc: torch.FloatTensor = None
    perception_acc: torch.FloatTensor = None

class KarelSoFun(KarelDemoSeq2seqModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.action_head = nn.Linear(config.d_model, 5)
        self.perception_head = nn.Linear(config.d_model, 5)
        self.action_accuracy = Accuracy(task="multiclass", num_classes=5, ignore_index=-100)


    def forward_so_fun(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actions=None,
        perceptions=None,
        perception_mask=None,
        **kwargs,
    ):
        import ipdb; ipdb.set_trace()
        # get the last hidden state from the encoder
        input_output_embedding = self.encode_states(input_output_states)
        t5_encoder = self.decoder.get_encoder()
        encoder_outputs = t5_encoder(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=input_output_embedding,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        action_logits = self.action_head(last_hidden_state).permute(0, 2, 1)
        perception_logits = self.perception_head(last_hidden_state)
        action_loss = None
        perception_loss = None
        if actions is not None:
            action_loss = nn.CrossEntropyLoss()(action_logits, actions)
            action_acc = self.action_accuracy(action_logits, actions)
        if perceptions is not None:
            perception_loss = nn.BCEWithLogitsLoss(reduction="none")(perception_logits, perceptions)
            perception_loss = perception_loss * perception_mask.unsqueeze(-1)
            perception_loss = perception_loss.sum() / perception_mask.sum()
            # get the perception accuracy with perception_mask
            perception_prediction = perception_logits.detach().sigmoid() > 0.5
            perception_correct = perception_prediction.eq(perceptions).float() * perception_mask.unsqueeze(-1)
            perception_acc = perception_correct.sum() / 5 / perception_mask.sum()
        return SoFunOutput(
            action_loss=action_loss,
            perception_loss=perception_loss,
            action_acc=action_acc,
            perception_acc=perception_acc,
        )

    def forward(
        self,
        input_output_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions=None,
        perceptions=None,
        perception_mask=None,
        **kwargs,
    ):
        # actions: (batch_size, num_actions) with padding value -100
        # perceptions: (batch_size, num_perceptions) milti-hot object representation
        # perception_mask: (batch_size, num_perceptions) mask for padding
        outputs = super().forward(
            input_output_states=input_output_states,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs,
        )
        last_hidden_state = outputs.encoder_last_hidden_state
        action_logits = self.action_head(last_hidden_state).permute(0, 2, 1)
        perception_logits = self.perception_head(last_hidden_state)
        action_loss = None
        perception_loss = None
        if actions is not None:
            action_loss = nn.CrossEntropyLoss()(action_logits, actions)
            action_acc = self.action_accuracy(action_logits, actions)
        if perceptions is not None:
            perception_loss = nn.BCEWithLogitsLoss(reduction="none")(perception_logits, perceptions)
            perception_loss = perception_loss * perception_mask.unsqueeze(-1)
            perception_loss = perception_loss.sum() / perception_mask.sum()
            # get the perception accuracy with perception_mask
            perception_prediction = perception_logits.detach().sigmoid() > 0.5
            perception_correct = perception_prediction.eq(perceptions).float() * perception_mask.unsqueeze(-1)
            perception_acc = perception_correct.sum() / 5 / perception_mask.sum()

        return SoFunOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            action_loss=action_loss,
            perception_loss=perception_loss,
            action_acc=action_acc,
            perception_acc=perception_acc,
        )
