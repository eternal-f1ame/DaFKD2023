import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict
from model.base import BaseModel
from model.nlp.utils import ClassifierOutput


class Bone(BaseModel):
    def get_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.encoder_name)

    def freeze_backbone(self) -> None:
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad = False

    def safe_encoder(self, token_type_ids=None, *args, **kwargs):
        if "distil" in self.encoder_name:
            return self.encoder(*args, **kwargs)
        return self.encoder(token_type_ids=token_type_ids, *args, **kwargs)

class DiscriminatorForMultiLabelClassification(Bone):
    """Discriminator model for sequence classification tasks with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 4,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        **kwargs,
    ):
        super(DiscriminatorForMultiLabelClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.dropout = nn.Dropout(dropout_rate if classifier_dropout is None else classifier_dropout)
        if gan_training:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels + 1)
        else:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        if input_ids is None and external_states is None:
            raise AssertionError("Empty input: input_ids and external states are empty")

        if input_ids is None:
            sequence_output = external_states
        else:
            outputs = self.safe_encoder(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids
            )
            sequence_output = outputs.last_hidden_state[:, 0]  # get CLS embedding
            if external_states is not None:
                sequence_output = torch.cat([sequence_output, external_states], dim=0)
        sequence_output_drop = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_drop)
        fake_probs, fake_logits = None, None
        if self.gan_training:
            fake_logits = logits[:, [-1]]
            fake_probs = self.sigmoid(fake_logits)
            logits = logits[:, :-1]
        probs = self.sigmoid(logits)
        loss = self.compute_loss(
            logits=logits, probs=probs, fake_probs=fake_probs, labels=labels, labeled_mask=labeled_mask
        )
        return ClassifierOutput(
            loss=loss["real_loss"],
            fake_loss=loss["fake_loss"],
            logits=logits,
            fake_logits=fake_logits,
            probs=probs,
            fake_probs=fake_probs,
            hidden_states=sequence_output,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        fake_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        fake_probs: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.FloatTensor]:
        real_loss = torch.FloatTensor([0]).to(self.encoder.device)
        fake_loss = torch.FloatTensor([0]).to(self.encoder.device)
        if labels is not None:
            if labeled_mask is not None:
                labeled_mask = labeled_mask.bool()
                logits = logits[labeled_mask]
                labels = labels[labeled_mask]
            if logits.shape[0] > 0:
                real_loss = self.loss_fct(logits, labels.float())
        if self.gan_training:
            fake_loss = -torch.mean(torch.log(fake_probs + self.epsilon))
        return {"real_loss": real_loss, "fake_loss": fake_loss}
