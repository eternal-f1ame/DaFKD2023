import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.modeling_utils import SequenceSummary

from typing import Optional, Tuple, Dict

from base import BaseModel
from model.utils import ClassifierOutput, CustomAttention


class Discriminator(BaseModel):
    def get_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.encoder_name)

    def freeze_backbone(self) -> None:
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad = False

    def safe_encoder(self, token_type_ids=None, *args, **kwargs):
        if "distil" in self.encoder_name:
            return self.encoder(*args, **kwargs)
        return self.encoder(token_type_ids=token_type_ids, *args, **kwargs)


class DiscriminatorForSequenceClassification(Discriminator):
    """Discriminator model for sequence classification tasks with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 10,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        **kwargs,
    ):
        super(DiscriminatorForSequenceClassification, self).__init__()
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
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = CrossEntropyLoss(ignore_index=ce_ignore_index)
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")
            self.fake_index = -1
            print(f"Default fake label index is {self.fake_index}")

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
        fake_logits, fake_probs = None, None
        if self.gan_training:
            fake_logits = logits[:, [-1]]
            fake_probs = self.softmax(logits)[:, [-1]]
            logits = logits[:, :-1]
        probs = self.softmax(logits)

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
                real_loss = self.loss_fct(logits, labels)
        if self.gan_training:
            fake_loss = -torch.mean(torch.log(fake_probs + self.epsilon))
        return {"real_loss": real_loss, "fake_loss": fake_loss}


class DiscriminatorForMultiLabelClassification(Discriminator):
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


class DiscriminatorForMultipleChoice(Discriminator):
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
        super(DiscriminatorForMultipleChoice, self).__init__()
        self.num_labels = num_labels
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.has_seq_summary = False
        if "electra" in self.encoder_name.lower():
            self.sequence_summary = SequenceSummary(self.encoder.config)
            self.has_seq_summary = True
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.dropout = nn.Dropout(dropout_rate if classifier_dropout is None else classifier_dropout)
        if gan_training:
            out_size_clf = 2
        else:
            out_size_clf = 1
        self.classifier = nn.Linear(self.encoder.config.hidden_size, out_size_clf)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = CrossEntropyLoss(ignore_index=ce_ignore_index)
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
            input_ids = input_ids.view(-1, input_ids.size(-1))
            input_mask = input_mask.view(-1, input_mask.size(-1))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            outputs = self.safe_encoder(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids
            )
            if not self.has_seq_summary:
                if token_type_ids is not None:
                    sequence_output = outputs[1]  # pooled output
                else:
                    sequence_output = outputs[0][:, 0]  # (bs * num_choices, dim)
            else:
                sequence_output = outputs[0]
                sequence_output = self.sequence_summary(sequence_output)
            # add generator input to hidden states
            if external_states is not None:
                sequence_output = torch.cat([sequence_output, external_states], dim=0)

        sequence_output_drop = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_drop)
        fake_probs, fake_logits = None, None
        if self.gan_training:
            fake_probs = self.softmax(logits)[:, [1]]
            fake_logits = logits[:, [1]]
            logits = logits[:, [0]]
        if input_ids is not None:
            logits = logits.view(-1, self.num_labels)
        probs = self.softmax(logits)
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
                real_loss = self.loss_fct(logits, labels)
        if self.gan_training:
            fake_loss = -torch.mean(torch.log(fake_probs + self.epsilon))
        return {"real_loss": real_loss, "fake_loss": fake_loss}


class DiscriminatorForTokenClassification(Discriminator):
    """Discriminator model for token classification tasks with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 10,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        **kwargs,
    ):
        super(DiscriminatorForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
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
        self.softmax = nn.Softmax(dim=-1)
        self.ignore_index = ce_ignore_index
        self.loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")
            self.fake_index = -1
            print(f"Default fake label index is {self.fake_index}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        if input_ids is None and external_states is None:
            raise AssertionError("Empty input: input_ids and external states are empty")

        if input_ids is None:
            sequence_output = external_states
        else:
            outputs = self.safe_encoder(input_ids=input_ids, attention_mask=input_mask)
            sequence_output = outputs[0]
            if external_states is not None:
                sequence_output = torch.cat([sequence_output, external_states], dim=0)

        sequence_output_drop = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_drop)
        fake_logits, fake_probs = None, None
        if self.gan_training:
            fake_logits = logits[:, :, [-1]]
            fake_probs = self.softmax(logits)[:, :, [-1]].view(-1, 1)
            logits = logits[:, :, :-1]
        probs = self.softmax(logits)

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
    ) -> Optional[torch.FloatTensor]:
        real_loss = torch.FloatTensor([0]).to(self.encoder.device)
        fake_loss = torch.FloatTensor([0]).to(self.encoder.device)
        if labels is not None:
            if labeled_mask is not None:
                labeled_mask = labeled_mask.bool()
                logits = logits[labeled_mask]
                labels = labels[labeled_mask]
            if logits.shape[0] > 0:
                real_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.gan_training:
            fake_loss = -torch.mean(torch.log(fake_probs + self.epsilon))
        return {"real_loss": real_loss, "fake_loss": fake_loss}


class DiscriminatorForContextTokenClassification(Discriminator):
    """Discriminator model for token classification tasks with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 10,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        attention_dim: int = 200,
        **kwargs,
    ):
        super(DiscriminatorForContextTokenClassification, self).__init__()

        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(self.encoder.config.hidden_size)
        self.prelinear_attention = CustomAttention(self.encoder.config.hidden_size, attention_dim)
        if gan_training:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels + 1)
        else:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.ignore_index = ce_ignore_index
        self.loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")
            self.fake_index = -1
            print(f"Default fake label index is {self.fake_index}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        if input_ids is None and external_states is None:
            raise AssertionError("Empty input: input_ids and external states are empty")

        if input_ids is None:
            sequence_output = external_states
        else:
            outputs = self.safe_encoder(input_ids=input_ids, attention_mask=input_mask)
            sequence_output = outputs[0]
            if external_states is not None:
                sequence_output = torch.cat([sequence_output, external_states], dim=0)

        context_outputs = self.prelinear_attention(
            query=sequence_output, key=sequence_output, value=sequence_output
        )
        context_outputs = self.norm(context_outputs + sequence_output)
        logits = self.classifier(context_outputs)
        fake_logits, fake_probs = None, None
        if self.gan_training:
            fake_logits = logits[:, :, [-1]]
            fake_probs = self.softmax(logits)[:, :, [-1]].view(-1, 1)
            logits = logits[:, :, :-1]
        probs = self.softmax(logits)

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
    ) -> Optional[torch.FloatTensor]:
        real_loss = torch.FloatTensor([0]).to(self.encoder.device)
        fake_loss = torch.FloatTensor([0]).to(self.encoder.device)
        if labels is not None:
            if labeled_mask is not None:
                labeled_mask = labeled_mask.bool()
                logits = logits[labeled_mask]
                labels = labels[labeled_mask]
            if logits.shape[0] > 0:
                real_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.gan_training:
            fake_loss = -torch.mean(torch.log(fake_probs + self.epsilon))
        return {"real_loss": real_loss, "fake_loss": fake_loss}
