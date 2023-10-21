import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod, abstractproperty


class BaseModel(nn.Module, ABC):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def predict(
        self, loader: torch.utils.data.DataLoader, device: torch.device, gan=True
    ) -> tuple:
        raise NotImplementedError

    def freeze_encoder(self) -> None:
        if hasattr(self, "encoder"):
            for name, parameter in self.encoder.named_parameters():
                parameter.requires_grad = False
        else:
            raise ModuleNotFoundError("Encoder not found")
