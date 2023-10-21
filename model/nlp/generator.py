import torch
import torch.nn as nn

from base import BaseModel
from typing import Optional
from model.utils import CustomAttention


class SimpleSequenceGenerator(BaseModel):
    """Generator model class"""

    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.2, **kwargs):
        super(SimpleSequenceGenerator, self).__init__()
        self.dropout_rate = dropout_rate
        layers = self.get_block(input_size, output_size)
        layers.extend(self.get_block(output_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor):
        return self.layers(noise)

    def get_block(self, input_size: int, output_size: int):
        return [
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(output_size),
        ]


class SimpleTokenGenerator(BaseModel):
    """Generator model class"""

    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.2, **kwargs):
        super(SimpleTokenGenerator, self).__init__()
        self.dropout_rate = dropout_rate
        layers = self.get_block(input_size, output_size)
        layers.extend(self.get_block(output_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor):
        return self.layers(noise)

    def get_block(self, input_size: int, output_size: int):
        return [
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(output_size),
        ]


class ContextualTokenGenerator(BaseModel):
    """Contextual Generator model class with Attention"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout_rate: float = 0.2,
        need_mixed_proj_layer: bool = False,
        **kwargs
    ):
        super(ContextualTokenGenerator, self).__init__()
        self.dropout_rate = dropout_rate
        self.attention = CustomAttention(
            in_features=input_size,
            embed_dim=64,
        )
        self.out = nn.Sequential(*self.get_block(input_size, output_size))
        if need_mixed_proj_layer:
            self.mixed_proj_layer = nn.Linear(output_size, input_size)

    def forward(self, noise: torch.Tensor, real_encoded_samples: Optional[torch.Tensor] = None):
        if real_encoded_samples is None:
            context_noise = self.attention(query=noise, key=noise, value=noise, need_weights=False)
        else:
            real_proj = self.mixed_proj_layer(real_encoded_samples)
            noise_and_real = torch.cat([noise, real_proj], dim=1)
            context_noise = self.attention(
                query=noise_and_real, key=noise_and_real, value=noise_and_real, need_weights=False
            )
        return self.out(context_noise)

    def get_block(self, input_size: int, output_size: int):
        return [
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(output_size),
        ]
