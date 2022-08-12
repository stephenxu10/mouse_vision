import torch
import torch.nn as nn
from .base import ImageClassifier


class WrappedClassifier(ImageClassifier):

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        super(WrappedClassifier, self).__init__(**kwargs)
        self.raw_model = model

    def forward(self,
        images: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if images.shape[1]==1 and self.in_channels==3:
            images = images.expand(-1, 3, -1, -1)
        logits = self.raw_model(self.normalizer(images), **kwargs)
        return logits
