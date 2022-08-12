from typing import Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor


class Normalizer(nn.Module):
    r"""Image normalizer."""

    def __init__(self,
        mean: list[float],
        std: list[float],
    ):
        super(Normalizer, self).__init__()

        assert len(mean)==1 or len(std)==1 or len(mean)==len(std), "Shape of mean and std are inconsistent."
        assert min(std)>0, "std has to be positive."
        self.mean = nn.Parameter(
            torch.tensor(mean, dtype=torch.float)[..., None, None], requires_grad=False,
        )
        self.std = nn.Parameter(
            torch.tensor(std, dtype=torch.float)[..., None, None], requires_grad=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        return (images-self.mean)/self.std


class ImageClassifier(nn.Module):
    r"""A base class for image classifier."""

    def __init__(self,
        in_channels: int = 3,
        num_classes: int = 10,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        in_channels:
            The number of input channels.
        num_classes:
            The number of classes.
        mean, std: list of floats
            The mean and std parameters for input normalization.

        """
        super(ImageClassifier, self).__init__()
        self.in_channels, self.num_classes = in_channels, num_classes

        if mean is None:
            if self.in_channels==3:
                mean = [0.485, 0.456, 0.406]
            else:
                mean = [0.5]
        if std is None:
            if self.in_channels==3:
                std = [0.229, 0.224, 0.225]
            else:
                std = [0.2]
        self.normalizer = Normalizer(mean, std)

    def layer_activations(self, x: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor]:
        r"""
        Args
        ----
        x: (*, C, H, W)
            Normalized images.

        Returns
        -------
        pre_acts:
            Layer activations before non-linearity.
        post_acts:
            Layer activations after non-linearity.
        logits: (*, num_classes)
            Last layer activations as logits.

        """
        raise NotImplementedError

    def forward(self, images: Tensor) -> Tensor:
        r"""
        Args
        ----
        images: (*, C, H, W)
            Input images whose values are between [0, 1].

        Returns
        -------
        logits: (*, num_classes)
            Logits of distribution.

        """
        *_, logits = self.layer_activations(self.normalizer(images))
        return logits
