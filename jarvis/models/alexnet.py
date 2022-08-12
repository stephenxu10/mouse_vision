# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:42:08 2020

@author: Zhe
"""

from typing import List, Any

import torch
import torch.nn as nn

from . import ImageClassifier


class AlexNet(ImageClassifier):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    """

    def __init__(
            self,
            **kwargs: Any,
            ) -> None:
        super(AlexNet, self).__init__(**kwargs)
        in_channels, class_num = self.in_channels, self.class_num

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ),
            nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ),
            nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(),
                ),
            nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten(1),
                ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(256*6*66, 4096),
                nn.ReLU(),
                ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                ),
            ])
        self.fc = nn.Linear(4096, class_num)

    def layer_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""Returns activations of all layers.

        Args
        ----
        x: (N, C, H, W), tensor
            The normalized input images.

        Returns
        -------
        acts: list of tensors
            The activations for each layer.
        logits: tensor
            The logits.

        """
        acts = []
        for layer in self.layers:
            x = layer(x)
            acts.append(x)
        logits = self.fc(x)
        return acts, logits


def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    return model
