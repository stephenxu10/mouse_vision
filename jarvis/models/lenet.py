# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:16:01 2021

@author: Zhe
"""

from typing import List, Tuple, Any

import torch
import torch.nn as nn

from . import ImageClassifier


class LeNet(ImageClassifier):
    r"""LeNet model architecture from the
    `"Backpropagation Applied to..." <https://www.mitpressjournals.org/doi/abs/10.1162/neco.1989.1.4.541>`_
    paper.

    """

    def __init__(
            self,
            act_fun: str = 'ReLU',
            **kwargs: Any,
            ) -> None:
        super(LeNet, self).__init__(**kwargs)
        in_channels, class_num = self.in_channels, self.class_num

        if act_fun=='ReLU':
            self.act_fun = nn.ReLU()
        else:
            raise RuntimeError(f"activation function '{act_fun}' not recognized")
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(6, 16, kernel_size=5, padding=2),
                ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 120, kernel_size=5, padding=2),
                ),
            nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(1),
                nn.Linear(120, 84),
                ),
            ])
        self.fc = nn.Linear(84, out_features=class_num)

    def layer_activations(
            self,
            x: torch.Tensor
            ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        r"""Returns activations of all layers.

        Args
        ----
        x: (N, C, H, W), tensor
            The normalized input images.

        Returns
        -------
        pre_acts, post_acts: list of tensors
            The pre- and post-activations for each layer except the last one.
        logits: tensor
            The logits.

        """
        pre_acts, post_acts = [], []
        for layer in self.layers:
            x = layer(x)
            pre_acts.append(x)
            x = self.act_fun(x)
            post_acts.append(x)
        logits = self.fc(x)
        return pre_acts, post_acts, logits


def lenet(**kwargs: Any) -> LeNet:
    model = LeNet(**kwargs)
    return model
