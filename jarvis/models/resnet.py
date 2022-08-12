from typing import Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor

from .base import ImageClassifier


class ResBlock(nn.Module):
    r"""ResNet block."""

    def __init__(self,
        block_type: str,
        in_channels: int,
        base_channels: int,
        stride: int = 1,
    ):
        r"""
        Args
        ----
        block_type:
            The block type, can be ``Basic`` or ``Bottleneck``.
        in_channels:
            The input channel number.
        base_channels:
            The base channel number for layers.
        stride:
            The stride for the first convolution.

        """
        super(ResBlock, self).__init__()
        self.block_type = block_type
        if self.block_type=='Basic':
            expansion = 1
        if self.block_type=='Bottleneck':
            expansion = 4
        out_channels = expansion*base_channels
        self.in_channels, self.out_channels = in_channels, out_channels

        if self.block_type=='Basic':
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(base_channels),
            )
            self.nonlinear0 = nn.ReLU()
            self.layer1 = nn.Sequential(
                nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
            )
            self.nonlinear1 = nn.ReLU()
        if self.block_type=='Bottleneck':
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(base_channels),
            )
            self.nonlinear0 = nn.ReLU()
            self.layer1 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(base_channels),
            )
            self.nonlinear1 = nn.ReLU()
            self.layer2 = nn.Sequential(
                nn.Conv2d(base_channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.nonlinear2 = nn.ReLU()

        if stride==1 and in_channels==out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def layer_activations(self,
        x: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        r"""Returns activations of all layers.

        Args
        ----
        x: (*, C, H, W)
            The input to this block.

        Returns
        -------
        pre_acts, post_acts:
            The pre and post activations for each layer.

        """
        if self.block_type=='Basic':
            pre0 = self.layer0(x)
            post0 = self.nonlinear0(pre0)
            pre1 = self.layer1(post0)+self.shortcut(x)
            post1 = self.nonlinear1(pre1)
            pre_acts = [pre0, pre1]
            post_acts = [post0, post1]
        if self.block_type=='Bottleneck':
            pre0 = self.layer0(x)
            post0 = self.nonlinear0(pre0)
            pre1 = self.layer1(post0)
            post1 = self.nonlinear1(pre1)
            pre2 = self.layer2(post1)+self.shortcut(x)
            post2 = self.nonlinear2(pre2)
            pre_acts = [pre0, pre1, pre2]
            post_acts = [post0, post1, post2]
        return pre_acts, post_acts

    def forward(self, x: Tensor) -> Tensor:
        r"""Implements the forward pass of the ResNet block.

        Args
        ----
        x: (*, C, H, W)
            The input to this block.

        Returns
        -------
        The output of the block.

        """
        _, post_acts = self.layer_activations(x)
        return post_acts[-1]


class ResNet(ImageClassifier):
    r"""ResNet model.

    In addition to an input section, four more sections are stacked to form a
    classical `ResNet model <https://arxiv.org/abs/1512.03385>`_. Each section
    is composed of a few ResNet blocks, either basic ones or bottleneck ones.
    Within each section the spatial size of feature maps does not change.

    """

    def __init__(self,
        num_blocks: list[int],
        block_type: str,
        conv0_channels: int = 64,
        conv0_kernel_size: int = 7,
        conv0_stride: Optional[int] = None,
        base_channels: int = 64,
        **kwargs,
    ):
        r"""
        Args
        ----
        num_blocks: list
            The number of ResNet blocks in each section, usually of length 4.
        block_type: str
            The block type, can be ``Basic`` or ``Bottleneck``.
        conv0_channels: int
            The channel number for the first convolution layer.
        conv0_kernel_size: int
            The kernel size for the first convolution layer.
        base_channels: int
            The base channel number, used for ResBlock sections.

        """
        super(ResNet, self).__init__(**kwargs)
        in_channels, num_classes = self.in_channels, self.num_classes

        assert block_type in ['Basic', 'Bottleneck']
        self.block_nums, self.block_type = num_blocks, block_type
        self.section_num = len(num_blocks)

        assert conv0_kernel_size%2==1
        if conv0_stride is None:
            conv0_stride = 1 if conv0_kernel_size<7 else 2
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels, conv0_channels, kernel_size=conv0_kernel_size,
                padding=conv0_kernel_size//2, stride=conv0_stride, bias=False,
            ),
            nn.BatchNorm2d(conv0_channels),
        )
        self.nonlinear0 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if conv0_stride>1 else nn.Sequential()

        in_channels = conv0_channels
        base_channels = [base_channels*(2**i) for i in range(self.section_num)]
        strides = [1]+[2]*(self.section_num-1)

        self.sections = nn.ModuleList()
        for i in range(self.section_num):
            section, in_channels = self._make_section(
                num_blocks[i], block_type, in_channels, base_channels[i], strides[i],
            )
            self.sections.append(section)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def _make_section(self,
        num_blocks: int,
        block_type: str,
        in_channels: int,
        base_channels: int,
        stride: int,
    ) -> tuple[nn.Module, int]:
        r"""Constructs one ResNet section.

        Args
        ----
        num_blocks:
            The number of ResNet blocks in the section.
        block_type:
            The block type.
        in_channels:
            The input channel number.
        base_channels:
            The base channel number for this section.
        stride:
            The stride of the first ResNet block.

        Returns
        -------
        section:
            The constructed ResNet section.
        out_channels:
            The output channel number.

        """
        blocks = [ResBlock(block_type, in_channels, base_channels, stride)]
        out_channels = blocks[0].out_channels
        for _ in range(num_blocks-1):
            blocks.append(ResBlock(block_type, out_channels, base_channels))
        section = nn.ModuleList(blocks)
        return section, out_channels

    def layer_activations(self,
        x: Tensor
    ) -> tuple[list[Tensor], list[Tensor], Tensor]:
        r"""Returns activations of all layers."""
        pre0 = self.conv0(x)
        post0 = self.maxpool(self.nonlinear0(pre0))
        pre_acts, post_acts = [pre0], [post0]

        for section in self.sections:
            for block in section:
                _pre_acts, _post_acts = block.layer_activations(post_acts[-1])
                pre_acts += _pre_acts
                post_acts += _post_acts

        logits = self.fc(self.avgpool(post_acts[-1]).flatten(1))
        return pre_acts, post_acts, logits

    def load_pytorch_model(self, p_state, normalizer_mean=None, normalizer_std=None):
        r"""Loads state dict from a pre-trained pytorch resnet model.

        The key of state dictionary is renamed for compatiblity, and the fully
        connected layer are only loaded when the number of classes is correct.

        Args
        ----
        p_state:
            The state dict of a pytorch resnet model.
        normalizer_mean, normalizer_std:
            The mean and std for input normalizer. Default values will be used
            if they are not provided.

        """
        self.cpu()
        j_state = {} # state dict of jarvis resnet model
        if normalizer_mean is not None:
            j_state['normalizer.mean'] = normalizer_mean.cpu().reshape(self.in_channels, 1, 1)
        if normalizer_std is not None:
            j_state['normalizer.std'] = normalizer_std.cpu().reshape(self.in_channels, 1, 1)

        for key in p_state:
            if key.startswith('conv1'):
                new_key = key.replace('conv1', 'conv0.0')
            elif key.startswith('bn1'):
                new_key = key.replace('bn1', 'conv0.1')
            elif key.startswith('fc'):
                if p_state[key].shape[0]!=self.num_classes:
                    continue # load the core only when fc layer has different shape
                new_key = key
            elif key.startswith('layer'):
                new_key = key.replace('layer', 'sections.')
                new_key = new_key[:9]+str(int(new_key[9])-1)+new_key[10:] # can only handle sections fewer than 10
                for s, t in zip(
                        ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'downsample'],
                        ['layer0.0', 'layer0.1', 'layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'shortcut']
                        ):
                    new_key = new_key.replace(s, t)
            j_state[new_key] = p_state[key].cpu()

        self.load_state_dict(j_state, strict=False)


def resnet18(**kwargs):
    return ResNet(num_blocks=[2, 2, 2, 2], block_type='Basic', **kwargs)


def resnet34(**kwargs):
    return ResNet(num_blocks=[3, 4, 6, 3], block_type='Basic', **kwargs)


def resnet50(**kwargs):
    return ResNet(num_blocks=[3, 4, 6, 3], block_type='Bottleneck', **kwargs)


def resnet101(**kwargs):
    return ResNet(num_blocks=[3, 4, 23, 3], block_type='Bottleneck', **kwargs)


def resnet152(**kwargs):
    return ResNet(num_blocks=[3, 8, 36, 3], block_type='Bottleneck', **kwargs)
