import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from Convolutions.models import SparseConv2d

Tensor = torch.Tensor

class DenseCore(nn.Module):
    r"""Densely connected convolutional core.

    In a DenseNet core, each layer receives inputs from all previous layers, and
    all layers are concatenated as the final features. Multiple scales of the
    feature map are provided as outputs, and a spatial transformer will read a
    local patch on each map at the same position.

    """

    def __init__(self,
                 core_channels: List[int],
                 kernel_sizes: List[int],
                 in_channels: int = 1,
                 activation: str = 'ELU',
                 scale_num: int = 6,
                 scale_factor: float = 0.5,
                 readout_resol: int = 1,
                 i_transform: Optional[dict] = None,
                 ):
        r"""
        Args
        ----
        core_channels:
            The feature number of each layer.
        kernel_sizes:
            The kernel size of each layer.
        in_channels:
            The input feature channel.
        activation:
            The activation function, can be 'ELU' or 'ReLU'.
        scale_num:
            The number of scales.
        scale_factor:
            The scaling factor.
        readout_resol:
            The resolution of local patch to read from each scale.
        i_transform:
            The input image transformation, specified with keys 'shift' and
            'scale'.

        """
        super(DenseCore, self).__init__()
        assert len(core_channels)==len(kernel_sizes), 'require same number of channels and kernel sizes'
        for k_size in kernel_sizes:
            assert k_size%2==1, 'all kernel sizes should be odd'

        self.in_channels = in_channels
        assert activation in ['ELU', 'ReLU']
        if activation=='ELU':
            self.activation = nn.ELU()
        if activation=='ReLU':
            self.activation = nn.ReLU()

        self.scale_num, self.scale_factor = scale_num, scale_factor
        self.readout_resol = readout_resol

        self.layers = nn.ModuleList()
        for i, (out_c, k_size) in enumerate(zip(core_channels, kernel_sizes)):
            self.layers.append(nn.Sequential(
                SparseConv2d(36, 64, self.in_channels if i==0 else sum(core_channels[:i]), out_c, k_size, padding=k_size//2, connect_type="shuffle")
                ,
                nn.BatchNorm2d(out_c),
                self.activation,
            ))
        print("hi!")
        self.out_channels = sum(core_channels)
        self.patch_vol = self.out_channels*self.scale_num*(readout_resol**2)

        if i_transform is None:
            self.i_shift = nn.Parameter(torch.zeros(in_channels), requires_grad=False)
            self.i_scale = nn.Parameter(torch.ones(in_channels), requires_grad=False)
        else:
            assert i_transform['shift'].shape==(in_channels,)
            assert i_transform['scale'].shape==(in_channels,)
            self.i_shift = nn.Parameter(i_transform['shift'], requires_grad=False)
            self.i_scale = nn.Parameter(i_transform['scale'], requires_grad=False)

    def laplace_reg(self, layer_num: int = 1) -> Tensor:
        r"""Returns Laplacian regularizer on convolutional kernels.

        Laplacian loss is calculated for convolutional kernels of several bottom
        layers.

        Args
        ----
        layer_num:
            The number of bottom layers to regularize.

        Returns
        -------
        reg_loss:
            A scalar tensor of the averaged Laplacian loss.

        """
        lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])

        param_count, reg_loss = 0, 0.
        for layer_idx in range(layer_num):
            conv_w = self.layers[layer_idx][0].weight

            param_count += np.prod(conv_w.shape)
            H, W = conv_w.kernel_size, conv_w.kernel_size
            reg_loss = reg_loss+F.conv2d(conv_w.view(-1, 1, H, W), lap_kernel[None, None].to(conv_w)).pow(2).sum()
        reg_loss = reg_loss/param_count
        return reg_loss


    def weight_reg(self) -> Tensor:
        r"""Returns L2 regularizer on convolutional kernels.

        Square loss is calculated for convolutional kernels of all layers.

        Returns
        -------
        reg_loss:
            A scalar of averaged square loss.

        """
        param_count, reg_loss = 0, 0.
        for layer in self.layers:
            param_count += np.prod(layer[0].weight.shape)
            reg_loss = reg_loss+layer[0].weight.pow(2).sum()
        reg_loss = reg_loss/param_count
        return reg_loss

    def forward(self, images: Tensor) -> List[Tensor]:
        r"""Performs forward pass of the core.

        Args
        ----
        images: (N, C_in, H, W)
            A batch of input images, with values range in :math:`[0, 1]`.

        Returns
        -------
        features:
            The feature maps of different scales, each element is of shape
            `(N, C_out, H_i, W_i)`. `C_out` is the sum of all feature channel
            numbers of all layers.

        """
        images = (images-self.i_shift[:, None, None])/self.i_scale[:, None, None]
        features = []
        for i, layer in enumerate(self.layers):
            if i==0:
                features.append(layer(images))
            else:
                # dense connection from all previous layers
                features.append(layer(torch.cat(features, dim=1)))
        features = [torch.cat(features, dim=1)]
        for i in range(1, self.scale_num):
            features.append(F.interpolate(
                features[-1], scale_factor=self.scale_factor,
                mode='area', recompute_scale_factor=False,
            ))
        return features


class MultiLayerPerceptron(nn.Module):
    r"""Multi-layer perceptron (MLP).

    The MLP stacks linear layers with nonlinear activation function at hidden
    layers.

    """

    def __init__(self,
                 in_feature: int,
                 out_feature: int,
                 hidden_features: Optional[List[int]] = None,
                 activation: str = 'ReLU',
                 ):
        r"""
        Args
        ----
        in_feature, out_feature: int
            The input and output feature number.
        hidden_features: list
            The feature number of each hidden layer.
        activation: str
            The activataion function, can be 'ELU' or 'ReLU'.

        """
        super(MultiLayerPerceptron, self).__init__()

        if hidden_features is None:
            hidden_features = []
        assert activation in ['ELU', 'ReLU']
        if activation=='ELU':
            self.activation = nn.ELU()
        if activation=='ReLU':
            self.activation = nn.ReLU()

        self.layers = nn.ModuleList()
        for in_f, out_f in zip([in_feature]+hidden_features, hidden_features+[out_feature]):
            self.layers.append(nn.Linear(in_f, out_f))

    def weight_reg(self) -> Tensor:
        r"""Returns L2 regularizer on weights.

        Returns
        -------
        reg_loss:
            A scalar of averaged square loss.

        """
        param_count, reg_loss = 0, 0.
        for layer in self.layers:
            param_count += np.prod(layer.weight.shape)
            reg_loss = reg_loss+layer.weight.pow(2).sum()
        reg_loss = reg_loss/param_count
        return reg_loss

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Performs forward pass of the MLP.

        Args
        ----
        inputs: (batch_size, in_feature)
            A batch of inputs.

        Returns
        -------
        outputs: (batch_size, out_feature)
            A batch of outputs.

        """
        for i, layer in enumerate(self.layers):
            if i==0:
                outputs = layer(inputs)
            else:
                outputs = layer(self.activation(outputs))
        return outputs

class Modulator(MultiLayerPerceptron):
    r"""Modulator for neuron-specific gain from behaviors.

    The modulator receives a 3-D mice behavior data and returns response gains
    for all neurons. Outputs of an MLP is passed to a softplus function to be
    positive.

    """

    def __init__(self,
                 neuron_num: int,
                 features: List[int],
                 activation: str = 'ReLU',
                 b_transform: Optional[dict] = None,
                 ):
        r"""
        Args
        ----
        neuron_num: int
            The number of neurons.
        features: list
            The feature number of each hidden layer.
        activation: str
            The activataion function, can be 'ELU' or 'ReLU'.
        b_transform: dict
            The input behavior transformation, specified with keys `'shift'` and
            `'scale'`.

        """
        super(Modulator, self).__init__(3, neuron_num, features, activation)
        self.neuron_num = neuron_num

        if b_transform is None:
            self.b_shift = nn.Parameter(torch.zeros(3), requires_grad=False)
            self.b_scale = nn.Parameter(torch.ones(3), requires_grad=False)
        else:
            assert b_transform['shift'].shape==(3,)
            assert b_transform['scale'].shape==(3,)
            self.b_shift = nn.Parameter(b_transform['shift'], requires_grad=False)
            self.b_scale = nn.Parameter(b_transform['scale'], requires_grad=False)

    def forward(self, behaviors: Tensor) -> Tensor:
        r"""Implements forward pass of the modulator.

        Args
        ----
        behaviors: (batch, 3)
            The mice behavior data.

        Returns
        -------
        gains: (batch_size, neuron_num)
            The neuron-specific gains.

        """
        behaviors = (behaviors-self.b_shift)/self.b_scale
        gains = F.softplus(super(Modulator, self).forward(behaviors), beta=np.log(2))
        return gains

class Shifter(MultiLayerPerceptron):
    r"""Shifter based on eye positions.

    The shifter converts mice eye position to a readout location from feature
    maps.

    """

    def __init__(self,
                 features: List[int],
                 activation: str = 'ReLU',
                 p_transform: Optional[dict] = None,
                 ):
        r"""
        Args
        ----
        features:
            The feature numbers of hidden layers.
        activation:
            The activataion function, can be ``'ELU'`` or ``'ReLU'``.
        p_transform:
            The input pupil center transformation, specified with keys `'shift'`
            and `'scale'`.

        """
        super(Shifter, self).__init__(2, 2, features, activation)

        if p_transform is None:
            self.p_shift = nn.Parameter(torch.zeros(2), requires_grad=False)
            self.p_scale = nn.Parameter(torch.ones(2), requires_grad=False)
        else:
            assert p_transform['shift'].shape==(2,)
            assert p_transform['scale'].shape==(2,)
            self.p_shift = nn.Parameter(p_transform['shift'], requires_grad=False)
            self.p_scale = nn.Parameter(p_transform['scale'], requires_grad=False)

    def forward(self, pupil_centers: Tensor) -> Tensor:
        r"""Implements forward pass of the shifter.

        Args
        ----
        pupil_centers: (batch_size, 2)
            The mice eye positions.

        Returns
        -------
        shifts: (batch_size, 2)
            The readout shifts, with values in the range (-1, 1).

        """
        pupil_centers = (pupil_centers-self.p_shift)/self.p_scale
        shifts = torch.tanh(super(Shifter, self).forward(pupil_centers))
        return shifts


class NeuralModel(nn.Module):
    r"""Neural predictive model.

    A full model that predicts neural responses is constructed from a
    convolutional core, modulator and shifter.

    """

    def __init__(self,
                 core: DenseCore,
                 modulator: Modulator,
                 shifter: Shifter,
                 patch_size: Optional[Tuple[int]] = None,
                 activation: str = 'Softplus',
                 bank_size: int = 12,
                 ):
        r"""
        Args
        ----
        core:
            The convolutional core for getting image features.
        modulator:
            The gain modulator using behavior data.
        shifter:
            The readout shifter using eye position data.
        patch_size:
            The normalized size of readout patch size, `(h, w)`. It is only
            valid when `core.readout_resol` is greater than 1.
        activation:
            The activation after the linear readout from feature maps, can be
            'ReLU' or 'Softplus'.
        bank_size:
            The size of default `behaviors` and `pupil_centers`. The default
            banks are samples from training set, and are used to output response
            when behavior data is not provided.

        """
        super(NeuralModel, self).__init__()
        self.core, self.modulator, self.shifter = core, modulator, shifter

        self.neuron_num = self.modulator.neuron_num
        self.readout_locs = nn.Parameter(torch.rand((self.neuron_num, 2), dtype=torch.float)*.2-.1)
        in_features = core.patch_vol
        self.readout_weight = nn.Parameter(torch.randn((self.neuron_num, in_features), dtype=torch.float)/(in_features**0.5))
        self.readout_bias = nn.Parameter(torch.zeros((self.neuron_num,), dtype=torch.float))

        if core.readout_resol>1:
            # prepare a meshgrid of readout locations
            if patch_size is None:
                patch_size = [18/36, 18/64]

            dh, dw = torch.meshgrid(
                torch.linspace(-patch_size[0], patch_size[0], core.readout_resol, dtype=torch.float),
                torch.linspace(-patch_size[1], patch_size[1], core.readout_resol, dtype=torch.float),
            )
            self.basic_grid = nn.Parameter(torch.stack([dh, dw], dim=2).view(-1, 2), requires_grad=False)
            self.margin = nn.Parameter(torch.tensor(patch_size, dtype=torch.float), requires_grad=False)
        else:
            self.basic_grid = nn.Parameter(torch.zeros((1, 2), dtype=torch.float), requires_grad=False)
            self.margin = nn.Parameter(torch.zeros(2, dtype=torch.float), requires_grad=False)

        assert activation in ['ReLU', 'Softplus']
        if activation=='ReLU':
            self.activation = nn.ReLU()
        if activation=='Softplus':
            self.activation = nn.Softplus()

        self.bank_size = bank_size
        self.sampled_behaviors = nn.Parameter(torch.zeros(bank_size, 3, dtype=torch.float), requires_grad=False)
        self.sampled_pupil_centers = nn.Parameter(torch.zeros(bank_size, 2, dtype=torch.float), requires_grad=False)

    def readout_weight_reg(self) -> Tensor:
        r"""Returns L2 regularizer on readout weights.

        Returns
        -------
        reg_loss:
            A scalar of square loss.

        """
        reg_loss = self.readout_weight.pow(2).mean()
        return reg_loss

    def readout_loc_reg(self, order: int = 6) -> Tensor:
        r"""Returns power regularizer on readout locations.

        Readout locations are normalized to (-1, 1), therefore a power loss
        encourages the neurons to read near image center.

        Returns
        -------
        reg_loss:
            A scalar of readout location loss.

        """
        assert order>0 and order%2==0
        reg_loss = self.readout_locs.pow(order).mean()
        return reg_loss

    def forward(self,
        images: Tensor,
        behaviors: Optional[Tensor] = None,
        pupil_centers: Optional[Tensor] = None,
    ):
        r"""Implements forward pass of the full model.

        If either `bahaviors` or `pupil_centers` is not provided, `gains` or
        `shifts` will be calculated as the averaged output using default values
        from the bank, which is constantly updated during training.

        Args
        ----
        images: (batch_size, C, H, W)
            Input images whose values are in [0, 1].
        behaviors: (batch_size, 3)
            The mice behavior data, e.g. running speed, pupil size.
        pupil_centers: (batch_size, 2)
            The mice eye position data.

        Returns
        -------
        outputs: (batch_size, neuron_num)
            The neural responses of all neurons.

        """
        features = self.core(images) # list of tensors (batch_size, C_out, H_i, W_i)

        if pupil_centers is None:
            shifts = self.shifter(self.sampled_pupil_centers).mean(dim=0, keepdim=True).expand(images.shape[0], -1) # (batch_size, 2)
        else:
            shifts = self.shifter(pupil_centers) # (batch_size, 2)
            if self.training: # update bank from training set
                update_num = min(self.bank_size, len(pupil_centers))
                for i, j in zip(
                    random.sample(range(self.bank_size), update_num),
                    random.sample(range(len(pupil_centers)), update_num),
                ):
                    self.sampled_pupil_centers.data[i] = pupil_centers[j]
        grid = (torch.tanh(shifts[:, None]+self.readout_locs[None])*(1-self.margin))[:, :, None]+self.basic_grid[None, None] # (batch_size, neuron_num, resol**2, 2)
        features = [F.grid_sample(f, grid, padding_mode='border', align_corners=True) for f in features]

        features = torch.cat(features, dim=1) # (batch_size, channels*scale_num, neuron_num, resol**2)
        features = features.permute(0, 2, 1, 3).reshape(-1, self.neuron_num, self.core.patch_vol) # (batch_size, neuron_num, patch_vol)
        outputs = self.activation((features*self.readout_weight).sum(dim=2)+self.readout_bias) # (batch_size, neuron_num)

        if behaviors is None:
            gains = self.modulator(self.sampled_behaviors).mean(dim=0) # (neuron_num,)
        else:
            gains = self.modulator(behaviors) # (batch_size, neuron_num)
            if self.training:
                update_num = min(self.bank_size, len(behaviors))
                for i, j in zip(
                    random.sample(range(self.bank_size), update_num),
                    random.sample(range(len(behaviors)), update_num),
                ):
                    self.sampled_behaviors.data[i] = behaviors[j]
        outputs = outputs*gains
        return outputs
