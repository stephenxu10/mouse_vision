# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:13:59 2020

@author: Zhe
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def loo_mean(responses):
    r"""Returns leave-one-out mean responses.

    Args
    ----
    responses: (repeat_num, neuron_num), ndarray
        The responses of all trials corresponding to one oracle stimulus.

    Returns
    -------
    An ndarray of shape `(repeat_num, neuron_num)`, containing leave-one-out
    mean responses for each trial.

    """
    return (responses.sum(axis=0)-responses)/(responses.shape[0]-1)


def response_corrs(responses_1, responses_2):
    r"""Returns correlation coefficients of all neurons.

    Args
    ----
    responses_1, responses_2: (trial_num, neuron_num), ndarray
        The responses to calculate correlation of. Usually one is true data,
        and the other is estimation.

    Returns
    -------
    An ndarray of shape `(neuron_num,)`, containing correlation coefficients
    for each neuron.

    """
    return np.array([np.corrcoef(r_1, r_2)[0, 1] for \
                     r_1, r_2 in zip(responses_1.T, responses_2.T)])


def oracle_frac(loo_corrs, pred_corrs, neuron_weights=None):
    r"""Returns oracle fraction.

    Args
    ----
    loo_corrs: (neuron_num,), ndarray
        The leave-one-out prediction correlation coefficients of all neurons.
    pred_corrs: (neuron_num,), ndarray
        The model prediction correlation coefficients of all neurons, estimated
        using non-visual inputs of each trial.
    neuron_weights: nn.Parameter
        Neuron weights from ResponseLoss object.

    Returns
    -------
    o_frac: float
        Oracle fraction calculated by fitting a linear function with zero
        intercept. This value could be greater than 1 since the model uses real
        non-visual inputs.

    """
    if neuron_weights is None:
        w = np.ones(len(loo_corrs))
    else:
        w = neuron_weights.data.cpu().numpy()
    o_frac = np.sum(w*loo_corrs*pred_corrs)/np.sum(w*loo_corrs*loo_corrs)
    return o_frac


class ResponseLoss(nn.Module):
    r"""Poisson loss for training predictive models.

    Args
    ----
    neuron_weight: (neuron_num,), tensor
        The positive weight assigned to each neuron.
    eps: float
        Epsilon used for numerical stability.
    reduction: str
        Reduction mode, can be ``'none'``, ``'mean'`` or ``'sum'``.

    """
    def __init__(self, neuron_weights=None, eps=1e-8, reduction='mean'):
        super(ResponseLoss, self).__init__()

        if neuron_weights is None:
            self.neuron_weights = None
        else:
            assert torch.all(neuron_weights>=0.)
            self.neuron_weights = nn.Parameter(neuron_weights, requires_grad=False)

        self.eps = eps
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction

    def forward(self, outputs, targets):
        loss = F.poisson_nll_loss(outputs, targets, log_input=False, full=False,
                                  eps=self.eps, reduction='none')
        if self.neuron_weights is None:
            loss = loss.mean(dim=1)
        else:
            loss = (loss*self.neuron_weights).sum(dim=1)/self.neuron_weights.sum()
        if self.reduction=='none':
            return loss
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='sum':
            return loss.sum()


class Stopper:
    r"""Stopper for training control.

    If the validation loss does not decrease for a certain of epochs, training
    jumps back to the last best epoch. If certain number of jumps happen before
    the maximum epoch number, the training stops early.

    Args
    ----
    min_epoch, max_epoch: int
        Minimum and maximum epoch number.
    jump_num: int
        The number of allowed jumps.
    cool_down: int
        The minimum number of epochs between two jumps.
    tolerance: int
        The number of non-improving epochs before a jump.

    """
    def __init__(self, min_epoch, max_epoch, jump_num, cooldown, tolerance):
        assert min_epoch<max_epoch
        self.min_epoch, self.max_epoch = min_epoch, max_epoch
        self.jump_num = jump_num
        self.cooldown = cooldown
        assert min_epoch>=tolerance
        self.tolerance = tolerance

        self.count, self.timer = 0, 0

    def step(self, losses_valid):
        r"""Updates stopper.

        Args
        ----
        losses_valid: list
            The list of validation loss.

        Returns
        -------
        completed: bool
            Whether the training is completed.
        last_idx: int
            The epoch index to jump to.

        """
        self.timer = max(0, self.timer-1)
        epoch_num = len(losses_valid)-1
        completed, last_idx = False, epoch_num
        if epoch_num==self.max_epoch:
            completed = True
        elif epoch_num>=self.min_epoch and self.timer==0:
            if losses_valid[-self.tolerance]<=min(losses_valid[-self.tolerance:]):
                last_idx = losses_valid.index(min(losses_valid))
                self.count += 1
                self.timer = self.cooldown
                if self.count>=self.jump_num:
                    completed = True
        return completed, last_idx
