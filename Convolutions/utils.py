"""
author: Zhe Li
email: zhe.li@bcm.edu
"""

import numpy as np
from typing import Tuple

Array = np.ndarray

def random_pos_channel(
    pos_in: Array,
    connect_type: str,
) -> Array:
    r"""Returns randomized input positions for one channel.

    Args
    ----
    pos_in: (K, S_out)
        The input position index describing a normal convolution connectivity.
        `K` is the number of kernel parameters for a fixed input-output channel
        pair, i.e. `kernel_size**2`. `S_out` is the number of output positions,
        i.e. `H_out*W_out`. `pos_in[k, pos_out]` is the input position index
        of the parameter `k` connecting to `pos_out`.
    connect_type: str
        The sparse connection type, can be 'normal', 'shuffle' or 'scatter'. No
        change will be made for 'normal'. Locality is preserved for 'shuffle'
        but not for 'scatter'.

    Returns
    -------
    pos_in: (K, S_out)
        The randomized input position index.

    """
    if connect_type=='shuffle':
        pos_in = np.stack([np.random.permutation(pos_in[:, i]) for i in range(pos_in.shape[1])], axis=1)
    elif connect_type=='scatter':
        pos_in = np.stack([np.random.permutation(pos_in[i]) for i in range(pos_in.shape[0])], axis=0)
    return pos_in

def random_pos_layer(
    pos_in: Array,
    connect_type: str,
    in_channels: int, out_channels: int,
    in_consistent: bool = True, out_consistent: bool = True,
) -> Array:
    r"""Returns randomized input positions for one layer.

    Multiple randomized versions are generated for each i/o channel combination.
    However if channel consistency is required for either input or output,
    duplicates are used.

    Args
    ----
    pos_in: (K, S_out)
        The input position index describing a normal convolution connectivity.
        See `random_pos_channel` for more information.
    connect_type:
        The sparse connection type, can be 'normal', 'shuffle' or 'scatter'. See
        `random_pos_channel` for more information.
    in_channels, out_channels:
        The number of input and output channels.
    in_consistent, out_consistent:
        Whether the spatial pattern is consistent across input or output
        channels.

    Returns
    -------
    pos_in: (out_channels, in_channels, K, S_out)
        The randomized input position index. `pos_in[c_out, c_in, k, pos_out]`
        is the input position index of the parameter `k` connectiing to output
        output position `pos_out` from channel `c_in` to channel `c_out`.

    """
    K, S_out = pos_in.shape
    pos_in = np.array([
        [
            random_pos_channel(pos_in, connect_type) for _ in range(in_channels if in_consistent else 1)
        ] for _ in range(out_channels if out_consistent else 1)
    ])
    pos_in = np.broadcast_to(pos_in, (out_channels, in_channels, K, S_out)).copy()

    return pos_in

def sparse_support(
    H_in: int, W_in: int,
    in_channels: int, out_channels: int,
    kernel_size: int, stride: int = 1, padding: int = 1,
    padding_mode: str = 'zeros',
    connect_type: str = 'normal',
    in_consistent: bool = True, out_consistent: bool = True,
) -> Tuple[int, int, Array, Tuple[int]]:
    r"""Constructs support for a sparse 2D convolution.

    The algorithm starts from a normal convolution connectivity and randomizes
    the input end of each connection. All valid connections are gathered by
    their input indices and output indices in the flattened tensor.

    Args
    ----
    H_in, W_in:
        The height and width of inputs.
    in_channels, out_channels:
        The number of input and output channels.
    kernel_size, stride, padding:
        The kernel size, stride and padding of a baseline convolution.
    padding_mode:
        The padding mode, can be 'zeros' or 'circular'.
    connect_type:
        The sparse connection type, can be 'normal', 'shuffle' or 'scatter'. See
        `random_pos_channel` for more information.
    in_consistent, out_consistent:
        Whether the spatial pattern is consistent across input or output
        channels.

    Returns
    -------
    H_out, W_out:
        The height and width of outputs.
    w_idxs: (3, *)
        Coordinate list of weight parameter index. Each column of `w_idxs` is
        `(pos_in, pos_out, k)`.
    w_size: tuple
        The size of sparse weight matrix, `(n_in, n_out)`.

    """
    H_out = (H_in+2*padding-(kernel_size-1)-1)//stride+1
    W_out = (W_in+2*padding-(kernel_size-1)-1)//stride+1
    n_in, n_out = in_channels*H_in*W_in, out_channels*H_out*W_out

    # prepare output positions
    c_out, c_in, param_idx, pos_out = np.meshgrid(
        np.arange(out_channels), np.arange(in_channels),
        np.arange(kernel_size**2), np.arange(H_out*W_out),
        indexing='ij',
    )
    param_idx += (c_out*in_channels+c_in)*(kernel_size**2)

    # prepare input positions for a normal convolution kernel
    y_out, x_out = np.unravel_index(np.arange(H_out*W_out), (H_out, W_out))
    dy, dx = np.unravel_index(np.arange(kernel_size**2), (kernel_size, kernel_size))
    y_in = y_out[None]*stride+dy[:, None]
    x_in = x_out[None]*stride+dx[:, None]
    pos_in = np.ravel_multi_index((y_in, x_in), (H_in+padding*2, W_in+padding*2))

    # create randomized input positions for all i/o channel combinations
    pos_in = random_pos_layer(
        pos_in, connect_type, in_channels, out_channels,
        in_consistent, out_consistent
    )

    # find valid indices
    y_in, x_in = np.unravel_index(pos_in, (H_in+2*padding, W_in+2*padding))
    y_in, x_in = y_in-padding, x_in-padding
    if padding_mode=='circular':
        y_in = y_in%H_in
        x_in = x_in%W_in
    valid_mask = np.all([y_in>=0, y_in<H_in, x_in>=0, x_in<W_in], axis=0)

    # compute unit indices for sparse weight matrix
    idx_in = np.ravel_multi_index(
        (c_in[valid_mask], y_in[valid_mask], x_in[valid_mask]),
        (in_channels, H_in, W_in)
    )
    y_out, x_out = np.unravel_index(pos_out, (H_out, W_out))
    idx_out = np.ravel_multi_index(
        (c_out[valid_mask], y_out[valid_mask], x_out[valid_mask]),
        (out_channels, H_out, W_out)
    )
    w_idxs = np.stack([idx_out, idx_in, param_idx[valid_mask]])
    w_size = (n_out, n_in)
    return H_out, W_out, w_idxs, w_size
