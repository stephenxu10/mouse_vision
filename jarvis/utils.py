import argparse, random, torch
import numpy as np

Array = np.ndarray


def time_str(t_elapse, progress=1.):
    r"""Returns a formatted string for a duration.

    Args
    ----
    t_elapse: float
        The elapsed time in seconds.
    progress: float
        The estimated progress, used for estimating field width.

    """
    field_width = int(np.log10(max(t_elapse, 1e-6)/60/progress))+1
    return '{{:{}d}}m{{:05.2f}}s'.format(field_width).format(int(t_elapse//60), t_elapse%60)


def progress_str(i, total, show_percent=False):
    r"""Returns a formatted string for progress.

    Args
    ----
    i： int
        The current iteration index.
    total: int
        The total iteration number.
    show_percent: bool
        Whether to show percentage or not.

    """
    field_width = int(np.log10(total))+1
    disp_str = '{{:{}d}}/{{:{}d}}'.format(field_width, field_width).format(i, total)
    if show_percent:
        disp_str += ', ({:6.1%})'.format(i/total)
    return disp_str


def get_seed(seed=None, max_seed=1000):
    r"""Returns a random seed."""
    if seed is None:
        return random.randrange(max_seed)
    else:
        return seed%max_seed


def set_seed(seed, strict=False):
    r"""Sets random seed for random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def flatten(nested_dict):
    r"""Flattens a nested dictionary.

    A nested dictionary like `{'A': {'B', val}}` will be converted to
    `{('B', '@', 'A'), val}`.

    Args
    ----
    nested_dict: dict
        A nested dictionary possibly contains dictionaries as values.

    Returns
    -------
    flat_dict: dict
        A flat dictionary with tuple keys for hierarchy.

    """
    flat_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict) and len(val)>0:
            for subkey, subval in flatten(val).items():
                flat_dict[(subkey, '@', key)] = subval
        else:
            flat_dict[key] = val
    return flat_dict


def nest(flat_dict):
    r"""Nests a flat dictionary.

    A flat dictionary like `{('B', '@', 'A'), val}` will be converted to
    `{'A': {'B', val}}`.

    Args
    ----
    flat_dict: dict
        A flat dictionary with tuple keys for hierarchy.

    Returns
    -------
    nested_dict: dict
        A nested dictionary possibly contains dictionaries as values.

    """
    nested_dict = {}
    for key, val in flat_dict.items():
        if isinstance(key, tuple) and len(key)==3 and key[1]=='@':
            subkey, _, parkey = key
            if parkey not in nested_dict:
                nested_dict[parkey] = {}
            # nested_dict[parkey].update(nest({subkey: val}))
            nested_dict[parkey][subkey] = val
        else:
            nested_dict[key] = val
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            nested_dict[key] = nest(val)
    return nested_dict


def fill_defaults(spec, d_spec):
    r"""Fills in default values of specification dictionary."""
    f_spec = flatten(spec)
    for key, val in flatten(d_spec).items():
        if key not in f_spec:
            f_spec[key] = val
    return nest(f_spec)


def numpy_dict(state):
    r"""Returns a state dictionary with tensors replaced by numpy arrays.

    Each tensor is converted to a tuple containing the numpy array and tensor
    dtype.

    Args
    ----
    state: dict
        State dictionary potentially containing tensors, returned by torch
        module, optimizer or scheduler.

    Returns
    -------
    A dictionary with same structure, with tensors converted to numpy arrays.

    """
    f_state = flatten(state)
    for key, val in f_state.items():
        if isinstance(val, torch.Tensor):
            f_state[key] = (val.data.cpu().clone().numpy(), val.dtype)
    return nest(f_state)


def tensor_dict(state, device='cpu'):
    r"""Returns a state dictionary with numpy arrays replaced by tensors.

    This is the inverted function of `numpy_dict`.

    Args
    ----
    state: dict
        The state dictionary converted by `numpy_dict`.
    device:
        Tensor device of the converted state dictionary.

    """
    f_state = flatten(state)
    for key, val in f_state.items():
        if isinstance(val, tuple) and len(val)==2 and isinstance(val[0], np.ndarray) and isinstance(val[1], torch.dtype):
            f_state[key] = torch.tensor(val[0], dtype=val[1], device=device)
    return nest(f_state)


def job_parser():
    r"""Returns a base parser for job processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-wait', default=1, type=float,
                        help="seconds of wait before each job")
    parser.add_argument('--num-works', default=0, type=int,
                        help="number of works to process")
    parser.add_argument('--patience', default=168, type=float,
                        help="hours since last modification")
    return parser


def sgd_optimizer(model, lr, momentum=0.9, weight_decay=0.):
    r"""Returns a SGD optimizer.

    Only parameters whose name ends with ``'weight'`` will be trained with
    weight decay.

    Args
    ----
    model: nn.Module
        The pytorch model.
    lr: float
        Learning rate for all parameters.
    momentum: float
        The momentum parameter for SGD.
    weight_decay: float
        The weight decay parameter for layer weights but not biases.

    Returns
    -------
    optimizer: optimizer
        The SGD optimizer.

    """
    params = []
    params.append({
        'params': [param for name, param in model.named_parameters() if name.endswith('weight')],
        'weight_decay': weight_decay,
        })
    params.append({
        'params': [param for name, param in model.named_parameters() if not name.endswith('weight')],
        })
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    return optimizer


def cyclic_scheduler(optimizer, phase_len=4, num_phases=3, gamma=0.3):
    r"""Returns a simple cyclic scheduler.

    Learning rate is scheduled to follow cycles, each of which contains a fixed
    number of phases. At the beginning of each cycle the learning rate is reset
    to the initial value.

    Args
    ----
    optimizer: optimizer
        The pytorch optimizer.
    phase_len:
        The length of each phase, during which the learning rate is fixed.
    num_phases: int
        The number of phasese within each cycle. Learning rate decays by a
        fixed factor between phases.
    gamma: float
        The decay factor between phases, must be in `(0, 1]`.

    Returns
    -------
    scheduler: scheduler
        The cyclic scheculer.

    """
    cycle_len = phase_len*num_phases
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: gamma**((epoch%cycle_len)//phase_len)
        )
    return scheduler
