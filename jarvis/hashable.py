import numpy as np
from .utils import Array


def to_hashable(val):
    r"""Converts to hashable data type.

    """
    if isinstance(val, list):
        return HashableList(val)
    if isinstance(val, tuple):
        return HashableTuple(val)
    if isinstance(val, set):
        return HashableSet(val)
    if isinstance(val, dict):
        return HashableDict(val)
    if isinstance(val, Array):
        return HashableArray(val)

    try:
        hash(val)
    except:
        raise TypeError('hashable type is not implemented')
    else:
        return val


def _is_custom_hashable(val):
    r"""Returns whether the input is a custom hashable type.

    All custom hashable class implements `native` method.

    """
    return (
        isinstance(val, HashableList)
        or isinstance(val, HashableTuple)
        or isinstance(val, HashableSet)
        or isinstance(val, HashableDict)
        or isinstance(val, HashableArray)
        )


class HashableList(list):
    r"""Hashable list class.

    """

    def __init__(self, vals):
        super(HashableList, self).__init__([to_hashable(val) for val in vals])

    def __hash__(self):
        return hash(tuple(self))

    def native(self):
        converted = []
        for val in self:
            converted.append(val.native() if _is_custom_hashable(val) else val)
        return converted


class HashableTuple(tuple):
    r"""Hashable tuple class.

    """

    def __new__(cls, vals):
        return super(HashableTuple, cls).__new__(cls, tuple(to_hashable(val) for val in vals))

    def native(self):
        converted = []
        for val in self:
            converted.append(val.native() if _is_custom_hashable(val) else val)
        return tuple(converted)


class HashableSet(set):
    r"""Hashable set class.

    """

    def __init__(self, vals):
        super(HashableSet, self).__init__([to_hashable(val) for val in vals])

    def __hash__(self):
        return hash(frozenset(self))

    def native(self):
        converted = []
        for val in self:
            converted.append(val.native() if _is_custom_hashable(val) else val)
        return set(converted)


class HashableDict(dict):
    r"""Hashable dictionary class.

    """

    def __init__(self, vals):
        super(HashableDict, self).__init__((key, to_hashable(val)) for key, val in vals.items())

    def __hash__(self):
        return hash(frozenset(self.items()))

    def native(self):
        converted = {}
        for key, val in self.items():
            converted[key] = val.native() if _is_custom_hashable(val) else val
        return converted


class HashableArray(HashableList):

    def __init__(self, x: Array):
        self.shape = x.shape
        self.dtype = x.dtype
        vals = list(x.reshape(-1))
        super(HashableArray, self).__init__(vals)

    def native(self):
        return np.array(self, dtype=self.dtype).reshape(self.shape)
