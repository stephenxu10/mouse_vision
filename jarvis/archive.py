import os, pickle, random, time
from typing import Optional
from .hashable import to_hashable
from .utils import progress_str, time_str


class Archive:
    r"""Dictionary-like class that stores data externally."""
    _alphabet = ['{:X}'.format(i) for i in range(16)]

    def __init__(self,
        store_dir: Optional[str] = None,
        key_len: int = 8,
        path_len: int = 2,
        max_try: int = 30,
        pause: float = 0.5,
        hashable: bool = False,
    ):
        r"""
        Args
        ----
        store_dir: str
            The directory for storing data. If `store_dir` is ``None``, an
            internal dictionary `__store__` is used.
        key_len: int
            The length of keys.
        path_len: int
            The length of external file names, should be no greater than
            `key_len`.
        max_try: int
            The maximum number of trying to read/write external files.
        pause: float
            The time (seconds) between two consecutive read/write attempts.
        hashable: bool
            Whether the record value should be hashable. Useful when saving
            configuration dictionaries.

        """
        self.store_dir, self.key_len = store_dir, key_len
        if self.store_dir is None:
            self.__store__ = {}
        else:
            assert key_len>=path_len, "File name length should be no greater than key length."
            os.makedirs(self.store_dir, exist_ok=True)
            self.path_len, self.max_try, self.pause = path_len, max_try, pause
        self.hashable = hashable

    def __repr__(self):
        if self.store_dir is None:
            return 'Archive object with no external storage'
        else:
            return 'Archive object stored in {}'.format(self.store_dir)

    def __setitem__(self, key, val):
        if self.store_dir is None:
            self.__store__[key] = to_hashable(val) if self.hashable else val
        else:
            store_path = self._store_path(key)
            records = self._safe_read(store_path) if os.path.exists(store_path) else {}
            records[key] = to_hashable(val) if self.hashable else val
            self._safe_write(records, store_path)

    def __getitem__(self, key):
        _no_exist_msg = f"{key} does not exist."
        if self.store_dir is None:
            assert key in self.__store__, _no_exist_msg
            return self.__store__[key]
        else:
            store_path = self._store_path(key)
            assert os.path.exists(store_path), _no_exist_msg
            records = self._safe_read(store_path)
            assert key in records, _no_exist_msg
            return records[key]

    def __contains__(self, key):
        if self.store_dir is None:
            return key in self.__store__
        else:
            try:
                store_path = self._store_path(key)
            except:
                return False # invalid key
            if not os.path.exists(store_path):
                return False
            records = self._safe_read(store_path)
            return key in records

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(list(self.keys()))

    def _is_valid_key(self, key):
        r"""Returns if a key is valid."""
        if not (isinstance(key, str) and len(key)==self.key_len):
            return False
        for c in key:
            if c not in Archive._alphabet:
                return False
        return True

    def _store_path(self, key):
        r"""Returns the path of external file associated with a key."""
        assert self._is_valid_key(key), f"Invalid key '{key}' encountered."
        return '{}/{}.axv'.format(self.store_dir, key[:self.path_len])

    def _store_paths(self):
        r"""Returns all valid external files in the directory."""
        return [
            f'{self.store_dir}/{f}' for f in os.listdir(self.store_dir)
            if f.endswith('.axv') and len(f)==(self.path_len+4)
            ]

    def _safe_read(self, store_path):
        r"""Safely reads a file.

        This is designed for cluster use. An error is raised when too many
        attempts have failed.

        """
        count = 0
        while count<self.max_try:
            try:
                with open(store_path, 'rb') as f:
                    records = pickle.load(f)
            except:
                count += 1
                time.sleep(self.pause*(0.8+random.random()*0.4))
            else:
                break
        if count==self.max_try:
            raise RuntimeError(f"Max number ({count}) of reading tried and failed.")
        return records

    def _safe_write(self, records, store_path):
        r"""Safely writes a file."""
        count = 0
        while count<self.max_try:
            try:
                with open(store_path, 'wb') as f:
                    pickle.dump(records, f)
            except:
                count += 1
                time.sleep(self.pause*(0.8+random.random()*0.4))
            else:
                break
        if count==self.max_try:
            raise RuntimeError(f"Max number ({count}) of writing tried and failed.")

    def _random_key(self):
        r"""Returns a random key."""
        return ''.join(random.choices(Archive._alphabet, k=self.key_len))

    def _new_key(self):
        r"""Returns a new key."""
        while True:
            key = self._random_key()
            if key not in self:
                break
        return key

    def keys(self):
        r"""A generator for keys."""
        if self.store_dir is None:
            for key in self.__store__.keys():
                yield key
        else:
            store_paths = self._store_paths()
            random.shuffle(store_paths) # randomization to avoid cluster conflict
            for store_path in store_paths:
                records = self._safe_read(store_path)
                for key in records.keys():
                    yield key

    def values(self):
        r"""A generator for values."""
        if self.store_dir is None:
            for val in self.__store__.values():
                yield val
        else:
            store_paths = self._store_paths()
            random.shuffle(store_paths)
            for store_path in store_paths:
                records = self._safe_read(store_path)
                for val in records.values():
                    yield val

    def items(self):
        r"""A generator for items."""
        if self.store_dir is None:
            for key, val in self.__store__.items():
                yield key, val
        else:
            store_paths = self._store_paths()
            random.shuffle(store_paths)
            for store_path in store_paths:
                records = self._safe_read(store_path)
                for key, val in records.items():
                    yield key, val

    def get_key(self, val):
        r"""Returns the key of a record.

        Returns
        -------
        key: Optional[str]
            The key of record being searched for. ``None`` if not found.

        """
        assert self.hashable, "Key search is implemented for hashable records only."
        h_val = to_hashable(val)
        for key, val in self.items():
            if val==h_val:
                return key
        return None

    def add(self, val):
        r"""Adds a new item if it has not already existed.

        Returns
        -------
        key: str
            The key of added record.

        """
        assert self.hashable, "Value addition is implemented for hashable records only."
        key = self.get_key(val)
        if key is None:
            key = self._new_key()
            self[key] = val
        return key

    def pop(self, key):
        r"""Pops out an item by key."""
        if self.store_dir is None:
            return self.__store__.pop(key, None)
        else:
            store_path = self._store_path(key)
            if not os.path.exists(store_path):
                return None
            records = self._safe_read(store_path)
            val = records.pop(key, None)
            if records:
                self._safe_write(records, store_path)
            else:
                os.remove(store_path) # remove empty external file
            return val

    def prune(self):
        r"""Removes corrupted files.

        Returns
        -------
        removed: list[str]
            The name of removed files.

        """
        removed = []
        if self.store_dir is None:
            print("No external store detected.")
        else:
            store_paths = self._store_paths()
            verbose, tic = None, time.time()
            for i, store_path in enumerate(store_paths, 1):
                try:
                    self._safe_read(store_path)
                except:
                    print("{} corrupted, will be removed".format(store_path))
                    os.remove(store_path)
                    removed.append(store_path[(-4-self.path_len):-4])
                if i%(-(-len(store_paths)//10))==0 or i==len(store_paths):
                    toc = time.time()
                    if verbose is None:
                        # display progress if estimated time is longer than 20 mins
                        verbose = (toc-tic)/i*len(store_paths)>1200
                    if verbose:
                        print("{} ({})".format(
                            progress_str(i, len(store_paths)),
                            time_str(toc-tic, progress=i/len(store_paths))
                            ))
            if removed:
                print("{} corrupted files removed.".format(len(removed)))
            else:
                print("No corrupted files detected.")
        return removed

    def get_duplicates(self):
        r"""Returns all duplicate records.

        Returns
        -------
        duplicates: dict[Hashable, list[str]]
            A dictionary of which the key is the duplicate record value, while
            the value is the list of keys associated with it.

        """
        assert self.hashable, "Duplicate detection is implemented for hashable records only."
        inv_dict = {} # dictionary for inverse archive
        for key, val in self.items():
            if val in inv_dict:
                inv_dict[val].append(key)
            else:
                inv_dict[val] = [key]
        duplicates = dict((val, keys) for val, keys in inv_dict.items() if len(keys)>1)
        return duplicates

    def to_internal(self):
        r"""Moves external storage to internal."""
        if self.store_dir is not None:
            self.__store__ =  dict((key, val) for key, val in self.items())
            self.store_dir = None
