import random, time
import numpy as np
from collections.abc import Iterable
from typing import Optional, List
from .utils import time_str, flatten, nest
from .archive import Archive
from .hashable import HashableDict


class BaseJob:
    r"""Base class for batch processing.

    The job is associated with different directories storing configurations,
    status, ckpts and previews of all works. Method `main` need to be
    implemented by child class.

    """

    def __init__(self,
        store_dir: Optional[str] = None,
        read_only: bool = False,
        s_path_len: int = 2, s_pause: float = 1.,
        l_path_len: int = 3, l_pause: float = 5.,
    ):
        r"""
        Args
        ----
        store_dir:
            Directory for storage. Archives including `configs`, `stats`,
            `ckpts` and `previews` will be saved in separate directories.
        read_only:
            If the job is read-only or not.
        s_path_len, s_pause:
            Short path length and pause time for `configs`, as well as `stats`
            and `previews`.
        l_path_len, l_pause:
            Long path length and pause time for `ckpts`.

        """
        self.store_dir = store_dir
        if self.store_dir is None:
            self.configs = Archive(hashable=True)
            self.stats = Archive()
            self.ckpts = Archive()
            self.previews = Archive()
        else:
            self.configs = Archive(
                f'{self.store_dir}/configs', path_len=s_path_len, pause=s_pause, hashable=True,
            )
            self.stats = Archive(
                f'{self.store_dir}/stats', path_len=s_path_len, pause=s_pause,
            )
            self.ckpts = Archive(
                f'{self.store_dir}/ckpts', path_len=l_path_len, pause=l_pause,
            )
            self.previews = Archive(
                f'{self.store_dir}/previews', path_len=s_path_len, pause=s_pause,
            )
        self.read_only = read_only
        if self.store_dir is not None and self.read_only:
            for axv in [self.configs, self.stats, self.previews]:
                axv.to_internal()

    def main(self,
        config: dict,
        num_epochs: int = 1,
        resume: bool = True,
        verbose: int = 1,
    ):
        r"""Main function that needs implementation by subclasses.

        Args
        ----
        config:
            A configuration dict for the work.
        num_epochs:
            Number of epochs of the work. If not explicit epochs can be defined,
            use `num_epochs=1` for a single pass.
        resume:
            Whether to resume from existing checkpoint. If ``False``, compute
            from scratch.
        verbose:
            Level of information display. No message will be printed when
            'verbose' is no greater than 0.

        Returns
        -------
        ckpt, preview:
            Archive records that will be saved in `self.ckpts` and
            `self.previews` respectively.

        """
        raise NotImplementedError

    def process(self,
        config: HashableDict,
        num_epochs: int = 1,
        resume: bool = True,
        verbose: int = 1,
    ):
        key = self.configs.add(config)
        if verbose>0:
            print(f"Processing {key}...")
        try:
            assert resume
            epoch, ckpt, preview = self.load_ckpt(config)

            assert epoch>=num_epochs
            if verbose>0:
                print("Checkpoint{} loaded.".format(
                    '' if epoch==1 else f' (epoch {epoch})',
                ))
        except:
            tic = time.time()
            ckpt, preview = self.main(config, num_epochs, resume, verbose)
            self.save_ckpt(config, num_epochs, ckpt, preview)
            toc = time.time()
            if verbose>0:
                print("Checkpoint{} saved. ({})".format(
                    '' if num_epochs==1 else f' (epoch {num_epochs})', time_str(toc-tic),
                ))
        return ckpt, preview

    def batch(self,
        configs: Iterable,
        num_epochs: int = 1,
        num_works: int = 0,
        patience: float = 168,
        max_errors: int = 0,
        verbose: int = 1,
    ):
        r"""Batch processing.

        Args
        ----
        configs:
            An iterable object containing work configurations. Some are
            potentially processed already.
        num_works:
            The number of works to process. If it is 0, the processing stops
            when no work is left in `configs`.
        patience:
            Patience time for processing an incomplete work, in hours. The last
            modified time of a work is recorded in `self.stats`.

        """
        w_count, e_count, interrupted = 0, 0, False
        for config in configs:
            try:
                key = self.configs.add(config)
                try:
                    stat = self.stats[key]
                except:
                    stat = {'epoch': 0, 'toc': -float('inf')}
                if stat['epoch']>=num_epochs or (time.time()-stat['toc'])/3600<patience:
                    continue
                stat['toc'] = time.time()
                self.stats[key] = stat

                if verbose>0:
                    print("------------")
                self.process(config, num_epochs, verbose=verbose)
                w_count += 1
                if verbose>0:
                    print("------------")
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as e:
                if max_errors==0:
                    raise e
                e_count += 1
                if e_count==max_errors:
                    interrupted = True
                    break
                else:
                    continue
            else:
                if num_works>0 and w_count==num_works:
                    break
        if verbose>0:
            print("\n{} works processed.".format(w_count))
            if not interrupted and (num_works==0 or w_count<num_works):
                print("All works are processed or being processed.")

    def strs2config(self, arg_strs: Optional[List[str]] = None):
        r"""Returns work configuration.

        Args
        ----
        arg_strs:
            A string list that can be parsed to argument parser.

        Returns
        -------
        config: dict
            A dictionary specified by `arg_strs`.

        """
        raise NotImplementedError

    def get_config(self, config: Optional[dict] = None) -> dict:
        r"""Returns work configuration.

        This method fills in default values of necessary keys, and check the
        consistency of values if necessary. Needs to be implemented by child
        class.

        Args
        ----
        config:
            Potentially partial configuration for a work.

        Returns
        -------
        config:
            Full configuration for `main` method.

        """

    def grid_search(self, search_spec: dict, **kwargs):
        r"""Grid hyper-parameter search.

        Work configurations are constructed randomly from `search_spec`, using
        the method `get_config`. The configuration generator is passed to the
        method `batch`.

        search_spec:
            The work configuration search specification, can be nested. It has
            the same key structure and a valid `config` for `main` method, and
            values at the leaf level are lists containing possible values of a
            `config`.

        """
        f_spec = flatten(search_spec)
        f_keys = list(f_spec.keys())
        val_lists = [f_spec[key] for key in f_keys]
        space_dim = [len(v) for v in val_lists]
        total_num = np.prod(space_dim)

        def config_gen():
            for idx in random.sample(range(total_num), total_num):
                sub_idxs = np.unravel_index(idx, space_dim)
                f_config = {}
                for i, f_key in enumerate(f_keys):
                    f_config[f_key] = val_lists[i][sub_idxs[i]]
                config = self.get_config(nest(f_config))
                yield config
        self.batch(config_gen(), **kwargs)

    def load_ckpt(self, config):
        r"""Loads checkpoint."""
        key = self.configs.add(config)
        epoch = self.stats[key]['epoch']
        ckpt = self.ckpts[key]
        preview = self.previews[key]
        return epoch, ckpt, preview

    def save_ckpt(self, config, epoch, ckpt, preview):
        r"""Saves checkpoint."""
        key = self.configs.add(config)
        self.stats[key] = {'epoch': epoch, 'toc': time.time()}
        self.ckpts[key] = ckpt
        self.previews[key] = preview

    @staticmethod
    def _is_matched(config, cond=None):
        r"""Checks if a configuration matches condition."""
        if cond is None:
            return True
        flat_config, flat_cond = flatten(config), flatten(cond)
        for key in flat_cond:
            if not(key in flat_config and flat_config[key]==flat_cond[key]):
                return False
        return True

    def completed(self, min_epoch=1, cond=None):
        r"""A generator for completed works."""
        for key, stat in self.stats.items():
            if stat['epoch']>=min_epoch:
                try:
                    config = self.configs[key]
                except:
                    continue
                if self._is_matched(config, cond):
                    yield key

    def best_work(self,
        min_epochs: int = 1,
        cond: Optional[dict] = None,
        p_key: str = 'loss_test',
        reverse: Optional[bool] = None,
        verbose: int = 1,
    ):
        r"""Returns the best work given conditions.

        Args
        ----
        min_epochs:
            Minimum number of epochs.
        cond:
            Conditioned value of work configurations. Only completed work with
            matching values will be considered.
        p_key:
            The key of `preview` for comparing works.
        reverse:
            Returns work with the largest value of ``'p_key'`` when `reverse` is
            ``True``, otherwise the smallest.

        Returns
        -------
        best_key: str
            The key of best work.

        """
        assert min_epochs>0
        if reverse is None:
            reverse = p_key.startswith('acc')
        best_val = -float('inf') if reverse else float('inf')
        best_key = None
        count = 0
        for key in self.completed(min_epochs, cond):
            val = self.previews[key][p_key]
            if reverse:
                if val>best_val:
                    best_val = val
                    best_key = key
            else:
                if val<best_val:
                    best_val = val
                    best_key = key
            count += 1
        if verbose>0:
            if min_epochs==1:
                print(f"{count} completed works found.")
            else:
                print(f"{count} works trained with at least {min_epochs} epochs found.")
        return best_key

    def pop(self, key):
        r"""Pops out a work by key."""
        config = self.configs.pop(key)
        stat = self.stats.pop(key)
        ckpt = self.ckpts.pop(key)
        preview = self.previews.pop(key)
        return config, stat, ckpt, preview
