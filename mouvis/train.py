import os, json, pickle, time
import numpy as np
import torch
from typing import Optional, Tuple, Dict, List

from jarvis import BaseJob
from jarvis.utils import (
    job_parser, fill_defaults, cyclic_scheduler,
    time_str, progress_str, numpy_dict, tensor_dict,
)

from .models import DenseCore, Modulator, Shifter, NeuralModel
from .utils import ResponseLoss, response_corrs, oracle_frac

Tensor = torch.Tensor
Dataset = torch.utils.data.Dataset

DEVICE = 'cuda'
NUM_WORKERS = 4
EVAL_BATCH_SIZE = 16
TRAIN_NUM_INFOS = 6
EVAL_INTERVAL = 5
SAVE_INTERVAL = 20
NUM_EPOCHS = 240


class NeuralTrainJob(BaseJob):
    r"""Job for training neural predictive models."""

    def __init__(self,
        store_dir: str,
        device: str = DEVICE,
        num_workers: int = NUM_WORKERS,
        eval_batch_size: int = EVAL_BATCH_SIZE,
        train_num_infos: int = TRAIN_NUM_INFOS,
        eval_interval: int = EVAL_INTERVAL,
        save_interval: int = SAVE_INTERVAL,
        **kwargs,
    ):
        super(NeuralTrainJob, self).__init__(store_dir=f'{store_dir}/models', **kwargs)
        self.data_dir = f'{store_dir}/data'
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_workers = num_workers
        self.eval_batch_size = eval_batch_size
        self.train_num_infos = train_num_infos
        assert eval_interval>0
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        with open(f'{store_dir}/jsons/data.json', 'r') as f:
            self.d_defaults = json.load(f)
        with open(f'{store_dir}/jsons/model.json', 'r') as f:
            self.m_defaults = json.load(f)
        with open(f'{store_dir}/jsons/train.json', 'r') as f:
            self.t_defaults = json.load(f)
        with open(f'{store_dir}/jsons/reg.json', 'r') as f:
            self.r_defaults = json.load(f)

    def get_config(self, config=None) -> dict:
        r"""Implements parent method."""
        config = fill_defaults(
            config or {}, {
                'data_config': self.d_defaults,
                'model_config': self.m_defaults,
                'train_config': self.t_defaults,
                'reg_config': self.r_defaults,
            }
        )
        return config

    def main(self, config, num_epochs=NUM_EPOCHS, resume=True, verbose=1):
        data_config = config['data_config']
        model_config = config['model_config']
        train_config = config['train_config']
        reg_config = config['reg_config']
        if verbose>0:
            print(f"data_config:\n{data_config}")
            print(f"model_config:\n{model_config}")
            print(f"train_config:\n{train_config}")
            print(f"reg_config:\n{reg_config}")
            print("Training a neural predictive model...")

        # prepare datasets
        _, _, scaled_means, loo_corrs = self.meta_info(data_config)
        dsets, neuron_num = self.prepare_datasets(
            data_config, train_config,
        )

        # prepare models
        if train_config['preprocess']:
            images, behaviors, pupil_centers, _ = dsets['train'][:]
            i_transform = {
                'shift': images.mean(axis=(0, 2, 3)),
                'scale': images.std(axis=(0, 2, 3)),
            }
            b_transform = {
                'shift': behaviors.mean(axis=0),
                'scale': behaviors.std(axis=0),
            }
            p_transform = {
                'shift': pupil_centers.mean(axis=0),
                'scale': pupil_centers.std(axis=0),
            }
        else:
            i_transform, b_transform, p_transform = None, None, None
        model = self.prepare_model(neuron_num, model_config, i_transform, b_transform, p_transform)
        _, behaviors, pupil_centers, _ = next(iter(torch.utils.data.DataLoader(
            dsets['train'], batch_size=model.bank_size, shuffle=True,
        )))
        model = model.cuda()
        model.sampled_behaviors.data = behaviors
        model.sampled_pupil_centers.data = pupil_centers
        if verbose>0:
            print("\nModel randomly initialized")

        # prepare optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = cyclic_scheduler(
            optimizer, phase_len=train_config['phase_len'],
            num_phases=train_config['num_phases'],
            gamma=train_config['decay_rate'],
        )

        # define objective function
        if train_config['neuron_weight']=='none':
            criterion = ResponseLoss()
        if train_config['neuron_weight']=='s_mean':
            criterion = ResponseLoss(neuron_weights=scaled_means)
        if train_config['neuron_weight']=='l_corr':
            criterion = ResponseLoss(neuron_weights=loo_corrs.clamp_min(0.01))

        # load checkpoint or initialize one
        try:
            assert resume
            epoch, ckpt, preview = self.load_ckpt(config)
            model.load_state_dict(tensor_dict(ckpt['model_state']))
            optimizer.load_state_dict(tensor_dict(ckpt['optimizer_state']))
            eval_records = ckpt['eval_records']
            best_state, best_epoch = ckpt['best_state'], ckpt['best_epoch']
            min_loss = ckpt['min_loss']
            if verbose>0:
                print(f"Checkpoint (epoch {epoch}) loaded.")
        except:
            epoch = 0
            ckpt = {
                'model_state': numpy_dict(model.state_dict()),
                'optimizer_state': numpy_dict(optimizer.state_dict()),
            }
            eval_records = {'val': {}, 'test': {}}
            for key in eval_records:
                if verbose>0:
                    if key=='val':
                        print("Validation set")
                    if key=='test':
                        print("Testing set")
                outputs, targets = self.dataset_forward(dsets[key], model)
                loss = criterion.cpu()(outputs, targets).item()
                if verbose>0:
                    print("Loss {:7.4f}".format(loss))
                if key=='test':
                    pred_corrs = response_corrs(outputs.numpy(), targets.numpy())
                    o_frac = oracle_frac(loo_corrs.data.cpu().numpy(), pred_corrs, criterion.neuron_weights)
                    if verbose>0:
                        print('oracle fraction: {:.3f}'.format(o_frac))
                    eval_records[key][epoch] = (loss, o_frac)
                else:
                    eval_records[key][epoch] = (loss,)
            best_state, best_epoch = ckpt['model_state'], epoch
            min_loss, = eval_records['val'][epoch]
            ckpt.update({
                'eval_records': eval_records,
                'best_state': best_state, 'best_epoch': best_epoch,
                'min_loss': min_loss,
            })
            preview = {
                'eval_records': ckpt['eval_records'], 'best_epoch': ckpt['best_epoch'],
            }
            self.save_ckpt(config, epoch, ckpt, preview)

        # main training loop
        while epoch<num_epochs:
            epoch += 1
            if verbose>0:
                print("Epoch {}".format(progress_str(epoch, num_epochs)))
            tic = time.time()
            self.train(
                dsets['train'], model, optimizer, criterion, train_config, reg_config, verbose,
            )
            scheduler.step()
            ckpt.update({
                'model_state': numpy_dict(model.state_dict()),
                'optimizer_state': numpy_dict(optimizer.state_dict()),
            })
            toc = time.time()
            if verbose>0:
                print("Elapsed time {}".format(time_str(toc-tic)))

            if epoch%self.eval_interval==0 or epoch==num_epochs:
                for key in eval_records:
                    if verbose>0:
                        if key=='val':
                            print("Validation set")
                        if key=='test':
                            print("Testing set")
                    outputs, targets = self.dataset_forward(dsets[key], model)
                    loss = criterion.cpu()(outputs, targets).item()
                    if verbose>0:
                        print("Loss {:7.4f}".format(loss))
                    if key=='test':
                        pred_corrs = response_corrs(outputs.numpy(), targets.numpy())
                        o_frac = oracle_frac(loo_corrs.data.cpu().numpy(), pred_corrs, criterion.neuron_weights)
                        if verbose>0:
                            print('oracle fraction: {:.3f}'.format(o_frac))
                        eval_records[key][epoch] = (loss, o_frac)
                    else:
                        eval_records[key][epoch] = (loss,)
                if eval_records['val'][epoch][0]<min_loss:
                    best_state, best_epoch = ckpt['model_state'], epoch
                    min_loss, = eval_records['val'][epoch]
                    ckpt.update({
                        'eval_records': eval_records,
                        'best_state': best_state, 'best_epoch': best_epoch,
                        'min_loss': min_loss,
                    })
                    preview = {
                        'eval_records': ckpt['eval_records'], 'best_epoch': ckpt['best_epoch'],
                    }
                    if verbose>0:
                        print(f"Better model obtained at epoch {epoch}")
            if self.save_interval>0 and epoch%self.save_interval==0:
                self.save_ckpt(config, epoch, ckpt, preview)
                if verbose>0:
                    print(f"Checkpoint (epoch {epoch}) saved.")
        return ckpt, preview


    def meta_info(self,
        data_config: dict,
    ) -> Tuple[List[int], Tensor, Tensor, Tensor]:
        r"""Returns meta information of a scan.

        Args
        ----
        data_config:
            The data configuration dictionary.

        Returns
        -------
        oracle_nums: list
            The number of oracle trials for each oracle image.
        sn_ratios: (neuron_num,)
            The signal-to-noise ratio of all neurons.
        scaled_means: (neuron_num,)
            The scaled mean response of all neurons.
        loo_corrs: (neuron_num,)
            The leave-one-out correlation coefficient of all neurons.

        """
        scan, area = data_config['scan'], data_config['area']
        PATH = "store/data/splitted/20505-10-14_basic.pickle"
        with open(PATH, 'rb') as f:
            saved = pickle.load(f)

        oracle_nums = saved['oracle_nums']
        sn_ratios = torch.tensor(saved['sn_ratios'][area], dtype=torch.float)
        scaled_means = torch.tensor(saved['scaled_means'][area], dtype=torch.float)
        loo_corrs = torch.tensor(saved['loo_corrs'][area], dtype=torch.float)
        return oracle_nums, sn_ratios, scaled_means, loo_corrs

    def prepare_datasets(self,
        data_config: dict,
        train_config: Optional[dict] = None,
    ) -> Tuple[Dict[str, Dataset], int]:
        r"""Prepares datasets from scan data.

        Args
        ----
        data_config:
            The data configuration dictionary.
        train_config:
            The training configuration dictionary. Only testing set is returned
            when set to ``None``.

        Returns
        -------
        dataset_train, dataset_valid: TensorDataset
            The training and validation set, images, behaviors, pupil centers and
            neural resposnes of normal trials. Only returned when `train_config` is
            not ``None``.
        dataset_test: TensorDataset
            The testing set, containing data of oracle trials.
        neuron_num: int
            The numer of neurons.

        """
        scan, area = data_config['scan'], data_config['area']

        PATH = "store/data/splitted/20505-10-14_basic.pickle"
        with open(PATH, 'rb') as f:
            saved = pickle.load(f)

        oracle_nums, oracle_ids = saved['oracle_nums'], saved['oracle_ids']
        normal_ids = saved['normal_ids']

        PATH = "store/data/images/imagenet_examples.pickle"
        with open(PATH, 'rb') as f:
            image_dict = pickle.load(f)

        oracle_images = np.concatenate([
            np.stack([image_dict[oracle_id]]*oracle_num) for oracle_num, oracle_id in zip(oracle_nums, oracle_ids)
        ])/255.
        normal_images = np.stack([
            image_dict[normal_id] for normal_id in normal_ids
        ])/255.

        oracle_behaviors = saved['behaviors']['oracle']
        normal_behaviors = saved['behaviors']['normal']

        oracle_pupil_centers = saved['pupil_centers']['oracle']
        normal_pupil_centers = saved['pupil_centers']['normal']

        neuron_num = saved['neuron_nums'][area]

        PATH = "store/data/splitted/20505-10-14_V1.pickle"
        with open(PATH, 'rb') as f:
            saved = pickle.load(f)

        oracle_responses = saved['oracle']
        normal_responses = saved['normal']
        oracle_responses[oracle_responses<data_config['eps']] = data_config['eps']
        normal_responses[normal_responses<data_config['eps']] = data_config['eps']

        dset_test = torch.utils.data.TensorDataset(
            torch.tensor(oracle_images, dtype=torch.float),
            torch.tensor(oracle_behaviors, dtype=torch.float),
            torch.tensor(oracle_pupil_centers, dtype=torch.float),
            torch.tensor(oracle_responses, dtype=torch.float),
        )
        if train_config is None:
            dset_train, dset_val = None, None
        else:
            valid_num = int(0.05*len(normal_ids))
            train_num = len(normal_ids)-valid_num
            dset_train, dset_val = torch.utils.data.random_split(
                torch.utils.data.TensorDataset(
                    torch.tensor(normal_images, dtype=torch.float),
                    torch.tensor(normal_behaviors, dtype=torch.float),
                    torch.tensor(normal_pupil_centers, dtype=torch.float),
                    torch.tensor(normal_responses, dtype=torch.float),
                ),
                [train_num, valid_num],
            )
        dsets = {
            'train': dset_train,
            'val': dset_val,
            'test': dset_test,
        }
        return dsets, neuron_num

    def prepare_model(self,
        neuron_num: int,
        model_config: dict,
        i_transform: Optional[dict] = None,
        b_transform: Optional[dict] = None,
        p_transform: Optional[dict] = None,
    ) -> NeuralModel:
        r"""Prepares a neural predictive model.

        Args
        ----
        neuron_num:
            Number of neurons.
        model_config:
            The model configuration dictionary.
        i_transform, b_trainsform, p_transform:
            Input transformations for images, behaviors and pupil centers. See
            `.models` for more details.

        Returns
        -------
        model: NeuralModel
            A neural model with specified configuration.

        """
        core = DenseCore(**model_config['core_config'], i_transform=i_transform)
        modulator = Modulator(neuron_num, **model_config['modulator_config'], b_transform=b_transform)
        shifter = Shifter(**model_config['shifter_config'], p_transform=p_transform)
        model = NeuralModel(
            core, modulator, shifter,
            patch_size=model_config['patch_size'],
            bank_size=model_config['bank_size'],
        )
        return model

    def dataset_forward(self,
        dataset: Dataset,
        model: NeuralModel,
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns forward pass on one dataset.

        Args
        ----
        dataset:
            The neural scan dataset.
        model:
            The neural predictive model.

        Returns
        -------
        outputs, targets: (trial_num, neuron_num)
            The model outputs and the real neural responses.

        """
        device = self.device
        model.eval().to(device)

        loader = torch.utils.data.DataLoader(dataset, self.eval_batch_size)
        _outputs, _targets = [], []
        for images, behaviors, pupil_centers, targets in loader:
            with torch.no_grad():
                outputs = model(
                    images.to(device), behaviors.to(device), pupil_centers.to(device),
                ).cpu()
            _outputs.append(outputs)
            _targets.append(targets)
        outputs = torch.cat(_outputs)
        targets = torch.cat(_targets)
        return outputs, targets

    def train(self,
        dataset, model, optimizer, criterion, train_config, reg_config, verbose,
    ):

        device = self.device
        model.train().to(device)
        criterion = criterion.to(device)

        betas = [
            np.exp(reg_config['log_beta_core_laplace']),
            np.exp(reg_config['log_beta_core_weight']),
            np.exp(reg_config['log_beta_modulator_weight']),
            np.exp(reg_config['log_beta_shifter_weight']),
            np.exp(reg_config['log_beta_readout_weight']),
            np.exp(reg_config['log_beta_readout_loc']),
        ] # coefficients of regularization losses

        loader = torch.utils.data.DataLoader(dataset, train_config['batch_size'], shuffle=True, drop_last=True)
        batch_num = len(loader)
        for batch_idx, (images, behaviors, pupil_centers, targets) in enumerate(loader, 1):
            outputs = model(images.to(device), behaviors.to(device), pupil_centers.to(device))
            fit_loss = criterion(outputs, targets.to(device))
            reg_loss =  torch.stack([beta*l for beta, l in zip(
                betas, [
                    model.core.laplace_reg(), model.core.weight_reg(),
                    model.modulator.weight_reg(), model.shifter.weight_reg(),
                    model.readout_weight_reg(), model.readout_loc_reg(),
                ]
            )]).sum()
            total_loss = fit_loss+reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if verbose>0 and (batch_idx%(-(-batch_num//self.train_num_infos))==0 or batch_idx==batch_num):
                print('{}: [fit loss: {:7.4f}], [reg loss: {:.4f}]'.format(
                    progress_str(batch_idx, batch_num, True), fit_loss.item(), reg_loss.item(),
                ))

    def fetch_model(self, config: dict):
        data_config = config['data_config']
        model_config = config['model_config']
        train_config = config['train_config']

        _, _, scaled_means, loo_corrs = self.meta_info(data_config)
        dsets, neuron_num = self.prepare_datasets(
            data_config, train_config,
        )

        model = self.prepare_model(neuron_num, model_config)
        try:
            epoch, ckpt, _ = self.load_ckpt(config)
            _, o_frac = ckpt['eval_records']['test'][ckpt['best_epoch']]
            print("Best model fetched at checkpoint {}/{}".format(ckpt['best_epoch'], epoch))
        except:
            raise RuntimeError("No completed training found.")
        model.load_state_dict(tensor_dict(ckpt['best_state']))
        outputs, targets = self.dataset_forward(dsets['test'], model)
        pred_corrs = response_corrs(outputs, targets)

        return model, pred_corrs, loo_corrs, o_frac