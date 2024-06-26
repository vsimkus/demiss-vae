import os
from typing import Any, Dict
import pytorch_lightning as pl
import torch
import numpy as np

from vgiwae.utils.imputation_distribution import (IMPUTATION_INIT, ImputationDistribution)
from vgiwae.shared.vae_imputation import VAESampler_Base

def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output

def nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output

class ImputerModuleBase(pl.LightningModule):
    """
    A LightningModule base class which initialises an imputation distribution
    using the dataset in datamodule.

    Args:
        num_imputations_train:             Number of imputations for the training data.
        imputation_init_train:             How to initialise the imputations of training data.
        num_imputations_val:               Number of imputations for the validation data.
        imputation_init_val:               How to initialise the imputations of validation data.
        init_imputation_distribution:      If False the imputation distributions are not initialised.
        mvb_delay_imputation:              The number of epochs to wait before starting to impute using the VAE.
        mvb_resample_initial_imputations:  If True the initial imputations are resampled every epoch.
        vae_sampler:                       Sampler object implementing the VAESampler_Base interface.
        save_imputations:                  Stores imputations to a file.
    """
    def __init__(self,
                 num_imputations_train: int = 1,
                 imputation_init_train: IMPUTATION_INIT = IMPUTATION_INIT('zero'),
                 num_imputations_val: int = 1,
                 imputation_init_val: IMPUTATION_INIT = IMPUTATION_INIT('zero'),
                 init_imputation_distribution: bool = False,
                 mvb_delay_imputation: int = 1,
                 mvb_resample_initial_imputations: bool = False,
                 vae_sampler: VAESampler_Base = None,
                 save_imputations: bool = False
                 ):
        super().__init__()
        self.save_hyperparameters()

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def on_train_end(self):
        out = super().on_train_end()

        if self.hparams.save_imputations:
            exp_path = self.logger.experiment.get_logdir()
            path = os.path.join(exp_path, 'imputations')
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, '{}_imputations_last.npz')
            if hasattr(self, 'train_imp_distribution'):
                self.train_imp_distribution.save_imputations(path.format('train'))
            if hasattr(self, 'val_imp_distribution'):
                self.val_imp_distribution.save_imputations(path.format('val'))
            if hasattr(self, 'test_imp_distribution'):
                self.test_imp_distribution.save_imputations(path.format('test'))

        return out

    def setup(self, stage):
        super().setup(stage)
        # TODO: detect resume and resume from saved imputations

        if stage == 'fit':
            if self.hparams.init_imputation_distribution:
                self.train_imp_distribution = self.create_imputation_distribution(
                                                    self.trainer.datamodule.train_data,
                                                    num_imputations=self.hparams.num_imputations_train,
                                                    imputation_init=self.hparams.imputation_init_train)
                self.train_imp_distribution.init_imputations()
                self.val_imp_distribution = self.create_imputation_distribution(
                                                    self.trainer.datamodule.val_data,
                                                    num_imputations=self.hparams.num_imputations_val,
                                                    imputation_init=self.hparams.imputation_init_val)
                self.val_imp_distribution.init_imputations()
        elif stage == 'test':
            # TODO
            pass

    def create_imputation_distribution(self,
                                       dataset: torch.utils.data.Dataset,
                                       num_imputations: int,
                                       imputation_init: IMPUTATION_INIT):
        return ImputationDistribution(
                dataset=dataset,
                target_idx=0,
                mis_idx=1,
                index_idx=dataset.index_idx,
                num_imputations=num_imputations,
                imputation_init=imputation_init
            )

    def update_imputations(self, imputed_batch, stage):
        clip_values_max = None
        clip_values_min = None
        # If clipping is enabled, calculate the clipping values
        if self.hparams.vae_sampler.clip_imputations:
            if self.hparams.vae_sampler.clipping_mode == 'batch':
                X_temp = imputed_batch[0]
                X_temp = X_temp[:, 0].clone()
                X_temp[~M.squeeze(1)] = float('nan')
                clip_values_max = nanmax(X_temp, dim=0)[0]*1.5
                clip_values_min = nanmin(X_temp, dim=0)[0]*1.5
                del X_temp
            elif self.hparams.vae_sampler.clipping_mode == 'dataset':
                if stage in ('train', 'val'):
                    dataset = self.datamodule.train_data_core
                else:
                    raise NotImplementedError(f'Invalid {stage=}')
                X_temp = imputed_batch[0]
                clip_values_max = torch.tensor(dataset.data_max*1.5, device=X_temp.device)
                clip_values_min = torch.tensor(dataset.data_min*1.5, device=X_temp.device)
                del X_temp
            else:
                raise NotImplementedError(f'Invalid {self.hparams.clipping_mode=}')

        if self.current_epoch >= self.hparams.mvb_delay_imputation:
            X, M = imputed_batch[:2]

            with torch.inference_mode():
                X_imp = self.hparams.vae_sampler(X, M, model=self,
                                                 clip_values_min=clip_values_min,
                                                 clip_values_max=clip_values_max)

                # Set imputed datapoints
                imputed_batch = (X_imp,) + imputed_batch[1:] #  -1] + (Z_imp,)
                if stage == 'train':
                    self.train_imp_distribution.set_imputed_datapoints(imputed_batch)
                elif stage == 'val':
                    self.val_imp_distribution.set_imputed_datapoints(imputed_batch)
                else: #elif stage == 'test':
                    # TODO: add this if necessary
                    # self.test_imp_distribution.set_imputed_datapoints(imputed_batch)
                    raise NotImplementedError()

        return imputed_batch

    def resample_imputations(self, stage):
        """
        Resamples the initial imputations using the base imputer.
        """
        if self.current_epoch < self.hparams.mvb_delay_imputation and self.hparams.mvb_resample_initial_imputations:
            if stage == 'train':
                self.train_imp_distribution.init_imputations()
            elif stage == 'val':
                self.val_imp_distribution.init_imputations()
            else: #elif stage == 'test':
                # TODO: add this if necessary
                # self.test_imp_distribution.init_imputations()
                raise NotImplementedError()
