from typing import List, Optional, Tuple, Type, Union

import torch
from einops import reduce
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vgiwae.shared.vae_marginal_logprob import get_marginal_logprob, get_posterior_divs

from .vae import VAE
from .iwae import IWAE
from .mvbvae import MVBVAE
from .mvbiwae import MVBIWAE

from .multiple_vae import MultipleVAE
from .multiple_iwae import MultipleIWAE


def create_marglogprob_subclass(classname: str, parent_classes: Tuple[Type[object]]) -> Type[object]:
    """Computes marginal log-likelihood on validation data (and optionally training data) by numerically integrating the latents."""
    def __init__(self, *args, marginal_eval_batchsize=-1, marginal_eval_freq=1, marginal_eval_train=False,
                 latent_grid_min=None, latent_grid_max=None, latent_grid_steps=None,
                 compute_complete_var_distribution_kl=False, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)
        # With this dynamic subclassing save_hyperparameters does not automatically save the arguments of the subclass, so save them manually
        self.hparams.marginal_eval_batchsize = marginal_eval_batchsize
        self.hparams.marginal_eval_freq = marginal_eval_freq
        self.hparams.marginal_eval_train = marginal_eval_train

        self.hparams.latent_grid_min = latent_grid_min
        self.hparams.latent_grid_max = latent_grid_max
        if self.hparams.latent_grid_min is not None and self.hparams.latent_grid_max is not None:
            assert self.hparams.latent_grid_min < self.hparams.latent_grid_max
        self.hparams.latent_grid_steps = latent_grid_steps

        self.hparams.compute_complete_var_distribution_kl = compute_complete_var_distribution_kl

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> Optional[STEP_OUTPUT]:
        out = super(type(self), self).training_step(batch, batch_idx)

        with torch.inference_mode():
            if (self.hparams.marginal_eval_train and
                    (self.current_epoch % self.hparams.marginal_eval_freq == 0
                    or self.current_epoch == self.trainer.max_epochs-1)):
                marginal_log_prob, comp_marginal_log_prob = get_marginal_logprob(self, batch, compute_complete=True, marginal_eval_batchsize=self.hparams.marginal_eval_batchsize,
                                                                                 grid_min=self.hparams.latent_grid_min, grid_max=self.hparams.latent_grid_max, grid_steps=self.hparams.latent_grid_steps)
                avg_log_prob = reduce(marginal_log_prob, 'b -> ', 'mean')
                avg_comp_log_prob = reduce(comp_marginal_log_prob, 'b -> ', 'mean')

                self.log('marginal_logprob/train', avg_log_prob, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('complete_marginal_logprob/train', avg_comp_log_prob, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                if self.hparams.compute_complete_var_distribution_kl:
                    imputed_batch = self.train_imp_distribution.get_imputed_datapoints(batch)

                    comp_kl_fow, comp_kl_rev, complete_jsd, incomplete_kl_fow, incomplete_kl_rev, incomplete_jsd, complete_imps_kl_fow, complete_imps_kl_rev, complete_imps_jsd = \
                        get_posterior_divs(self, batch, imputed_batch, marginal_eval_batchsize=self.hparams.marginal_eval_batchsize,
                                           grid_min=self.hparams.latent_grid_min, grid_max=self.hparams.latent_grid_max, grid_steps=self.hparams.latent_grid_steps)
                    comp_kl_fow = reduce(comp_kl_fow, 'b -> ', 'mean')
                    comp_kl_rev = reduce(comp_kl_rev, 'b -> ', 'mean')
                    complete_jsd = reduce(complete_jsd, 'b -> ', 'mean')
                    incomplete_kl_fow = reduce(incomplete_kl_fow, 'b -> ', 'mean')
                    incomplete_kl_rev = reduce(incomplete_kl_rev, 'b -> ', 'mean')
                    incomplete_jsd = reduce(incomplete_jsd, 'b -> ', 'mean')
                    comp_imps_kl_fow = reduce(complete_imps_kl_fow, 'b k -> ', 'mean')
                    comp_imps_kl_rev = reduce(complete_imps_kl_rev, 'b k -> ', 'mean')
                    comp_imps_jsd = reduce(complete_imps_jsd, 'b k -> ', 'mean')

                    self.log('complete_posterior_kl_fow/train', comp_kl_fow, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_posterior_kl_rev/train', comp_kl_rev, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_posterior_jsd/train', complete_jsd, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('incomplete_posterior_kl_fow/train', incomplete_kl_fow, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('incomplete_posterior_kl_rev/train', incomplete_kl_rev, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('incomplete_posterior_jsd/train', incomplete_jsd, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_imps_posterior_kl_fow/train', comp_imps_kl_fow, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_imps_posterior_kl_rev/train', comp_imps_kl_rev, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_imps_posterior_jsd/train', comp_imps_jsd, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return out

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        out = super(type(self), self).validation_step(batch, batch_idx)

        with torch.inference_mode():
            if (self.current_epoch % self.hparams.marginal_eval_freq == 0
                    or self.current_epoch == self.trainer.max_epochs-1):
                marginal_log_prob, comp_marginal_log_prob = get_marginal_logprob(self, batch, compute_complete=True, marginal_eval_batchsize=self.hparams.marginal_eval_batchsize,
                                                                                 grid_min=self.hparams.latent_grid_min, grid_max=self.hparams.latent_grid_max, grid_steps=self.hparams.latent_grid_steps)
                avg_log_prob = reduce(marginal_log_prob, 'b -> ', 'mean')
                avg_comp_log_prob = reduce(comp_marginal_log_prob, 'b -> ', 'mean')

                self.log('marginal_logprob/val', avg_log_prob, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('complete_marginal_logprob/val', avg_comp_log_prob, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                if self.hparams.compute_complete_var_distribution_kl:
                    imputed_batch = self.val_imp_distribution.get_imputed_datapoints(batch)

                    comp_kl_fow, comp_kl_rev, complete_jsd, incomplete_kl_fow, incomplete_kl_rev, incomplete_jsd, complete_imps_kl_fow, complete_imps_kl_rev, complete_imps_jsd = \
                        get_posterior_divs(self, batch, imputed_batch, marginal_eval_batchsize=self.hparams.marginal_eval_batchsize,
                                           grid_min=self.hparams.latent_grid_min, grid_max=self.hparams.latent_grid_max, grid_steps=self.hparams.latent_grid_steps)
                    comp_kl_fow = reduce(comp_kl_fow, 'b -> ', 'mean')
                    comp_kl_rev = reduce(comp_kl_rev, 'b -> ', 'mean')
                    complete_jsd = reduce(complete_jsd, 'b -> ', 'mean')
                    incomplete_kl_fow = reduce(incomplete_kl_fow, 'b -> ', 'mean')
                    incomplete_kl_rev = reduce(incomplete_kl_rev, 'b -> ', 'mean')
                    incomplete_jsd = reduce(incomplete_jsd, 'b -> ', 'mean')
                    comp_imps_kl_fow = reduce(complete_imps_kl_fow, 'b k -> ', 'mean')
                    comp_imps_kl_rev = reduce(complete_imps_kl_rev, 'b k -> ', 'mean')
                    comp_imps_jsd = reduce(complete_imps_jsd, 'b k -> ', 'mean')

                    self.log('complete_posterior_kl_fow/val', comp_kl_fow, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_posterior_kl_rev/val', comp_kl_rev, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_posterior_jsd/val', complete_jsd, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('incomplete_posterior_kl_fow/val', incomplete_kl_fow, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('incomplete_posterior_kl_rev/val', incomplete_kl_rev, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('incomplete_posterior_jsd/val', incomplete_jsd, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_imps_posterior_kl_fow/val', comp_imps_kl_fow, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_imps_posterior_kl_rev/val', comp_imps_kl_rev, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log('complete_imps_posterior_jsd/val', comp_imps_jsd, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return out

    overrides = {
        '__init__': __init__,
        'validation_step': validation_step,
        'training_step': training_step
    }

    subclass = type(classname, parent_classes, overrides)

    # Let the docstring of the new class be the one from the parent class + description of subclass
    subclass.__doc__ = parent_classes[0].__doc__ + '\n\nComputes marginal log-likelihood on validation data (and optionally training data) by numerically integrating the latents.'

    return subclass


VAEMargLogprob = create_marglogprob_subclass('VAEMargLogprob', (VAE,))
IWAEMargLogprob = create_marglogprob_subclass('IWAEMargLogprob', (IWAE,))
MVBVAEMargLogprob = create_marglogprob_subclass('MVBVAEMargLogprob', (MVBVAE,))
MVBIWAEMargLogprob = create_marglogprob_subclass('MVBIWAEMargLogprob', (MVBIWAE,))

MultipleVAEMargLogprob = create_marglogprob_subclass('MultipleVAEMargLogprob', (MultipleVAE,))
MultipleIWAEMargLogprob = create_marglogprob_subclass('MultipleIWAEMargLogprob', (MultipleIWAE,))
