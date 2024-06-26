import os.path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from einops import reduce, rearrange
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vgiwae.models.vae import VAE
from vgiwae.shared.iwae import compute_dmis_log_unnormalised_importance_weights
from vgiwae.shared.vae_enums import DISTRIBUTION


class MultipleIWAE(VAE):
    """
    A Multiple-IWAE model with missing data.

    Args:
        num_importance_samples_for_each_component:      The number of samples (for each mixture component) used in importance sampling.
        kl_analytic:                                    For IWAE should only be False.
        var_latent_DREG:                                if true uses the gradients from "Doubly Reparametrized Gradient Estimators" by Tucker et al. 2018
        use_looser_bound:                               Use looser bound from Shi et al 2019, or Kviman et al 2023
    """

    def __init__(self,
                 *args,
                 num_importance_samples_for_each_component: int = 1,
                 kl_analytic: bool = False, # Override
                 var_latent_DREG: bool = False,
                 log_var_distribution_mixture_component_weights_epoch_frequency: int = -1,
                 use_looser_bound: bool = False,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs, kl_analytic=kl_analytic)
        self.save_hyperparameters()

        assert not kl_analytic, 'Analytical KL is not tractable for a IWAE.'

        assert not (self.hparams.var_latent_DREG and self.hparams.var_latent_STL),\
            'Cannot use STL and DREG gradients at the same time.'

        assert not self.hparams.var_latent_DREG, 'DREG gradients are not yet implemented for multiple-IWAE.'

        assert self.hparams.var_latent_distribution in (DISTRIBUTION.stratified_mixture1_normal_with_eps,
                                                        DISTRIBUTION.stratified_mixture5_normal_with_eps,
                                                        DISTRIBUTION.stratified_mixture15_normal_with_eps,
                                                        DISTRIBUTION.stratified_mixture25_normal_with_eps,
                                                        DISTRIBUTION.stratified_mixture1_normal,
                                                        DISTRIBUTION.stratified_mixture5_normal,
                                                        DISTRIBUTION.stratified_mixture15_normal,
                                                        DISTRIBUTION.stratified_mixture25_normal),\
            'Only mixture-distributions are supported for encoder in multiple-IWAE.'

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        elbo = self.estimate_multiple_iwelbo(batch)
        loss = -elbo

        # logs metrics
        self.log('elbo/train', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        with torch.inference_mode():
            elbo = self.estimate_multiple_iwelbo(batch)
            loss = -elbo

        # logs metrics
        self.log('elbo/val', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def estimate_multiple_iwelbo(self, batch):
        return estimate_multiple_iwelbo(self, batch,
                                        var_latent_STL=self.hparams.var_latent_STL,
                                        var_latent_DREG=self.hparams.var_latent_DREG,
                                        num_latent_samples=self.hparams.num_latent_samples,
                                        num_importance_samples_for_each_component=self.hparams.num_importance_samples_for_each_component,
                                        use_looser_bound=self.hparams.use_looser_bound
        )

    def on_train_start(self) -> None:
        out = super().on_train_start()

        self.mixture_weight_logits = []

        return out

    def on_train_end(self) -> None:
        out = super().on_train_end()

        if self.hparams.log_var_distribution_mixture_component_weights_epoch_frequency != -1:
            mixture_weight_logits = np.stack(self.mixture_weight_logits, 0)

            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), f'mixture_var_weight_logits.npz'),
                                mixture_weight_logits=mixture_weight_logits)

        return out

    def on_train_epoch_start(self):
        out = super().on_train_epoch_start()

        self.mixture_weight_logits_epoch = []

        return out

    def on_train_epoch_end(self):
        out = super().on_train_epoch_end()

        if (self.hparams.log_var_distribution_mixture_component_weights_epoch_frequency != -1
                and self.current_epoch % self.hparams.log_var_distribution_mixture_component_weights_epoch_frequency == 0):
            mixture_weight_logits_epoch = np.concatenate(self.mixture_weight_logits_epoch, axis=0)

            self.mixture_weight_logits.append(mixture_weight_logits_epoch)


        return out


def log_weights_to_multiple_iwelbo(log_weights, var_latent_distr, *, create_Z_hook=None, var_latent_DREG=False):
    num_importance_samples = log_weights.shape[1]
    # Add mixture-log-probs
    log_weights = log_weights + var_latent_distr._mixture_distribution.logits - torch.log(torch.tensor(num_importance_samples))

    # Reshape to (z, i*k, b)
    log_weights = rearrange(log_weights, 'z i b k -> z (i k) b')

    # Compute the total IWELBO term
    iwelbo = torch.logsumexp(log_weights, dim=1)

    if var_latent_DREG and create_Z_hook is not None:
        with torch.no_grad():
            norm_weights = torch.softmax(log_weights, dim=1)
            create_Z_hook(norm_weights)

    return iwelbo

def log_weights_to_multiple_iwelbo_looser(log_weights, var_latent_distr, *, create_Z_hook=None, var_latent_DREG=False):
    num_importance_samples = log_weights.shape[1]
    # Log-average log-weights
    log_weights = log_weights - torch.log(torch.tensor(num_importance_samples))

    # Compute the total IWELBO term
    # Log-sum-exp over I
    iwelbo = torch.logsumexp(log_weights, dim=1)
    # Weighted average over K
    iwelbo = torch.sum(iwelbo * torch.exp(var_latent_distr._mixture_distribution.logits), dim=-1)

    if var_latent_DREG and create_Z_hook is not None:
        raise NotImplementedError()
        # with torch.no_grad():
        #     norm_weights = torch.softmax(log_weights, dim=1)
        #     create_Z_hook(norm_weights)

    return iwelbo

def compute_multiple_iwelbo(X: torch.Tensor,
                            M: torch.Tensor,
                            Z: torch.Tensor,
                            var_latent_distr: torch.distributions.Distribution,
                            var_latent_distr_detached: torch.distributions.Distribution,
                            prior_distr: torch.distributions.Distribution,
                            generator_distr: torch.distributions.Distribution,
                            var_latent_STL: bool = False,
                            var_latent_DREG: bool = False,
                            use_looser_bound: bool = False) -> torch.Tensor:

    # Use detached distribution for computing the score if using STL or DREG gradients
    var_latent_distr_ = (var_latent_distr_detached if var_latent_STL or var_latent_DREG
                            else var_latent_distr)

    # Compute the unnormalised importance weights
    log_weights = compute_dmis_log_unnormalised_importance_weights(X.unsqueeze(1), M.unsqueeze(1), Z,
                                                                    var_latent_distr=var_latent_distr_,
                                                                    var_comp_neg_idx=-2,
                                                                    prior_distr=prior_distr,
                                                                    generator_distr=generator_distr,
                                                                    )
    # Shape of weights is (z, i, b, k)

    def create_Z_hook(norm_weights):
        if Z.requires_grad:
            Z.register_hook(lambda grad: norm_weights[..., None] * grad)

    # Compute the avg IWELBO term
    # NOTE: need to take gradients wrt var_latent_distr._mixture_distribution.logits (so should pass the non-detached version)
    if not use_looser_bound:
        iwelbo = log_weights_to_multiple_iwelbo(log_weights, var_latent_distr, create_Z_hook=create_Z_hook,
                                                var_latent_DREG=var_latent_DREG)
    else:
        iwelbo = log_weights_to_multiple_iwelbo_looser(log_weights, var_latent_distr, create_Z_hook=create_Z_hook,
                                                       var_latent_DREG=var_latent_DREG)
    iwelbo = reduce(iwelbo, 'z b -> b', 'mean')

    return iwelbo

def estimate_multiple_iwelbo(model, batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                             var_latent_STL: bool = False,
                             var_latent_DREG: bool = False,
                             num_latent_samples: int = 1,
                             num_importance_samples_for_each_component: int = 1,
                             use_looser_bound: bool = False) -> torch.Tensor:
    X, M = batch[:2]

    # Forward pass over the VAE
    # NOTE: X*M to ensure no leakage of missing values for the base (fully-observed) model
    vae_forward_outputs = model.vae_forward(X*M, M,
                                            return_detached_latent_distr=var_latent_STL or var_latent_DREG,
                                            Z_sample_shape=(num_latent_samples, num_importance_samples_for_each_component,))
    var_latent_distr = vae_forward_outputs.var_latent_distr
    var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
    Z = vae_forward_outputs.Z
    prior_distr = vae_forward_outputs.prior_distr
    generator_distr = vae_forward_outputs.generator_distr

    if (model.training
            and hasattr(model.hparams, 'log_var_distribution_mixture_component_weights_epoch_frequency')
            and model.hparams.log_var_distribution_mixture_component_weights_epoch_frequency != -1
            and model.current_epoch % model.hparams.log_var_distribution_mixture_component_weights_epoch_frequency == 0):
        # Log mixture logits during training
        logits = var_latent_distr._mixture_distribution.logits.detach().cpu().numpy()
        model.mixture_weight_logits_epoch.append(logits)

    # Compute per data-point elbo
    elbo = compute_multiple_iwelbo(X, M, Z, var_latent_distr, var_latent_distr_detached, prior_distr, generator_distr,
                                    var_latent_STL=var_latent_STL,
                                    var_latent_DREG=var_latent_DREG,
                                    use_looser_bound=use_looser_bound)

    # Averaged elbo
    elbo = reduce(elbo, 'b -> ', 'mean')

    return elbo
