import os.path
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from einops import reduce
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vgiwae.models.vae import VAE
from vgiwae.shared.vae_enums import DISTRIBUTION
from vgiwae.shared.stratified_mixture_same_family import StratifiedMixtureSameFamily


class MultipleVAE(VAE):
    """
    A Multiple-VAE model with missing data.

    Args:
        kl_analytic:                                    For MultipleVAE should only be False.
    """

    def __init__(self,
                 *args,
                 kl_analytic: bool = False, # Override
                 log_var_distribution_mixture_component_weights_epoch_frequency: int = -1,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs, kl_analytic=kl_analytic)
        self.save_hyperparameters()

        assert not kl_analytic, 'Analytical KL is not implemented for a MultipleVAE.'

        assert self.hparams.var_latent_distribution in (DISTRIBUTION.stratified_mixture1_normal_with_eps,
                                                        DISTRIBUTION.stratified_mixture5_normal_with_eps,
                                                        DISTRIBUTION.stratified_mixture15_normal_with_eps,
                                                        DISTRIBUTION.stratified_mixture25_normal_with_eps,
                                                        DISTRIBUTION.stratified_mixture1_normal,
                                                        DISTRIBUTION.stratified_mixture5_normal,
                                                        DISTRIBUTION.stratified_mixture15_normal,
                                                        DISTRIBUTION.stratified_mixture25_normal),\
            'Only mixture-distributions are supported for encoder in multiple-VAE.'

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        elbo = estimate_multiple_elbo(self, batch,
                                      kl_analytic=self.hparams.kl_analytic,
                                      var_latent_STL=self.hparams.var_latent_STL,
                                      num_latent_samples=self.hparams.num_latent_samples)
        loss = -elbo

        # logs metrics
        self.log('elbo/train', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        with torch.inference_mode():
            elbo = estimate_multiple_elbo(self, batch,
                                          kl_analytic=self.hparams.kl_analytic,
                                          var_latent_STL=self.hparams.var_latent_STL,
                                          num_latent_samples=self.hparams.num_latent_samples)
            loss = -elbo

        # logs metrics
        self.log('elbo/val', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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

def compute_multiple_elbo_w_montecarlokl(
                    X: torch.Tensor,
                    M: torch.Tensor,
                    Z: torch.Tensor,
                    var_latent_distr: torch.distributions.Distribution,
                    var_latent_distr_detached: torch.distributions.Distribution,
                    prior_distr: torch.distributions.Distribution,
                    generator_distr: torch.distributions.Distribution,
                    var_latent_STL: bool = False) -> torch.Tensor:
    # Compute cross-entropy term of observed data
    generator_log_prob = generator_distr.log_prob(X.unsqueeze(1))*M.unsqueeze(1)
    generator_log_prob = reduce(generator_log_prob, 'z b ... d -> z b ...', 'sum')

    # Compute prior latent probability
    prior_logprob = prior_distr.log_prob(Z)
    prior_logprob = reduce(prior_logprob, 'z b ... d -> z b ...', 'sum')

    # Compute the log-prob of samples under the latent distribution
    if isinstance(var_latent_distr, StratifiedMixtureSameFamily):
        var_comp_neg_idx = -2
        # Handle the computation differently for StratifiedMixtureSameFamily
        idxs = np.arange(-len(Z.shape), 0)
        rest_idxs = np.delete(idxs, np.argwhere(idxs==var_comp_neg_idx))
        Z_aug = torch.permute(Z, [var_comp_neg_idx] + list(rest_idxs))

        if var_latent_STL:
            latent_logprob = var_latent_distr_detached.log_prob(Z_aug)
        else:
            latent_logprob = var_latent_distr.log_prob(Z_aug)

        # Undo the permute
        idxs = np.arange(-len(Z.shape)+1+1, 0, dtype=int)
        idxs = np.insert(idxs, idxs.size+(var_comp_neg_idx+1+1), 0)
        latent_logprob = torch.permute(latent_logprob, list(idxs))
    else:
        if var_latent_STL:
            # NOTE: alternatively could use this https://github.com/pyro-ppl/pyro/pull/2599/
            latent_logprob = var_latent_distr_detached.log_prob(Z)
        else:
            latent_logprob = var_latent_distr.log_prob(Z)
        latent_logprob = reduce(latent_logprob, 'z b ... d -> z b ...', 'sum')

    # Component probabilities
    comp_probs = torch.softmax(var_latent_distr._mixture_distribution.logits, dim=-1)

    # Compute per-data-point elbo
    total_ratio = generator_log_prob + prior_logprob - latent_logprob
    total_ratio = (total_ratio * comp_probs).sum(-1)

    elbos = reduce(total_ratio, 'z b ... -> b ...', 'mean')

    return elbos

def estimate_multiple_elbo(model, batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                           kl_analytic: bool = False,
                           var_latent_STL: bool = False,
                           num_latent_samples: int = 1) -> torch.Tensor:
    X, M = batch[:2]

    # Forward pass over the VAE
    # NOTE: X*M to ensure no leakage of missing values for the base (fully-observed) model
    vae_forward_outputs = model.vae_forward(X*M, M, return_detached_latent_distr=var_latent_STL,
                                            Z_sample_shape=(num_latent_samples,))
    var_latent_distr = vae_forward_outputs.var_latent_distr
    var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
    Z = vae_forward_outputs.Z
    prior_distr = vae_forward_outputs.prior_distr
    generator_distr = vae_forward_outputs.generator_distr

    if (model.training
            and model.hparams.log_var_distribution_mixture_component_weights_epoch_frequency != -1
            and model.current_epoch % model.hparams.log_var_distribution_mixture_component_weights_epoch_frequency == 0):
        # Log mixture logits during training
        logits = var_latent_distr._mixture_distribution.logits.detach().cpu().numpy()
        model.mixture_weight_logits_epoch.append(logits)

    # Compute per data-point elbo
    if kl_analytic:
        raise NotImplementedError('Analytical KL is not implemented for a MultipleVAE.')
    else:
        elbo = compute_multiple_elbo_w_montecarlokl(X, M, Z, var_latent_distr, var_latent_distr_detached, prior_distr, generator_distr,
                                                    var_latent_STL=var_latent_STL)

    # Averaged elbo
    elbo = reduce(elbo, 'b -> ', 'mean')

    return elbo
