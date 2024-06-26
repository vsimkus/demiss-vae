from typing import List, Optional, Tuple, Union

import torch
from einops import reduce
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vgiwae.models.vae import VAE
from vgiwae.shared.iwae import compute_smis_log_unnormalised_importance_weights


class IWAE(VAE):
    """
    A IWAE model with missing data.

    Args:
        num_importance_samples:     The number of samples used in importance sampling.
        kl_analytic:                For IWAE should only be False.
        var_latent_DREG:            if true uses the gradients from "Doubly Reparametrized Gradient Estimators" by Tucker et al. 2018
    """

    def __init__(self,
                 *args,
                 num_importance_samples: int = 1,
                 kl_analytic: bool = False, # Override
                 var_latent_DREG: bool = False,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs, kl_analytic=kl_analytic)
        self.save_hyperparameters()

        assert not kl_analytic, 'Analytical KL is not tractable for a IWAE.'

        assert not (self.hparams.var_latent_DREG and self.hparams.var_latent_STL),\
            'Cannot use STL and DREG gradients at the same time.'

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        elbo = estimate_iwelbo(self, batch,
                               var_latent_STL=self.hparams.var_latent_STL,
                               var_latent_DREG=self.hparams.var_latent_DREG,
                               num_latent_samples=self.hparams.num_latent_samples,
                               num_importance_samples=self.hparams.num_importance_samples)
        loss = -elbo

        # logs metrics
        self.log('elbo/train', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        with torch.inference_mode():
            elbo = estimate_iwelbo(self, batch,
                                   var_latent_STL=self.hparams.var_latent_STL,
                                   var_latent_DREG=self.hparams.var_latent_DREG,
                                   num_latent_samples=self.hparams.num_latent_samples,
                                   num_importance_samples=self.hparams.num_importance_samples)
            loss = -elbo

        # logs metrics
        self.log('elbo/val', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def log_weights_to_iwelbo(self, *args, **kwargs):
        return log_weights_to_iwelbo(*args, **kwargs)

def log_weights_to_iwelbo(log_weights, *, create_Z_hook=None, var_latent_DREG=False):
    # Compute the total IWELBO term
    # NOTE: sometimes it is implemented as below, but this loses numerical accuracy, due to the normalisation.
    # with torch.no_grad():
    #     norm_weights = torch.softmax(log_weights, dim=1)
    # iwelbo = reduce(norm_weights*log_weights, 'z i b ... -> z b ...', 'sum')
    iwelbo = torch.logsumexp(log_weights, dim=1) - torch.log(torch.tensor(log_weights.shape[1]))

    if var_latent_DREG and create_Z_hook is not None:
        with torch.no_grad():
            norm_weights = torch.softmax(log_weights, dim=1)
            create_Z_hook(norm_weights)

        # if Z.requires_grad:
        #     # Multiply the gradients by the norm_weights,
        #     # this is effectively the same as using sg(sq_norm_weights)*log_weights as the objective
        #     Z.register_hook(lambda grad: norm_weights.unsqueeze(-1) * grad)
    # else:
    #     iwelbo = torch.logsumexp(log_weights, dim=1) - torch.log(torch.tensor(log_weights.shape[1]))

    return iwelbo

def compute_iwelbo(X: torch.Tensor,
                   M: torch.Tensor,
                   Z: torch.Tensor,
                   var_latent_distr: torch.distributions.Distribution,
                   var_latent_distr_detached: torch.distributions.Distribution,
                   prior_distr: torch.distributions.Distribution,
                   generator_distr: torch.distributions.Distribution,
                   var_latent_STL: bool = False,
                   var_latent_DREG: bool = False) -> torch.Tensor:
    assert not (var_latent_DREG and var_latent_STL),\
            'Cannot use STL and DREG gradients at the same time.'

    # Use detached distribution for computing the score if using STL or DREG gradients
    var_latent_distr_ = (var_latent_distr_detached if var_latent_STL or var_latent_DREG
                            else var_latent_distr)

    # Compute the unnormalised importance weights
    log_weights = compute_smis_log_unnormalised_importance_weights(X, M, Z,
                                                                    var_latent_distr_,
                                                                    prior_distr,
                                                                    generator_distr)
    # Shape of weights is (z, i, b)

    def create_Z_hook(norm_weights):
        if Z.requires_grad:
            Z.register_hook(lambda grad: norm_weights[..., None] * grad)

    # Compute the avg IWELBO term
    iwelbo = log_weights_to_iwelbo(log_weights, create_Z_hook=create_Z_hook, var_latent_DREG=var_latent_DREG)
    iwelbo = reduce(iwelbo, 'z b -> b', 'mean')

    return iwelbo

def estimate_iwelbo(model, batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                    var_latent_STL: bool = False,
                    var_latent_DREG: bool = False,
                    num_latent_samples: int = 1,
                    num_importance_samples: int = 1,
                    *,
                    sample_stratifieddistr_without_stratification_norsample=False) -> torch.Tensor:
    X, M = batch[:2]

    # Forward pass over the VAE
    # NOTE: X*M to ensure no leakage of missing values for the base (fully-observed) model
    vae_forward_outputs = model.vae_forward(X*M, M,
                                            return_detached_latent_distr=var_latent_STL or var_latent_DREG,
                                            Z_sample_shape=(num_latent_samples, num_importance_samples,),
                                            sample_stratifieddistr_without_stratification_norsample=sample_stratifieddistr_without_stratification_norsample)
    var_latent_distr = vae_forward_outputs.var_latent_distr
    var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
    Z = vae_forward_outputs.Z
    prior_distr = vae_forward_outputs.prior_distr
    generator_distr = vae_forward_outputs.generator_distr

    # Compute per data-point elbo
    elbo = compute_iwelbo(X, M, Z, var_latent_distr, var_latent_distr_detached, prior_distr, generator_distr,
                          var_latent_STL=var_latent_STL,
                          var_latent_DREG=var_latent_DREG)

    # Averaged elbo
    elbo = reduce(elbo, 'b -> ', 'mean')

    return elbo
