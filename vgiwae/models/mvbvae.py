from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import torch
from einops import reduce
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vgiwae.shared.iwae import latent_mixture_logprob
from vgiwae.shared.mvb2_manual_optimisation import mvb2_manual_optimisation_step

from .vae import VAE


class MVB_OBJECTIVE(Enum):
    mvb = auto() # Use MVB objective for theta and phi
    cvi = auto() # Use CVI objective for theta and phi
    mvb2 = auto() # Use MVB objective for phi and CVI objective for theta

class MVBVAE(VAE):
    """
    A VAE model with missing data.
    Uses the marginalised variational bound.

    Args:
        mvb_sample_num_imputations:       If None, all imputations are used in the objective, if int, only a
                                          subset of all imputations is sampled and used in the objective
        mvb_objective:                    Which objective to use for generator/encoder.
        mvb_update_val_imps:              If true, updates the validation imputations at the end of each epoch.
        mvb_eval_val_loss:                If true evaluates the validation loss (with imputation sampling),
                                          it can cause issues due to inference generalisation gap.
        mvb2_optim_gradclip_val:          Gradient clipping value for MVB with two objectives.
        mvb2_optim_gradclip_alg:          Gradient clipping method for MVB with two objectives.
    """

    def __init__(self,
                 *args,
                 mvb_sample_num_imputations: int = None,
                 mvb_objective: MVB_OBJECTIVE = MVB_OBJECTIVE['mvb'],
                 mvb_update_val_imps: bool = True,
                 mvb_eval_val_loss: bool = False,
                 mvb2_optim_gradclip_val: float = 0,
                 mvb2_optim_gradclip_alg: str = 'norm',
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if self.hparams.mvb_objective is MVB_OBJECTIVE['mvb2']:
            # NOTE: this disables PytorchLightning's automatic optimisation,
            # hence you must handle optimisation manually in training_step.
            # This also means that some of PL's features won't work, such as
            # gradient clipping, gradient accumulation, learning rate scheduling, etc.
            # See https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html
            # or https://pytorch-lightning.readthedocs.io/en/1.5.8/common/lightning_module.html?highlight=automatic_optimization#automatic-optimization
            self.automatic_optimization = False

    mvb2_manual_optimisation_step = mvb2_manual_optimisation_step

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.resample_imputations(stage='train')

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self.resample_imputations(stage='val')

    def estimate_mvb2_objectives(self, Xt: torch.Tensor, M: torch.Tensor):
        """
        Compute the MVB objective for fitting the encoder, and the CVI objective for fitting the decoder.

        NOTE: you shouldn't optimise this objective naively on all parameters.
        Instead see the code in `self.mvb2_manual_optimisation_step()`, which ensures that you optimise
        theta using CVI objective and phi using MVB objective.
        """
        # Forward pass over the VAE
        vae_forward_outputs = self.vae_forward(Xt, M, return_detached_latent_distr=self.hparams.var_latent_STL)
        var_latent_distr = vae_forward_outputs.var_latent_distr
        var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
        Z = vae_forward_outputs.Z
        prior_distr = vae_forward_outputs.prior_distr
        generator_distr = vae_forward_outputs.generator_distr

        # Compute cross-entropy term of observed data
        generator_logprob_Z = generator_distr.log_prob(Xt)

        # Compute prior latent probability
        prior_logprob = prior_distr.log_prob(Z)
        prior_logprob = reduce(prior_logprob, 'z b k d -> z b k', 'sum')

        #
        # Compute the generator objective using CVI
        #

        # Compute the marginal log-probability of the generator
        generator_logprob_Z_marg = generator_logprob_Z*M
        generator_logprob_Z_marg = reduce(generator_logprob_Z_marg, 'z b k d -> z b k', 'sum')
        # generator_logprob_Z_marg = reduce(generator_logprob_Z_marg, 'z b k -> b k', 'mean')

        # NOTE: The prior logprob (if prior has no params) and the latent logprob are not needed,
        # so could be dropped to save a bit of computation!

        # Compute the mixture latent log-probability
        # Since the denominator has zero gradients wrt theta, we can use no_grad to same some graph computation
        with torch.no_grad():
            cvi_mixture_latent_logprob = latent_mixture_logprob(Z, var_latent_distr, var_comp_neg_idx=-2)

        # Compute the CVI objective
        generator_obj = generator_logprob_Z_marg + prior_logprob - cvi_mixture_latent_logprob
        generator_obj = reduce(generator_obj, 'z b k -> b k', 'mean')

        #
        # Compute the encoder objective for using MVB
        #

        # Compute the log-probability of the generator
        generator_logprob_Z = reduce(generator_logprob_Z, 'z b k d -> z b k', 'sum')
        generator_logprob_Z = reduce(generator_logprob_Z, 'z b k -> b k', 'mean')

        if self.hparams.kl_analytic:
            # Compute analytical -KL(q(z|x) || p(z)) term
            KL_neg = -torch.distributions.kl_divergence(var_latent_distr, prior_distr)
            KL_neg = reduce(KL_neg, 'b k d -> b k', 'sum')
        else:
            # # Compute prior latent probability
            # prior_logprob = prior_distr.log_prob(Z)
            # prior_logprob = reduce(prior_logprob, 'z b k d -> z b k', 'sum')

            if self.hparams.var_latent_STL:
                latent_logprob = var_latent_distr_detached.log_prob(Z)
            else:
                latent_logprob = var_latent_distr.log_prob(Z)
            latent_logprob = reduce(latent_logprob, 'z b k d -> z b k', 'sum')

            KL_neg = prior_logprob - latent_logprob
            KL_neg = reduce(KL_neg, 'z b k -> b k', 'mean')

        encoder_obj = KL_neg + generator_logprob_Z

        return generator_obj, encoder_obj

    def estimate_cvi_objective(self, Xt: torch.Tensor, M:torch.Tensor):
        """Compute the single CVI objective for fitting both the encoder and decoder."""
        # Forward pass over the VAE
        vae_forward_outputs = self.vae_forward(Xt, M, return_detached_latent_distr=self.hparams.var_latent_STL)
        var_latent_distr = vae_forward_outputs.var_latent_distr
        var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
        Z = vae_forward_outputs.Z
        prior_distr = vae_forward_outputs.prior_distr
        generator_distr = vae_forward_outputs.generator_distr

        # Compute the marginal log-probability of the generator
        generator_logprob_Z = generator_distr.log_prob(Xt)
        generator_logprob_Z_marg = generator_logprob_Z*M
        generator_logprob_Z_marg = reduce(generator_logprob_Z_marg, 'z b k d -> z b k', 'sum')
        # generator_logprob_Z_marg = reduce(generator_logprob_Z_marg, 'z b k -> b k', 'mean')

        # Compute prior latent probability
        prior_logprob = prior_distr.log_prob(Z)
        prior_logprob = reduce(prior_logprob, 'z b k d -> z b k', 'sum')

        # Compute the mixture latent log-probability
        # Use detached distribution for computing the score if using STL or DREG gradients
        assert not self.hparams.kl_analytic, 'Analytic KL not possible with the CVI objective'
        var_latent_distr_ = (var_latent_distr_detached if self.hparams.var_latent_STL
                             else var_latent_distr)
        cvi_mixture_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)

        # Compute the CVI objective
        cvi_obj = generator_logprob_Z_marg + prior_logprob - cvi_mixture_latent_logprob
        cvi_obj = reduce(cvi_obj, 'z b k -> b k', 'mean')

        return cvi_obj

    def estimate_mvb_objective(self, Xt: torch.Tensor, M: torch.Tensor):
        """Compute the single MVB objective for fitting both the encoder and decoder."""
        # Forward pass over the VAE
        vae_forward_outputs = self.vae_forward(Xt, M, return_detached_latent_distr=self.hparams.var_latent_STL)
        var_latent_distr = vae_forward_outputs.var_latent_distr
        var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
        Z = vae_forward_outputs.Z
        prior_distr = vae_forward_outputs.prior_distr
        generator_distr = vae_forward_outputs.generator_distr

        # Compute per data-point elbo, this just corresponds to the complete-data elbo (i.e. M=1)
        if self.hparams.kl_analytic:
            mvb = self.compute_elbo_w_analytic_kl(Xt, torch.ones_like(M), var_latent_distr, prior_distr, generator_distr)
        else:
            mvb = self.compute_elbo_w_montecarlokl(Xt, torch.ones_like(M), Z, var_latent_distr, var_latent_distr_detached, prior_distr, generator_distr)

        return mvb

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        # Update all imputations
        imputed_batch = self.train_imp_distribution.get_imputed_datapoints(batch)
        imputed_batch = self.update_imputations(imputed_batch, stage='train')
        # Sample a subset of imputations for optimisation
        imputed_batch = self.train_imp_distribution.get_imputed_datapoints(
                            batch, num_imputations=self.hparams.mvb_sample_num_imputations)

        Xt, M = imputed_batch[:2]
        if self.hparams.mvb_objective is MVB_OBJECTIVE['mvb2']:
            generator_obj, encoder_obj = self.estimate_mvb2_objectives(Xt, M)
            generator_obj = reduce(generator_obj, 'b k -> ', 'mean')
            encoder_obj = reduce(encoder_obj, 'b k -> ', 'mean')

            self.mvb2_manual_optimisation_step(-encoder_obj, -generator_obj)

            loss = -generator_obj

            self.log('mvb/train', encoder_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('cvi/train', generator_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        elif self.hparams.mvb_objective is MVB_OBJECTIVE['mvb']:
            mvb_obj = self.estimate_mvb_objective(Xt, M)
            mvb_obj = reduce(mvb_obj, 'b k -> ', 'mean')

            loss = -mvb_obj

            self.log('mvb/train', mvb_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        elif self.hparams.mvb_objective is MVB_OBJECTIVE['cvi']:
            cvi_obj = self.estimate_cvi_objective(Xt, M)
            cvi_obj = reduce(cvi_obj, 'b k -> ', 'mean')

            loss = -cvi_obj

            self.log('cvi/train', cvi_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        else:
            raise ValueError(f'Unknown MVB objective: {self.hparams.mvb_objective}')

        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        if self.hparams.mvb_update_val_imps:
            # Update all imputations
            imputed_batch = self.val_imp_distribution.get_imputed_datapoints(batch)
            imputed_batch = self.update_imputations(imputed_batch, stage='val')
            # Sample a subset of imputations for optimisation
            # imputed_batch = self.val_imp_distribution.get_imputed_datapoints(
            #                     batch, num_imputations=self.hparams.mvb_sample_num_imputations)

        if self.hparams.mvb_eval_val_loss:
            with torch.inference_mode():
                Xt, M = imputed_batch[:2]
                if self.hparams.mvb_objective is MVB_OBJECTIVE['mvb2']:
                    generator_obj, encoder_obj = self.estimate_mvb2_objectives(Xt, M)
                    generator_obj = reduce(generator_obj, 'b k -> ', 'mean')
                    encoder_obj = reduce(encoder_obj, 'b k -> ', 'mean')

                    loss = -generator_obj

                    self.log('mvb/val', encoder_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                    self.log('cvi/val', generator_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                elif self.hparams.mvb_objective is MVB_OBJECTIVE['mvb']:
                    mvb_obj = self.estimate_mvb_objective(Xt, M)
                    mvb_obj = reduce(mvb_obj, 'b k -> ', 'mean')

                    loss = -mvb_obj

                    self.log('mvb/val', mvb_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                elif self.hparams.mvb_objective is MVB_OBJECTIVE['cvi']:
                    cvi_obj = self.estimate_cvi_objective(Xt, M)
                    cvi_obj = reduce(cvi_obj, 'b k -> ', 'mean')

                    loss = -cvi_obj

                    self.log('cvi/val', cvi_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                else:
                    raise ValueError(f'Unknown MVB objective: {self.hparams.mvb_objective}')

                self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
