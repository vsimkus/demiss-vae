from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange, reduce
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vgiwae.shared.mvb2_manual_optimisation import mvb2_manual_optimisation_step
from vgiwae.shared.iwae import compute_smis_log_unnormalised_importance_weights, compute_dmis_log_unnormalised_importance_weights, latent_mixture_logprob

from .iwae import IWAE


class MVB_IWAE_OBJ(Enum):
    mvb_indep = auto() # Use MVB objective, treating each imputation particle separately
    mvb_all = auto() # PLACEHOLDER. Use MVB objective, treating all imputation particles together (NOT recommended, since it does not correspond to a valid bound)
    cvi_indep = auto() # Use CVI objective, treating each imputation particle separately
    cvi_all = auto() # Use CVI objective, treating all imputation particles together
    mvb2_encindep_decall = auto() # Use MVB objective (equivalent to mvb_indep) for phi and CVI objective (equivalent to cvi_all) for theta
    mvb2_encindep_decindep = auto() # Use MVB objective (equivalent to mvb_indep) for phi and CVI objective (equivalent to cvi_indep) for theta
    mvb2_encall_decall = auto() # Use mvb_all for the encoder and cvi_all for the decoder
    mvb2_encallweightedsmis_decall = auto() # Use gradient-weighted objective for encoder with SMIS weights and CVI for decoder
    mvb2_encallweighteddmmis_decall = auto() # Use gradient-weighted objective for encoder with DMMIS weights and CVI for decoder
    mvb2_all_encweighteddmmis_decweighteddmmis = auto() # Use gradient-weighted objective for encoder and decoder with DMMIS weights
    mvb2_all_encweightedsmis_decweightedsmis = auto() # PLACEHOLDER: Use gradient-weighted objective for encoder and decoder with SMIS weights
    mvb2_encallweighteddmmis_decallresampl = auto() # Use gradient-weighted objective for encoder with SMIS weights and CVI for decoder (with resampling)

class IS_WEIGHTS(Enum):
    smis = auto() # Standard importance sampling weights
    dmmis = auto() # Deterministic mixture importance sampling weights

class MVBIWAE(IWAE):
    """
    A IWAE model with missing data.
    Uses the marginalised variational bound.

    Args:
        mvb_sample_num_imputations:     If None, all imputations are used in the objective, if int, only a
                                        subset of all imputations is sampled and used in the objective
        mvb_update_val_imps:              If true, updates the validation imputations at the end of each epoch.
        mvb_eval_val_loss:              If true evaluates the validation loss (with imputation sampling),
                                        it can cause issues due to inference generalisation gap.
        enc_isweights:                  Which importance sampling weights to use for encoder.
        dec_isweights:                  Which importance sampling weights to use for decoder.
        mvb_objective:                  Which objective to use for generator/encoder.
        mvb2_optim_gradclip_val:        Gradient clipping value for MVB with two objectives.
        mvb2_optim_gradclip_alg:        Gradient clipping method for MVB with two objectives.
    """

    def __init__(self,
                 *args,
                 mvb_sample_num_imputations: int = None,
                 mvb_update_val_imps: bool = True,
                 mvb_eval_val_loss: bool = False,
                 enc_isweights: IS_WEIGHTS = IS_WEIGHTS['dmmis'],
                 dec_isweights: IS_WEIGHTS = IS_WEIGHTS['dmmis'],
                 mvb_objective: MVB_IWAE_OBJ = MVB_IWAE_OBJ['mvb2_encindep_decall'],
                 mvb2_optim_gradclip_val: float = 0,
                 mvb2_optim_gradclip_alg: str = 'norm',
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encindep_decindep,
                                          MVB_IWAE_OBJ.mvb2_encindep_decall,
                                          MVB_IWAE_OBJ.mvb2_encall_decall,
                                          MVB_IWAE_OBJ.mvb2_encallweightedsmis_decall,
                                          MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decall,
                                          MVB_IWAE_OBJ.mvb2_all_encweighteddmmis_decweighteddmmis,
                                          MVB_IWAE_OBJ.mvb2_all_encweightedsmis_decweightedsmis,
                                          MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decallresampl,):
            # NOTE: this disables PytorchLightning's automatic optimisation,
            # hence you must handle optimisation manually in training_step.
            # This also means that some of PL's features won't work, such as
            # gradient clipping, gradient accumulation, learning rate scheduling, etc.
            # See https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html
            # or https://pytorch-lightning.readthedocs.io/en/1.5.8/common/lightning_module.html?highlight=automatic_optimization#automatic-optimization
            self.automatic_optimization = False

        if self.hparams.mvb_objective in (MVB_IWAE_OBJ['mvb_indep'], MVB_IWAE_OBJ['mvb_all'],
                                          MVB_IWAE_OBJ['cvi_indep'], MVB_IWAE_OBJ['cvi_all'],
                                          MVB_IWAE_OBJ['mvb2_all_encweighteddmmis_decweighteddmmis'],
                                          MVB_IWAE_OBJ['mvb2_all_encweightedsmis_decweightedsmis'],):
            assert self.hparams.enc_isweights is self.hparams.dec_isweights, 'Encoder and decoder should use the same importance weights.'

    mvb2_manual_optimisation_step = mvb2_manual_optimisation_step

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.resample_imputations(stage='train')

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self.resample_imputations(stage='val')

    def log_weights_to_objective(self, log_weights, Z, *, objective_type: str):
        # Compute the total IWELBO term
        if objective_type == 'all':
            # NOTE: here I treat i*k as the number of importance samples
            log_weights = rearrange(log_weights, 'z i b k -> z (i k) b')

            def create_Z_hook(norm_weights):
                if Z.requires_grad:
                    norm_weights = rearrange(norm_weights, 'z (i k) b -> z i b k 1', i=Z.shape[1], k=Z.shape[3])
                    Z.register_hook(lambda grad: norm_weights * grad)

            iwelbo = self.log_weights_to_iwelbo(log_weights, create_Z_hook=create_Z_hook)
            iwelbo = reduce(iwelbo, 'z b -> b', 'mean')
        elif objective_type == 'indep':
            # NOTE: here I treat i as the number of importance samples
            def create_Z_hook(norm_weights):
                if Z.requires_grad:
                    Z.register_hook(lambda grad: norm_weights * grad)

            iwelbo = self.log_weights_to_iwelbo(log_weights, create_Z_hook=create_Z_hook)
            # NOTE: avg over z _and_ k
            iwelbo = reduce(iwelbo, 'z b k -> b', 'mean')
        else:
            raise NotImplementedError()

        return iwelbo

    def compute_iwelbo(self,
                       X: torch.Tensor,
                       M: torch.Tensor,
                       Z: torch.Tensor,
                       var_latent_distr: torch.distributions.Distribution,
                       var_latent_distr_detached: torch.distributions.Distribution,
                       prior_distr: torch.distributions.Distribution,
                       generator_distr: torch.distributions.Distribution) -> torch.Tensor:
        # NOTE: Override of the original, includes handling of k and i together

        # Compute the unnormalised importance weights
        var_latent_distr_ = (var_latent_distr_detached if self.hparams.var_latent_STL or self.hparams.var_latent_DREG
                             else var_latent_distr)
        assert self.hparams.enc_isweights is self.hparams.dec_isweights, 'Encoder and decoder should use the same importance weights.'
        is_weights = self.hparams.enc_isweights
        if is_weights is IS_WEIGHTS.smis:
            log_weights = compute_smis_log_unnormalised_importance_weights(X, M, Z,
                                                                           var_latent_distr_,
                                                                           prior_distr,
                                                                           generator_distr)
        elif is_weights is IS_WEIGHTS.dmmis:
            log_weights = compute_dmis_log_unnormalised_importance_weights(X, M, Z,
                                                                           var_latent_distr=var_latent_distr_,
                                                                           var_comp_neg_idx=-2,
                                                                           prior_distr=prior_distr,
                                                                           generator_distr=generator_distr,
                                                                           )
        else:
            raise NotImplementedError()

        if self.hparams.mvb_objective is MVB_IWAE_OBJ.mvb_indep or self.hparams.mvb_objective is MVB_IWAE_OBJ.cvi_indep:
            objective_type = 'indep'
        elif self.hparams.mvb_objective is MVB_IWAE_OBJ.mvb_all or self.hparams.mvb_objective is MVB_IWAE_OBJ.cvi_all:
            objective_type = 'all'
        else:
            raise NotImplementedError()

        return self.log_weights_to_objective(log_weights, Z, objective_type=objective_type)

    def compute_iwelbos_for_decoder_and_encoder(self,
                                                X: torch.Tensor,
                                                M: torch.Tensor,
                                                Z: torch.Tensor,
                                                var_latent_distr: torch.distributions.Distribution,
                                                var_latent_distr_detached: torch.distributions.Distribution,
                                                prior_distr: torch.distributions.Distribution,
                                                generator_distr: torch.distributions.Distribution):
        """
        NOTE: you shouldn't optimise this objective naively on all parameters.
        Instead see the code in `self.mvb2_manual_optimisation_step()`, which ensures that you optimise
        theta using CVI objective and phi using MVB objective.
        """

        # Compute cross-entropy term of observed data
        generator_logprob = generator_distr.log_prob(X)

        # Compute prior latent probability
        prior_logprob = prior_distr.log_prob(Z)
        prior_logprob = reduce(prior_logprob, '... d -> ...', 'sum')

        #
        # Compute the generator objective using CVI
        #

        # Compute the marginal log-probability of the generator
        generator_logprob_marg = generator_logprob*M
        generator_logprob_marg = reduce(generator_logprob_marg, '... d -> ...', 'sum')

        # Compute the log-prob of samples under the latent distribution
        # Can use the detached var distribution even for the generator objective
        var_latent_distr_ = (var_latent_distr_detached if self.hparams.var_latent_STL or self.hparams.var_latent_DREG
                             else var_latent_distr)
        if self.hparams.dec_isweights is IS_WEIGHTS.smis:
            dec_latent_logprob = var_latent_distr_.log_prob(Z)
            dec_latent_logprob = reduce(dec_latent_logprob, '... d -> ...', 'sum')
        elif self.hparams.dec_isweights is IS_WEIGHTS.dmmis:
            dec_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)
        else:
            raise NotImplementedError()

        # Compute generator log-weights
        log_weights_marg = generator_logprob_marg + prior_logprob - dec_latent_logprob

        # Compute generator objective
        if self.hparams.mvb_objective is MVB_IWAE_OBJ.mvb2_encindep_decindep:
            dec_objective_type = 'indep'
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encindep_decall, MVB_IWAE_OBJ.mvb2_encall_decall):
            dec_objective_type = 'all'
        else:
            raise NotImplementedError()

        iwelbo_marg = self.log_weights_to_objective(log_weights_marg, Z, objective_type=dec_objective_type)
        generator_obj = iwelbo_marg

        #
        # Compute the objective for the encoder using MVB
        #
        generator_logprob = reduce(generator_logprob, '... d -> ...', 'sum')

        # Compute the log-prob of samples under the latent distribution
        # NOTE: using var_latent_distr_ here (params need to be detached for STL and DREG)
        if self.hparams.enc_isweights is self.hparams.dec_isweights:
            # If the weights are same for dec and enc, then we can reuse the denominator
            enc_latent_logprob = dec_latent_logprob
        elif self.hparams.enc_isweights is IS_WEIGHTS.smis:
            enc_latent_logprob = var_latent_distr_.log_prob(Z)
            enc_latent_logprob = reduce(enc_latent_logprob, '... d -> ...', 'sum')
        elif self.hparams.enc_isweights is IS_WEIGHTS.dmmis:
            enc_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)
        else:
            raise NotImplementedError()

        # Compute encoder log-weights
        log_weights = generator_logprob + prior_logprob - enc_latent_logprob

        # Compute encoder objective
        if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encindep_decindep,
                                          MVB_IWAE_OBJ.mvb2_encindep_decall):
            enc_objective_type = 'indep'
        elif self.hparams.mvb_objective is MVB_IWAE_OBJ.mvb2_encall_decall:
            enc_objective_type = 'all'
        else:
            raise NotImplementedError()

        iwelbo = self.log_weights_to_objective(log_weights, Z, objective_type=enc_objective_type)
        encoder_obj = iwelbo

        return generator_obj, encoder_obj

    def compute_iwelbo_for_decoder_and_weighted_gradient_objective_encoder(self,
                                                X: torch.Tensor,
                                                M: torch.Tensor,
                                                Z: torch.Tensor,
                                                var_latent_distr: torch.distributions.Distribution,
                                                var_latent_distr_detached: torch.distributions.Distribution,
                                                prior_distr: torch.distributions.Distribution,
                                                generator_distr: torch.distributions.Distribution):
        """
        NOTE: you shouldn't optimise this objective naively on all parameters.
        Instead see the code in `self.mvb2_manual_optimisation_step()`, which ensures that you optimise
        theta using CVI objective and phi using MVB objective.
        """

        # Compute cross-entropy term of observed data
        generator_logprob = generator_distr.log_prob(X)

        # Compute prior latent probability
        prior_logprob = prior_distr.log_prob(Z)
        prior_logprob = reduce(prior_logprob, '... d -> ...', 'sum')

        #
        # Compute the generator objective using CVI
        #

        # Compute the marginal log-probability of the generator
        generator_logprob_marg = generator_logprob*M
        generator_logprob_marg = reduce(generator_logprob_marg, '... d -> ...', 'sum')

        # Compute the log-prob of samples under the latent distribution
        # Can use the detached var distribution even for the generator objective
        var_latent_distr_ = (var_latent_distr_detached if self.hparams.var_latent_STL or self.hparams.var_latent_DREG
                             else var_latent_distr)
        if self.hparams.dec_isweights is IS_WEIGHTS.smis:
            dec_latent_logprob = var_latent_distr_.log_prob(Z)
            dec_latent_logprob = reduce(dec_latent_logprob, '... d -> ...', 'sum')
        elif self.hparams.dec_isweights is IS_WEIGHTS.dmmis:
            dec_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)
        else:
            raise NotImplementedError()

        # Compute generator log-weights
        log_weights_marg = generator_logprob_marg + prior_logprob - dec_latent_logprob

        # Compute generator objective
        if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encallweightedsmis_decall, MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decall):
            dec_objective_type = 'all'
            iwelbo_marg = self.log_weights_to_objective(log_weights_marg, Z, objective_type=dec_objective_type)
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decallresampl,):
            # Use importance-resampled alternative to IWELBO (see Cremer "Reinterpreting Importance weighted autoencoders")
            K, I = log_weights_marg.shape[-1], log_weights_marg.shape[1]
            log_weights_marg_rearranged = rearrange(log_weights_marg, 'z i b k -> z b (i k)')

            importance_distr = torch.distributions.Categorical(logits=log_weights_marg_rearranged)
            idx = importance_distr.sample((K*I,))
            log_weights_marg_rearranged_resampled = log_weights_marg_rearranged[torch.arange(log_weights_marg_rearranged.shape[0]),
                                                                                torch.arange(log_weights_marg_rearranged.shape[1]),
                                                                                idx]
            iwelbo_marg = reduce(log_weights_marg_rearranged_resampled, 'ik z b -> b', 'mean')

        else:
            raise NotImplementedError()

        generator_obj = iwelbo_marg

        #
        # Compute the objective for the encoder using MVB
        #
        generator_logprob = reduce(generator_logprob, '... d -> ...', 'sum')

        # Compute the log-prob of samples under the latent distribution
        # NOTE: using var_latent_distr_ here (params need to be detached for STL and DREG)
        if self.hparams.enc_isweights is self.hparams.dec_isweights:
            # If the weights are same for dec and enc, then we can reuse the denominator
            enc_latent_logprob = dec_latent_logprob
        elif self.hparams.enc_isweights is IS_WEIGHTS.smis:
            enc_latent_logprob = var_latent_distr_.log_prob(Z)
            enc_latent_logprob = reduce(enc_latent_logprob, '... d -> ...', 'sum')
        elif self.hparams.enc_isweights is IS_WEIGHTS.dmmis:
            enc_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)
        else:
            raise NotImplementedError()

        # Compute encoder log-weights
        log_weights = generator_logprob + prior_logprob - enc_latent_logprob

        # Compute the unnormalised log-weights for re-weighting the gradients of the log_weights ratios above.
        gradient_weight_enc_latent_logprob = None
        if self.hparams.mvb_objective is MVB_IWAE_OBJ.mvb2_encallweightedsmis_decall:
            # Use SMIS denominator. NOTE: the below attempts to reuse the computations if they have already been done.
            if self.hparams.enc_isweights is IS_WEIGHTS.smis:
                gradient_weight_enc_latent_logprob = enc_latent_logprob
            elif self.hparams.dec_isweights is IS_WEIGHTS.smis:
                gradient_weight_enc_latent_logprob = dec_latent_logprob
            else:
                # If SMIS denominator was not yet computed above, then compute it now.
                gradient_weight_enc_latent_logprob = var_latent_distr_.log_prob(Z)
                gradient_weight_enc_latent_logprob = reduce(gradient_weight_enc_latent_logprob, '... d -> ...', 'sum')
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decall, MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decallresampl):
            # Use DMMIS denominator. NOTE: the below attempts to reuse the computations if they have already been done.
            if self.hparams.enc_isweights is IS_WEIGHTS.dmmis:
                gradient_weight_enc_latent_logprob = enc_latent_logprob
            elif self.hparams.dec_isweights is IS_WEIGHTS.dmmis:
                gradient_weight_enc_latent_logprob = dec_latent_logprob
            else:
                # If DMMIS denominator was not yet computed above, then compute it now.
                gradient_weight_enc_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)

        gradient_log_weights = generator_logprob_marg + prior_logprob - gradient_weight_enc_latent_logprob

        # Compute the weighted-gradent objective for the encoder
        log_weights = gradient_log_weights.detach() - log_weights.detach() + log_weights

        if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encallweightedsmis_decall,
                                          MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decall,
                                          MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decallresampl):
            enc_objective_type = 'all'
        else:
            raise NotImplementedError()

        # encoder_obj = self.log_weights_to_objective(log_weights, Z, objective_type=enc_objective_type)
        # Compute the total IWELBO term
        if enc_objective_type == 'all':
            # NOTE: here I treat i*k as the number of importance samples
            log_weights = rearrange(log_weights, 'z i b k -> z (i k) b')

            encoder_obj = self.log_weights_to_iwelbo(log_weights, create_Z_hook=None)
            encoder_obj = reduce(encoder_obj, 'z b -> b', 'mean')

            # NOTE: DREG gradients for this objective are a little different from the original.
            # We need to reparametrise the score function gradient here and add it as extra term.
            if self.hparams.var_latent_DREG:
                gradient_log_weights = rearrange(gradient_log_weights, 'z i b k -> z (i k) b')
                norm_gradient_weights = torch.softmax(gradient_log_weights, dim=1)
                one_minus_norm_gradient_weights = 1 - norm_gradient_weights

                variance_reduced_scorefn_gradients = ((one_minus_norm_gradient_weights * norm_gradient_weights).detach() * gradient_log_weights).sum(dim=1)
                variance_reduced_scorefn_gradients = reduce(variance_reduced_scorefn_gradients, 'z b -> b', 'mean')

                encoder_obj = encoder_obj - variance_reduced_scorefn_gradients + variance_reduced_scorefn_gradients.detach()

        elif enc_objective_type == 'indep':
            # NOTE: here I treat i as the number of importance samples
            encoder_obj = self.log_weights_to_iwelbo(log_weights, create_Z_hook=None)
            # NOTE: avg over z _and_ k
            encoder_obj = reduce(encoder_obj, 'z b k -> b', 'mean')

            # NOTE: DREG gradients for this objective are a little different from the original.
            # We need to reparametrise the score function gradient here and add it as extra term.
            if self.hparams.var_latent_DREG:
                gradient_log_weights = rearrange(gradient_log_weights, 'z i b k -> z i k b')
                norm_gradient_weights = torch.softmax(gradient_log_weights, dim=1)
                one_minus_norm_gradient_weights = 1 - norm_gradient_weights

                variance_reduced_scorefn_gradients = ((one_minus_norm_gradient_weights * norm_gradient_weights).detach() * gradient_log_weights).sum(dim=1)
                variance_reduced_scorefn_gradients = reduce(variance_reduced_scorefn_gradients, 'z b k -> b', 'mean')

                encoder_obj = encoder_obj - variance_reduced_scorefn_gradients + variance_reduced_scorefn_gradients.detach()
        else:
            raise NotImplementedError()

        return generator_obj, encoder_obj

    def compute_iwelbo_with_weighted_gradient_objective_for_decoder_and_encoder(self,
                                                X: torch.Tensor,
                                                M: torch.Tensor,
                                                Z: torch.Tensor,
                                                var_latent_distr: torch.distributions.Distribution,
                                                var_latent_distr_detached: torch.distributions.Distribution,
                                                prior_distr: torch.distributions.Distribution,
                                                generator_distr: torch.distributions.Distribution):
        """
        NOTE: you shouldn't optimise this objective naively on all parameters.
        Instead see the code in `self.mvb2_manual_optimisation_step()`, which ensures that you optimise
        theta using CVI objective and phi using MVB objective.
        """

        # Compute cross-entropy term of observed data
        generator_logprob = generator_distr.log_prob(X)

        # Compute prior latent probability
        prior_logprob = prior_distr.log_prob(Z)
        prior_logprob = reduce(prior_logprob, '... d -> ...', 'sum')

        #
        # Compute the generator objective using CVI
        #

        # Compute the marginal log-probability of the generator
        generator_logprob_marg = generator_logprob*M
        generator_logprob_marg = reduce(generator_logprob_marg, '... d -> ...', 'sum')

        # Compute the log-prob of samples under the latent distribution
        # Can use the detached var distribution even for the generator objective
        var_latent_distr_ = (var_latent_distr_detached if self.hparams.var_latent_STL or self.hparams.var_latent_DREG
                             else var_latent_distr)
        if self.hparams.dec_isweights is IS_WEIGHTS.smis:
            dec_latent_logprob = var_latent_distr_.log_prob(Z)
            dec_latent_logprob = reduce(dec_latent_logprob, '... d -> ...', 'sum')
        elif self.hparams.dec_isweights is IS_WEIGHTS.dmmis:
            dec_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)
        else:
            raise NotImplementedError()

        # # Compute generator log-weights
        # log_weights_marg = generator_logprob_marg + prior_logprob - dec_latent_logprob

        # # Compute generator objective
        # if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encallweightedsmis_decall, MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decall):
        #     dec_objective_type = 'all'
        # else:
        #     raise NotImplementedError()

        # iwelbo_marg = self.log_weights_to_objective(log_weights_marg, Z, objective_type=dec_objective_type)
        # generator_obj = iwelbo_marg

        #
        # Compute the objective for the encoder using MVB
        #
        generator_logprob = reduce(generator_logprob, '... d -> ...', 'sum')

        # Compute the log-prob of samples under the latent distribution
        # NOTE: using var_latent_distr_ here (params need to be detached for STL and DREG)
        if self.hparams.enc_isweights is self.hparams.dec_isweights:
            # If the weights are same for dec and enc, then we can reuse the denominator
            enc_latent_logprob = dec_latent_logprob
        elif self.hparams.enc_isweights is IS_WEIGHTS.smis:
            enc_latent_logprob = var_latent_distr_.log_prob(Z)
            enc_latent_logprob = reduce(enc_latent_logprob, '... d -> ...', 'sum')
        elif self.hparams.enc_isweights is IS_WEIGHTS.dmmis:
            enc_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)
        else:
            raise NotImplementedError()

        # Compute encoder log-weights
        log_weights = generator_logprob + prior_logprob - enc_latent_logprob

        # Compute the unnormalised log-weights for re-weighting the gradients of the log_weights ratios above.
        gradient_weight_enc_latent_logprob = None
        if self.hparams.mvb_objective is MVB_IWAE_OBJ.mvb2_all_encweightedsmis_decweightedsmis:
            # Use SMIS denominator. NOTE: the below attempts to reuse the computations if they have already been done.
            if self.hparams.enc_isweights is IS_WEIGHTS.smis:
                gradient_weight_enc_latent_logprob = enc_latent_logprob
            elif self.hparams.dec_isweights is IS_WEIGHTS.smis:
                gradient_weight_enc_latent_logprob = dec_latent_logprob
            else:
                # If SMIS denominator was not yet computed above, then compute it now.
                gradient_weight_enc_latent_logprob = var_latent_distr_.log_prob(Z)
                gradient_weight_enc_latent_logprob = reduce(gradient_weight_enc_latent_logprob, '... d -> ...', 'sum')
        elif self.hparams.mvb_objective is MVB_IWAE_OBJ.mvb2_all_encweighteddmmis_decweighteddmmis:
            # Use DMMIS denominator. NOTE: the below attempts to reuse the computations if they have already been done.
            if self.hparams.enc_isweights is IS_WEIGHTS.dmmis:
                gradient_weight_enc_latent_logprob = enc_latent_logprob
            elif self.hparams.dec_isweights is IS_WEIGHTS.dmmis:
                gradient_weight_enc_latent_logprob = dec_latent_logprob
            else:
                # If DMMIS denominator was not yet computed above, then compute it now.
                gradient_weight_enc_latent_logprob = latent_mixture_logprob(Z, var_latent_distr_, var_comp_neg_idx=-2)

        gradient_log_weights = generator_logprob_marg + prior_logprob - gradient_weight_enc_latent_logprob

        # Compute the weighted-gradent objective for the encoder
        log_weights = gradient_log_weights.detach() - log_weights.detach() + log_weights

        if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_all_encweighteddmmis_decweighteddmmis,):
            enc_objective_type = 'all'
        else:
            raise NotImplementedError()

        # encoder_obj = self.log_weights_to_objective(log_weights, Z, objective_type=enc_objective_type)
        # Compute the total IWELBO term
        if enc_objective_type == 'all':
            # NOTE: here I treat i*k as the number of importance samples
            log_weights = rearrange(log_weights, 'z i b k -> z (i k) b')

            encoder_obj = self.log_weights_to_iwelbo(log_weights, create_Z_hook=None)
            encoder_obj = reduce(encoder_obj, 'z b -> b', 'mean')
            generator_obj = encoder_obj

            # NOTE: DREG gradients for this objective are a little different from the original.
            # We need to reparametrise the score function gradient here and add it as extra term.
            if self.hparams.var_latent_DREG:
                gradient_log_weights = rearrange(gradient_log_weights, 'z i b k -> z (i k) b')
                norm_gradient_weights = torch.softmax(gradient_log_weights, dim=1)
                one_minus_norm_gradient_weights = 1 - norm_gradient_weights

                variance_reduced_scorefn_gradients = ((one_minus_norm_gradient_weights * norm_gradient_weights).detach() * gradient_log_weights).sum(dim=1)
                variance_reduced_scorefn_gradients = reduce(variance_reduced_scorefn_gradients, 'z b -> b', 'mean')

                encoder_obj = encoder_obj - variance_reduced_scorefn_gradients + variance_reduced_scorefn_gradients.detach()

        elif enc_objective_type == 'indep':
            # NOTE: here I treat i as the number of importance samples
            encoder_obj = self.log_weights_to_iwelbo(log_weights, create_Z_hook=None)
            # NOTE: avg over z _and_ k
            encoder_obj = reduce(encoder_obj, 'z b k -> b', 'mean')
            generator_obj = encoder_obj

            # NOTE: DREG gradients for this objective are a little different from the original.
            # We need to reparametrise the score function gradient here and add it as extra term.
            if self.hparams.var_latent_DREG:
                gradient_log_weights = rearrange(gradient_log_weights, 'z i b k -> z i k b')
                norm_gradient_weights = torch.softmax(gradient_log_weights, dim=1)
                one_minus_norm_gradient_weights = 1 - norm_gradient_weights

                variance_reduced_scorefn_gradients = ((one_minus_norm_gradient_weights * norm_gradient_weights).detach() * gradient_log_weights).sum(dim=1)
                variance_reduced_scorefn_gradients = reduce(variance_reduced_scorefn_gradients, 'z b k -> b', 'mean')

                encoder_obj = encoder_obj - variance_reduced_scorefn_gradients + variance_reduced_scorefn_gradients.detach()
        else:
            raise NotImplementedError()

        return generator_obj, encoder_obj

    def estimate_mvb2_objectives(self, Xt: torch.Tensor, M: torch.Tensor):
        """
        Compute the MVB objective for fitting the encoder, and the CVI objective for fitting the decoder.

        NOTE: you shouldn't optimise this objective naively on all parameters.
        Instead see the code in `self.mvb2_manual_optimisation_step()`, which ensures that you optimise
        theta using CVI objective and phi using MVB objective.
        """
        # Forward pass over the VAE
        vae_forward_outputs = self.vae_forward(Xt, M,
                                               return_detached_latent_distr=self.hparams.var_latent_STL or self.hparams.var_latent_DREG,
                                               Z_sample_shape=(self.hparams.num_latent_samples, self.hparams.num_importance_samples,))
        var_latent_distr = vae_forward_outputs.var_latent_distr
        var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
        Z = vae_forward_outputs.Z
        prior_distr = vae_forward_outputs.prior_distr
        generator_distr = vae_forward_outputs.generator_distr
        # NOTE: Shape of Z is (z, i, b, k, d)

        if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encindep_decall,
                                          MVB_IWAE_OBJ.mvb2_encindep_decindep,
                                          MVB_IWAE_OBJ.mvb2_encall_decall,):
            generator_obj, encoder_obj = self.compute_iwelbos_for_decoder_and_encoder(Xt, M, Z,
                                                                                    var_latent_distr,
                                                                                    var_latent_distr_detached,
                                                                                    prior_distr,
                                                                                    generator_distr)
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encallweightedsmis_decall,
                                            MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decall,
                                            MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decallresampl,):
            generator_obj, encoder_obj = self.compute_iwelbo_for_decoder_and_weighted_gradient_objective_encoder(Xt, M, Z,
                                                                                                                 var_latent_distr,
                                                                                                                 var_latent_distr_detached,
                                                                                                                 prior_distr,
                                                                                                                 generator_distr)
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_all_encweighteddmmis_decweighteddmmis,
                                            MVB_IWAE_OBJ.mvb2_all_encweightedsmis_decweightedsmis,):
            generator_obj, encoder_obj = self.compute_iwelbo_with_weighted_gradient_objective_for_decoder_and_encoder(Xt, M, Z,
                                                                                                                      var_latent_distr,
                                                                                                                      var_latent_distr_detached,
                                                                                                                      prior_distr,
                                                                                                                      generator_distr)
        else:
            raise NotImplementedError()

        return generator_obj, encoder_obj

    def estimate_cvi_objective(self, Xt: torch.Tensor, M: torch.Tensor):
        """Compute the CVI objective that uses a single objective for the encoder and decoder."""
        # Forward pass over the VAE
        vae_forward_outputs = self.vae_forward(Xt, M,
                                               return_detached_latent_distr=self.hparams.var_latent_STL or self.hparams.var_latent_DREG,
                                               Z_sample_shape=(self.hparams.num_latent_samples, self.hparams.num_importance_samples,))
        var_latent_distr = vae_forward_outputs.var_latent_distr
        var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
        Z = vae_forward_outputs.Z
        prior_distr = vae_forward_outputs.prior_distr
        generator_distr = vae_forward_outputs.generator_distr
        # NOTE: Shape of Z is (z, i, b, k, d)

        # Compute per data-point elbo, this just corresponds to the complete-data elbo (i.e. M=1)
        # NOTE: here I use the overriden implementation above
        cvi = self.compute_iwelbo(Xt, M, Z, var_latent_distr, var_latent_distr_detached, prior_distr, generator_distr)

        return cvi

    def estimate_mvb_objective(self, Xt: torch.Tensor, M: torch.Tensor):
        """Compute the MVB objective that uses a single objective for the encoder and decoder."""
        # Forward pass over the VAE
        vae_forward_outputs = self.vae_forward(Xt, M,
                                               return_detached_latent_distr=self.hparams.var_latent_STL or self.hparams.var_latent_DREG,
                                               Z_sample_shape=(self.hparams.num_latent_samples, self.hparams.num_importance_samples,))
        var_latent_distr = vae_forward_outputs.var_latent_distr
        var_latent_distr_detached = vae_forward_outputs.var_latent_distr_detached
        Z = vae_forward_outputs.Z
        prior_distr = vae_forward_outputs.prior_distr
        generator_distr = vae_forward_outputs.generator_distr
        # NOTE: Shape of Z is (z, i, b, k, d)

        # Compute per data-point elbo, this just corresponds to the complete-data elbo (i.e. M=1)
        # NOTE: here I use the overriden implementation above
        mvb = self.compute_iwelbo(Xt, torch.ones_like(M), Z, var_latent_distr, var_latent_distr_detached, prior_distr, generator_distr)

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
        if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encindep_decindep,
                                          MVB_IWAE_OBJ.mvb2_encindep_decall,
                                          MVB_IWAE_OBJ.mvb2_encall_decall,):
            generator_obj, encoder_obj = self.estimate_mvb2_objectives(Xt, M)
            generator_obj = reduce(generator_obj, 'b -> ', 'mean')
            encoder_obj = reduce(encoder_obj, 'b -> ', 'mean')

            self.mvb2_manual_optimisation_step(-encoder_obj, -generator_obj)

            loss = -generator_obj

            self.log('mvb/train', encoder_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('cvi/train', generator_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encallweightedsmis_decall,
                                            MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decall,):
            generator_obj, encoder_obj = self.estimate_mvb2_objectives(Xt, M)
            generator_obj = reduce(generator_obj, 'b -> ', 'mean')
            encoder_obj = reduce(encoder_obj, 'b -> ', 'mean')

            self.mvb2_manual_optimisation_step(-encoder_obj, -generator_obj)

            loss = -generator_obj

            self.log('mvb_gradientweightedobj/train', encoder_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('cvi/train', generator_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_all_encweighteddmmis_decweighteddmmis,
                                            MVB_IWAE_OBJ.mvb2_all_encweightedsmis_decweightedsmis,
                                            MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decallresampl,):
            generator_obj, encoder_obj = self.estimate_mvb2_objectives(Xt, M)
            generator_obj = reduce(generator_obj, 'b -> ', 'mean')
            encoder_obj = reduce(encoder_obj, 'b -> ', 'mean')

            self.mvb2_manual_optimisation_step(-encoder_obj, -generator_obj)

            loss = -generator_obj

            self.log('mvb_gradientweightedobj/train', encoder_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            # self.log('cvi/train', generator_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb_indep,
                                            MVB_IWAE_OBJ.mvb_all,):
            mvb_obj = self.estimate_mvb_objective(Xt, M)
            mvb_obj = reduce(mvb_obj, 'b -> ', 'mean')

            loss = -mvb_obj

            self.log('mvb/train', mvb_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.cvi_indep,
                                            MVB_IWAE_OBJ.cvi_all,):
            cvi_obj = self.estimate_cvi_objective(Xt, M)
            cvi_obj = reduce(cvi_obj, 'b -> ', 'mean')

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
                if self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encindep_decindep,
                                                  MVB_IWAE_OBJ.mvb2_encindep_decall,
                                                  MVB_IWAE_OBJ.mvb2_encall_decall,):
                    generator_obj, encoder_obj = self.estimate_mvb2_objectives(Xt, M)
                    generator_obj = reduce(generator_obj, 'b -> ', 'mean')
                    encoder_obj = reduce(encoder_obj, 'b -> ', 'mean')

                    loss = -generator_obj

                    self.log('mvb/val', encoder_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                    self.log('cvi/val', generator_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_encallweightedsmis_decall,
                                                    MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decall,):
                    generator_obj, encoder_obj = self.estimate_mvb2_objectives(Xt, M)
                    generator_obj = reduce(generator_obj, 'b -> ', 'mean')
                    encoder_obj = reduce(encoder_obj, 'b -> ', 'mean')

                    loss = -generator_obj

                    self.log('mvb_gradientweightedobj/val', encoder_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                    self.log('cvi/val', generator_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb2_all_encweighteddmmis_decweighteddmmis,
                                                    MVB_IWAE_OBJ.mvb2_all_encweightedsmis_decweightedsmis,
                                                    MVB_IWAE_OBJ.mvb2_encallweighteddmmis_decallresampl,):
                    generator_obj, encoder_obj = self.estimate_mvb2_objectives(Xt, M)
                    generator_obj = reduce(generator_obj, 'b -> ', 'mean')
                    encoder_obj = reduce(encoder_obj, 'b -> ', 'mean')

                    loss = -generator_obj

                    self.log('mvb_gradientweightedobj/val', encoder_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                    # self.log('cvi/val', generator_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.mvb_indep,
                                                    MVB_IWAE_OBJ.mvb_all,):
                    mvb_obj = self.estimate_mvb_objective(Xt, M)
                    mvb_obj = reduce(mvb_obj, 'b -> ', 'mean')

                    loss = -mvb_obj

                    self.log('mvb/val', mvb_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                elif self.hparams.mvb_objective in (MVB_IWAE_OBJ.cvi_indep,
                                                    MVB_IWAE_OBJ.cvi_all,):
                    cvi_obj = self.estimate_cvi_objective(Xt, M)
                    cvi_obj = reduce(cvi_obj, 'b -> ', 'mean')

                    loss = -cvi_obj

                    self.log('cvi/val', cvi_obj, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                else:
                    raise ValueError(f'Unknown MVB objective: {self.hparams.mvb_objective}')

                self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
