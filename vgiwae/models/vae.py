from collections import namedtuple
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, reduce
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vgiwae.overrides.continuous_bernoulli import ContinuousBernoulliPatched
from vgiwae.utils.base_imputer_module import ImputerModuleBase
from vgiwae.utils.estimate_total_train_steps import PLEstimateTotalTrainSteps
from vgiwae.shared.vae_enums import DISTRIBUTION, PRIOR_DISTRIBUTION, EPSILON
from vgiwae.shared.neural_nets import FullyConnectedNetwork, ResidualFCNetwork
from vgiwae.shared.stratified_mixture_same_family import StratifiedMixtureSameFamily
from vgiwae.shared.reparametrized_mixture_same_family import ReparametrizedMixtureSameFamily
from vgiwae.shared.bernoulli_between_m1_and_p1 import Bernoulli_m1_p1
from vgiwae.shared.vae_resnet import ResNetEncoder, ResNetDecoder

VAEForwardOutputs = namedtuple('VAEForwardOutputs',
                               ['var_latent_distr',
                                'var_latent_distr_detached',
                                'Z',
                                'prior_distr',
                                'generator_distr'])

class VAE(ImputerModuleBase, PLEstimateTotalTrainSteps):
    """
    A VAE model with missing data.

    Args:
        generator_network:          The neural network of the generator.
        generator_distribution:     The distribution of the generator.
        var_latent_network:         The neural network of the variational latent network.
        var_latent_distribution:    The distribution of the variational latents.
        prior_distribution:         The prior distribution of the latents.
        encoder_use_mis_mask:       if true appends the missingness mask to the encoder.
        var_latent_STL:             if true uses the gradients from "Sticking the Landing" by Roeder et al. 2017
        num_latent_samples:         The number of samples used in Monte Carlo averaging of the ELBO
        kl_analytic:                If true the KL term is computed analytically.

        lr_generator:               learning rate of the generator model
        amsgrad_generator:          if true use AMSGrad version of Adam for the generator model

        lr_latent:                  learning rate of the latent model
        amsgrad_latent:             if true use AMSGrad version of Adam for the latent model

        use_lr_scheduler:           if true uses Cosine learning rate scheduler
        max_scheduler_steps:        maximum number of steps in the lr scheduler
    """

    def __init__(self,
                 generator_network: nn.Module,
                 generator_distribution: DISTRIBUTION,
                 var_latent_network: nn.Module,
                 var_latent_distribution: DISTRIBUTION,
                 num_latent_samples: int,
                 lr_generator: float,
                 lr_latent: float,
                 *args,
                 prior_distribution: PRIOR_DISTRIBUTION = PRIOR_DISTRIBUTION.std_normal,
                 amsgrad_generator: bool = False,
                 amsgrad_latent: bool = False,
                 use_lr_scheduler: bool = False,
                 max_scheduler_steps: int = -1,
                 encoder_use_mis_mask: bool = False,
                 var_latent_STL: bool = False,
                 kl_analytic: bool = True,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        assert (not (var_latent_STL and kl_analytic)),\
            '"Sticking the Landing" cannot be used with analytical KL.'

        self.generator_network = generator_network
        self.var_latent_network = var_latent_network

    def predict_var_latent_params(self, X, M):
        if self.hparams.encoder_use_mis_mask:
            # NOTE: If necessary here can also add network-specific code
            if isinstance(self.var_latent_network, ResNetEncoder):
                params = self.var_latent_network(X, M)
            else:
                XM = rearrange([X, M], 'f ... d -> ... (f d)')
                params = self.var_latent_network(XM)
        else:
            params = self.var_latent_network(X)

        return params

    def get_prior(self):
        if self.hparams.prior_distribution is PRIOR_DISTRIBUTION.std_normal:
            prior_distr = torch.distributions.Normal(loc=0., scale=1.)
        else:
            raise NotImplementedError(f'Method not implemented for {self.hparams.prior_distribution=}')

        return prior_distr

    def split_params(self, params: torch.Tensor, num_params: int) -> torch.Tensor:
        """
        Rearranges parameter tensor such that each parameter-group is in its own "row".
        E.g. for gaussian parameters (b (means+logvars)) -> ((means logvars) b d)
        """
        return rearrange(params, '... (params d) -> params ... d', params=num_params)

    def create_distribution(self, params: torch.Tensor, distribution: DISTRIBUTION,
                            *, validate_args: bool = None) -> torch.distributions.Distribution:
        """
        Creates a torch.Distribution object from the parameters
        """
        if distribution is DISTRIBUTION.normal:
            # loc, logvar = self.split_params(params, num_params=2)
            # scale = torch.exp(logvar*0.5)
            loc, scale_raw = self.split_params(params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw)
            # scale += EPSILON
            distr = torch.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
        elif distribution is DISTRIBUTION.normal_with_eps:
            # loc, logvar = self.split_params(params, num_params=2)
            # scale = torch.exp(logvar*0.5)
            loc, scale_raw = self.split_params(params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw) + EPSILON
            distr = torch.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
        elif distribution is DISTRIBUTION.bern:
            distr = torch.distributions.Bernoulli(logits=params, validate_args=validate_args)
        elif distribution is DISTRIBUTION.bern_m1_p1:
            distr = Bernoulli_m1_p1(logits=params, validate_args=validate_args)
        elif distribution is DISTRIBUTION.cont_bern:
            # NOTE: Need to use probs, not logits, since probs are clamped to (1.19e-7, 1-1.19e-7)
            # whilst the logits aren't clamped in the correspondingly.
            # This causes divergence between logits and probs causes negative KL divergence errors.
            # https://github.com/pytorch/pytorch/issues/72525
            # logits = self.split_params(params, num_params=1)[0]
            # distr = torch.distributions.ContinuousBernoulli(logits=logits, validate_args=validate_args)
            # However, using probs parametrisation is also an issue, since the logits get truncated at -15 ~= log(1.19e-7)
            # which in turn results in posterior collapse of sorts
            # probs = torch.sigmoid(self.split_params(params, num_params=1)[0])
            # distr = torch.distributions.ContinuousBernoulli(probs=probs, validate_args=validate_args)
            # Instead we use a patched ContinuousBernoulli that implements table mean, kl and log-partition methods
            # ported from TFP implementation
            # logits = self.split_params(params, num_params=1)[0]
            distr = ContinuousBernoulliPatched(logits=torch.clamp(params, -18, 18))
        elif distribution is DISTRIBUTION.cont_bern_orig:
            # For comparison also added the option to run the original implementation
            distr = torch.distributions.ContinuousBernoulli(logits=params, validate_args=validate_args)
        elif distribution is DISTRIBUTION.cont_bern_orig_prob:
            # For comparison also added the option to run the original implementation
            probs = torch.sigmoid(params)
            distr = torch.distributions.ContinuousBernoulli(probs=probs, validate_args=validate_args)
        elif distribution is DISTRIBUTION.cont_bern_prob:
            probs = torch.sigmoid(params)
            distr = ContinuousBernoulliPatched(probs=probs, validate_args=validate_args)
        elif distribution in (DISTRIBUTION.stratified_mixture1_normal_with_eps,
                              DISTRIBUTION.stratified_mixture5_normal_with_eps,
                              DISTRIBUTION.stratified_mixture15_normal_with_eps,
                              DISTRIBUTION.stratified_mixture25_normal_with_eps):
            if distribution is DISTRIBUTION.stratified_mixture1_normal_with_eps:
                num_mixture_components = 1
            elif distribution is DISTRIBUTION.stratified_mixture5_normal_with_eps:
                num_mixture_components = 5
            elif distribution is DISTRIBUTION.stratified_mixture15_normal_with_eps:
                num_mixture_components = 15
            elif distribution is DISTRIBUTION.stratified_mixture25_normal_with_eps:
                num_mixture_components = 25
            # NOTE: need to handle this distributuon differently!
            # Only use it for the encoder, not the generator
            comp_params, comp_log_weights = params[..., :-num_mixture_components], params[..., -num_mixture_components:]
            loc, scale_raw = self.split_params(comp_params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw) + EPSILON

            # Rearrange mixture component params
            loc = rearrange(loc, '... (k z) -> ... k z', k=num_mixture_components)
            scale = rearrange(scale, '... (k z) -> ... k z', k=num_mixture_components)

            # Create the mixture distribution
            mix = torch.distributions.Categorical(logits=comp_log_weights)
            comp = torch.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
            comp = torch.distributions.Independent(comp, reinterpreted_batch_ndims=1)
            distr = StratifiedMixtureSameFamily(mix, comp)
        elif distribution in (DISTRIBUTION.stratified_mixture1_normal,
                              DISTRIBUTION.stratified_mixture5_normal,
                              DISTRIBUTION.stratified_mixture15_normal,
                              DISTRIBUTION.stratified_mixture25_normal):
            if distribution is DISTRIBUTION.stratified_mixture1_normal:
                num_mixture_components = 1
            elif distribution is DISTRIBUTION.stratified_mixture5_normal:
                num_mixture_components = 5
            elif distribution is DISTRIBUTION.stratified_mixture15_normal:
                num_mixture_components = 15
            elif distribution is DISTRIBUTION.stratified_mixture25_normal:
                num_mixture_components = 25
            # NOTE: need to handle this distributuon differently!
            # Only use it for the encoder, not the generator
            comp_params, comp_log_weights = params[..., :-num_mixture_components], params[..., -num_mixture_components:]
            loc, scale_raw = self.split_params(comp_params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw)

            # Rearrange mixture component params
            loc = rearrange(loc, '... (k z) -> ... k z', k=num_mixture_components)
            scale = rearrange(scale, '... (k z) -> ... k z', k=num_mixture_components)

            # Create the mixture distribution
            mix = torch.distributions.Categorical(logits=comp_log_weights)
            comp = torch.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
            comp = torch.distributions.Independent(comp, reinterpreted_batch_ndims=1)
            distr = StratifiedMixtureSameFamily(mix, comp)
        elif distribution in (DISTRIBUTION.reparametrised_mixture5_normal_with_eps,
                              DISTRIBUTION.reparametrised_mixture15_normal_with_eps,
                              DISTRIBUTION.reparametrised_mixture25_normal_with_eps,):
            if distribution is DISTRIBUTION.reparametrised_mixture5_normal_with_eps:
                num_mixture_components = 5
            elif distribution is DISTRIBUTION.reparametrised_mixture15_normal_with_eps:
                num_mixture_components = 15
            elif distribution is DISTRIBUTION.reparametrised_mixture25_normal_with_eps:
                num_mixture_components = 25
            # NOTE: need to handle this distributuon differently!
            # Only use it for the encoder, not the generator
            comp_params, comp_log_weights = params[..., :-num_mixture_components], params[..., -num_mixture_components:]
            loc, scale_raw = self.split_params(comp_params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw) + EPSILON

            # Rearrange mixture component params
            loc = rearrange(loc, '... (k z) -> ... k z', k=num_mixture_components)
            scale = rearrange(scale, '... (k z) -> ... k z', k=num_mixture_components)

            # Create the mixture distribution
            mix = torch.distributions.Categorical(logits=comp_log_weights)
            comp = torch.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
            comp = torch.distributions.Independent(comp, reinterpreted_batch_ndims=1)
            distr = ReparametrizedMixtureSameFamily(mix, comp)
        elif distribution in (DISTRIBUTION.reparametrised_mixture5_normal,
                              DISTRIBUTION.reparametrised_mixture15_normal,
                              DISTRIBUTION.reparametrised_mixture25_normal,):
            if distribution is DISTRIBUTION.reparametrised_mixture5_normal:
                num_mixture_components = 5
            elif distribution is DISTRIBUTION.reparametrised_mixture15_normal:
                num_mixture_components = 15
            elif distribution is DISTRIBUTION.reparametrised_mixture25_normal:
                num_mixture_components = 25
            # NOTE: need to handle this distributuon differently!
            # Only use it for the encoder, not the generator
            comp_params, comp_log_weights = params[..., :-num_mixture_components], params[..., -num_mixture_components:]
            loc, scale_raw = self.split_params(comp_params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw)

            # Rearrange mixture component params
            loc = rearrange(loc, '... (k z) -> ... k z', k=num_mixture_components)
            scale = rearrange(scale, '... (k z) -> ... k z', k=num_mixture_components)

            # Create the mixture distribution
            mix = torch.distributions.Categorical(logits=comp_log_weights)
            comp = torch.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
            comp = torch.distributions.Independent(comp, reinterpreted_batch_ndims=1)
            distr = ReparametrizedMixtureSameFamily(mix, comp)
        else:
            raise NotImplementedError(f'Method not implemented for {distribution = }')

        return distr

    @property
    def latent_dims(self):
        latent_dim = None
        if isinstance(self.generator_network, FullyConnectedNetwork):
            latent_dim = self.generator_network.layer_dims[0]
        elif isinstance(self.generator_network, ResidualFCNetwork):
            latent_dim = self.generator_network.input_dim
        elif isinstance(self.generator_network, ResNetDecoder):
            latent_dim = self.generator_network.latent_dim
        else:
            raise NotImplementedError(f'Method not implemented for {type(self.generator_network) = }')
        return latent_dim

    def sample_vae(self, num_samples):
        prior_distr = self.get_prior()

        Z = prior_distr.sample((num_samples, self.latent_dims)).to(device=self.device)
        generator_params = self.generator_network(Z)
        generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)
        X_samples = generator_distr.sample()

        return X_samples

    def vae_forward(self, X, M, *, return_detached_latent_distr, Z_sample_shape=None, sample_stratifieddistr_without_stratification_norsample=False):
        # TODO: technically we could always return var_latent_distr_detached, so the flag might not be necessary
        # Also might not need to return it at all, instead could use this: https://github.com/pyro-ppl/pyro/pull/2599/

        if Z_sample_shape is None:
            Z_sample_shape = (self.hparams.num_latent_samples,)

        # Create latent distribution and sample
        var_latent_params = self.predict_var_latent_params(X, M)
        var_latent_distr = self.create_distribution(var_latent_params, self.hparams.var_latent_distribution)
        var_latent_distr_detached = None
        if return_detached_latent_distr:
            var_latent_distr_detached = self.create_distribution(var_latent_params.detach(), self.hparams.var_latent_distribution)
        if sample_stratifieddistr_without_stratification_norsample:
            assert not var_latent_params.requires_grad, 'Cannot use stratifieddistr_without_stratification_norsample with gradients.'
            var_latent_distr.set_sample_not_stratified()
            Z = var_latent_distr.sample(sample_shape=Z_sample_shape)
        else:
            Z = var_latent_distr.rsample(sample_shape=Z_sample_shape)

        # Create prior distribution
        prior_distr = self.get_prior()

        # Create generator distribution
        generator_params = self.generator_network(Z)
        generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)

        return VAEForwardOutputs(
            var_latent_distr=var_latent_distr,
            var_latent_distr_detached=var_latent_distr_detached,
            Z=Z,
            prior_distr=prior_distr,
            generator_distr=generator_distr,
        )

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        elbo = estimate_elbo(self, batch,
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
            elbo = estimate_elbo(self, batch,
                                 kl_analytic=self.hparams.kl_analytic,
                                 var_latent_STL=self.hparams.var_latent_STL,
                                 num_latent_samples=self.hparams.num_latent_samples)
            loss = -elbo

        # logs metrics
        self.log('elbo/val', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.generator_network.parameters(),
             'amsgrad': self.hparams.amsgrad_generator,
             'lr': self.hparams.lr_generator},
            {'params': self.var_latent_network.parameters(),
             'amsgrad': self.hparams.amsgrad_latent,
             'lr': self.hparams.lr_latent}])

        opts = {
            'optimizer': optimizer
        }

        if self.hparams.use_lr_scheduler:
            max_steps = self.hparams.max_scheduler_steps if self.hparams.max_scheduler_steps != -1 else self.estimated_num_training_steps
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps, eta_min=0, last_epoch=-1)

            # NOTE: these configs are not available in MVB2 manual optimisation.
            # So if this changes, you should also check if it need to be changed in MVB2.
            opts['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
                'frequency': 1,
            }

        return opts

    def configure_gradient_clipping(self,
            optimizer: torch.optim.Optimizer,
            optimizer_idx: int,
            gradient_clip_val: Optional[Union[int, float]] = None,
            gradient_clip_algorithm: Optional[str] = None):
        # NOTE: We will only clip the gradients of the encoder network
        # Create a dummy optimiser object to work around pytorch-lightning
        # class DummyOpt(object):
        #     pass
        # dummy_opt = DummyOpt()
        # dummy_opt.param_groups = [{'params': self.var_latent_network.parameters()}]

        self.clip_gradients(optimizer, #dummy_opt,
                            gradient_clip_val=gradient_clip_val,
                            gradient_clip_algorithm=gradient_clip_algorithm)


def compute_elbo_w_analytic_kl(
                    X: torch.Tensor,
                    M: torch.Tensor,
                    var_latent_distr: torch.distributions.Distribution,
                    prior_distr: torch.distributions.Distribution,
                    generator_distr: torch.distributions.Distribution) -> torch.Tensor:
    # Compute cross-entropy term of observed data
    log_prob = generator_distr.log_prob(X)*M
    log_prob = reduce(log_prob, 'z b ... d -> z b ...', 'sum')
    log_prob = reduce(log_prob, 'z b ... -> b ...', 'mean')

    # Compute analytical -KL(q(z|x) || p(z)) term
    KL_neg = -torch.distributions.kl_divergence(var_latent_distr, prior_distr)
    KL_neg = reduce(KL_neg, 'b ... d -> b ...', 'sum')

    # Compute per-data-point elbos
    return log_prob + KL_neg


def compute_elbo_w_montecarlokl(
                    X: torch.Tensor,
                    M: torch.Tensor,
                    Z: torch.Tensor,
                    var_latent_distr: torch.distributions.Distribution,
                    var_latent_distr_detached: torch.distributions.Distribution,
                    prior_distr: torch.distributions.Distribution,
                    generator_distr: torch.distributions.Distribution,
                    var_latent_STL: bool = False) -> torch.Tensor:
    # Compute cross-entropy term of observed data
    generator_log_prob = generator_distr.log_prob(X)*M
    generator_log_prob = reduce(generator_log_prob, 'z b ... d -> z b ...', 'sum')
    # generator_log_prob = reduce(generator_log_prob, 'z b ... -> b ...', 'mean')

    # Compute prior latent probability
    prior_logprob = prior_distr.log_prob(Z)
    prior_logprob = reduce(prior_logprob, 'z b ... d -> z b ...', 'sum')

    # Compute the log-prob of samples under the latent distribution
    if var_latent_STL:
        # NOTE: alternatively could use this https://github.com/pyro-ppl/pyro/pull/2599/
        latent_logprob = var_latent_distr_detached.log_prob(Z)
    else:
        latent_logprob = var_latent_distr.log_prob(Z)

    if (not isinstance(var_latent_distr, (StratifiedMixtureSameFamily, ReparametrizedMixtureSameFamily)) and
            not isinstance(var_latent_distr_detached, (StratifiedMixtureSameFamily, ReparametrizedMixtureSameFamily))):
        latent_logprob = reduce(latent_logprob, 'z b ... d -> z b ...', 'sum')

    # Compute per-data-point elbo
    total = generator_log_prob + prior_logprob - latent_logprob
    elbos = reduce(total, 'z b ... -> b ...', 'mean')

    return elbos

def estimate_elbo(model, batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
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

    # Compute per data-point elbo
    if kl_analytic:
        elbo = compute_elbo_w_analytic_kl(X, M, var_latent_distr, prior_distr, generator_distr)
    else:
        elbo = compute_elbo_w_montecarlokl(X, M, Z, var_latent_distr, var_latent_distr_detached, prior_distr, generator_distr,
                                            var_latent_STL=var_latent_STL)

    # Averaged elbo
    elbo = reduce(elbo, 'b -> ', 'mean')

    return elbo
