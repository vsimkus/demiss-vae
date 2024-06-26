from math import prod

import torch
import torch.distributions as distr
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.distributions import MixtureSameFamily


"""
Copied from https://github.com/vsimkus/torch-reparametrised-mixture-distribution
"""

class ReparametrizedMixtureSameFamily(MixtureSameFamily):
    """
    Adds rsample method to the MixtureSameFamily class
    that implements implicit reparametrization.
    """
    has_rsample = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self._component_distribution.has_rsample:
            raise ValueError('Cannot reparameterize a mixture of non-reparameterizable components.')

    def rsample(self, sample_shape=torch.Size()):
        """Adds reparameterization (pathwise) gradients to samples of the mixture.

        Based on Tensorflow Probability implementation
        https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/distributions/mixture_same_family.py#L433-L498

        Implicit reparameterization gradients are

        .. math::
            dx/dphi = -(d transform(x, phi) / dx)^-1 * d transform(x, phi) / dphi,

        where transform(x, phi) is distributional transform that removes all
        parameters from samples x.

        We implement them by replacing x with
        -stop_gradient(d transform(x, phi) / dx)^-1 * transform(x, phi)]
        for the backward pass (gradient computation).
        The derivative of this quantity w.r.t. phi is then the implicit
        reparameterization gradient.
        Note that this replaces the gradients w.r.t. both the mixture
        distribution parameters and components distributions parameters.

        Limitations:
        1. Fundamental: components must be fully reparameterized.
        2. Distributional transform is currently only implemented for
            factorized components.

        Args:
            x: Sample of mixture distribution

        Returns:
            Tensor with same value as x, but with reparameterization gradients
        """
        x = self.sample(sample_shape=sample_shape)
        if not torch.is_grad_enabled():
            return x

        event_size = prod(self.event_shape)
        if event_size != 1:
            # Multivariate case
            x_2d_shape = (-1, event_size)

            # Perform distributional transform of x in [S, B, E] shape,
            # but have Jacobian of size [S*prod(B), prod(E), prod(E)].
            def reshaped_distributional_transform(x_2d):
                return torch.reshape(
                        self._distributional_transform(x_2d.reshape(x.shape)),
                        x_2d_shape)

            x_2d = x.reshape(x_2d_shape)

            # Compute transform (the gradients of this transform will be computed using autodiff)
            # transform_2d: [S*prod(B), prod(E)]
            transform_2d = reshaped_distributional_transform(x_2d)

            # Compute the Jacobian of the distributional transform
            def batched_jacobian_of_reshaped_distributional_transform(x_2d):
                # Used to compute the batched Jacobian for a function that takes a (B, N) and produces (B, M).
                # NOTE: the function must be independent for each element in B. Otherwise, this would be incorrect.
                # See: https://pytorch.org/functorch/1.13/notebooks/jacobians_hessians.html#batch-jacobian-and-batch-hessian
                def reshaped_distributional_transform_summed(x_2d):
                    return torch.sum(
                            reshaped_distributional_transform(x_2d),
                            dim=0)
                return jacobian(reshaped_distributional_transform_summed, x_2d).detach().movedim(1, 0)
            # jacobian: [S*prod(B), prod(E), prod(E)]
            jac = batched_jacobian_of_reshaped_distributional_transform(x_2d)

            # We only provide the first derivative; the second derivative computed by
            # autodiff would be incorrect, so we raise an error if it is requested.
            # TODO: prevent 2nd derivative of transform_2d.

            # Compute [- stop_gradient(jacobian)^-1 * transform] by solving a linear
            # system. The Jacobian is lower triangular because the distributional
            # transform for i-th event dimension does not depend on the next
            # dimensions.
            surrogate_x_2d = -torch.triangular_solve(transform_2d[..., None], jac.detach(), upper=False)[0]
            surrogate_x = surrogate_x_2d.reshape(x.shape)
        else:
            # For univariate distributions the Jacobian/derivative of the transformation is the
            # density, so the computation is much cheaper.
            transform = self._distributional_transform(x)
            log_prob_x = self.log_prob(x)

            if self._event_ndims > 1:
                log_prob_x = log_prob_x.reshape(log_prob_x.shape + (1,)*self._event_ndims)

            surrogate_x = -transform*torch.exp(-log_prob_x.detach())

        # Replace gradients of x with gradients of surrogate_x, but keep the value.
        return x + (surrogate_x - surrogate_x.detach())

    def _distributional_transform(self, x):
        """Performs distributional transform of the mixture samples.

        Based on Tensorflow Probability implementation
        https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/distributions/mixture_same_family.py#L500-L574

        Distributional transform removes the parameters from samples of a
        multivariate distribution by applying conditional CDFs:

        .. math::
            (F(x_1), F(x_2 | x1_), ..., F(x_d | x_1, ..., x_d-1))

        (the indexing is over the 'flattened' event dimensions).
        The result is a sample of product of Uniform[0, 1] distributions.

        We assume that the components are factorized, so the conditional CDFs become

        .. math::
          `F(x_i | x_1, ..., x_i-1) = sum_k w_i^k F_k (x_i),`

        where :math:`w_i^k` is the posterior mixture weight: for :math:`i > 0`
        :math:`w_i^k = w_k prob_k(x_1, ..., x_i-1) / sum_k' w_k' prob_k'(x_1, ..., x_i-1)`
        and :math:`w_0^k = w_k` is the mixture probability of the k-th component.

        Args:
            x: Sample of mixture distribution

        Returns:
            Result of the distributional transform
        """
        # Obtain factorized components distribution and assert that it's
        # a scalar distribution.
        if isinstance(self._component_distribution, distr.Independent):
            univariate_components = self._component_distribution.base_dist
        else:
            univariate_components = self._component_distribution

        # Expand input tensor and compute log-probs in each component
        x = self._pad(x)  # [S, B, 1, E]
        # NOTE: Only works with fully-factorised distributions!
        log_prob_x = univariate_components.log_prob(x)  # [S, B, K, E]

        event_size = prod(self.event_shape)
        if event_size != 1:
            # Multivariate case
            # Compute exclusive cumulative sum
            # log prob_k (x_1, ..., x_i-1)
            cumsum_log_prob_x = log_prob_x.reshape(-1, event_size)  # [S*prod(B)*K, prod(E)]
            cumsum_log_prob_x = torch.cumsum(cumsum_log_prob_x, dim=-1)
            cumsum_log_prob_x = cumsum_log_prob_x.roll(1, -1)
            cumsum_log_prob_x[:, 0] = 0
            cumsum_log_prob_x = cumsum_log_prob_x.reshape(log_prob_x.shape)

            logits_mix_prob = self._pad_mixture_dimensions(self._mixture_distribution.logits)

            # Logits of the posterior weights: log w_k + log prob_k (x_1, ..., x_i-1)
            log_posterior_weights_x = logits_mix_prob + cumsum_log_prob_x

            # Normalise posterior weights
            component_axis = -self._event_ndims-1
            posterior_weights_x = torch.softmax(log_posterior_weights_x, dim=component_axis)

            cdf_x = univariate_components.cdf(x)  # [S, B, K, E]
            return torch.sum(posterior_weights_x * cdf_x, dim=component_axis)
        else:
            # For univariate distributions logits of the posterior weights = log w_k
            log_posterior_weights_x = self._mixture_distribution.logits
            posterior_weights_x = torch.softmax(log_posterior_weights_x, dim=-1)
            posterior_weights_x = self._pad_mixture_dimensions(posterior_weights_x)

            cdf_x = univariate_components.cdf(x)  # [S, B, K, E]
            component_axis = -self._event_ndims-1
            return torch.sum(posterior_weights_x * cdf_x, dim=component_axis)


    def _log_cdf(self, x):
        x = self._pad(x)
        if callable(getattr(self._component_distribution, '_log_cdf', None)):
            log_cdf_x = self.component_distribution._log_cdf(x)
        else:
            # NOTE: This may be unstable
            log_cdf_x = torch.log(self.component_distribution.cdf(x))

        if isinstance(self.component_distribution, (distr.Bernoulli, distr.Binomial, distr.ContinuousBernoulli,
                                                    distr.Geometric, distr.NegativeBinomial, distr.RelaxedBernoulli)):
            log_mix_prob = torch.sigmoid(self.mixture_distribution.logits)
        else:
            log_mix_prob = F.log_softmax(self.mixture_distribution.logits, dim=-1)

        return torch.logsumexp(log_cdf_x + log_mix_prob, dim=-1)


if __name__ == "__main__":
    from torch.distributions.normal import Normal
    from math import prod

    dims = (2, 4)
    torch.manual_seed(1111)
    mixture_probs=torch.softmax(torch.randn(*dims), dim=-1)
    # mixture_probs=torch.softmax(torch.randn(dims[0]), dim=-1)
    locs = torch.arange(prod(dims)).float().reshape(*dims)
    stds = torch.abs(torch.randn(*dims))*3

    # Make sure with and without event dim we get the same result
    mixture = distr.Categorical(probs=mixture_probs)
    components = Normal(loc=locs, scale=stds)
    mog = ReparametrizedMixtureSameFamily(mixture_distribution=mixture,
                                          component_distribution=components)
    extra_ndims = 2
    components2 = Normal(loc=locs.reshape(locs.shape + (1,)*extra_ndims),
                               scale=stds.reshape(stds.shape + (1,)*extra_ndims))
    components2 = distr.Independent(components2, extra_ndims)
    mog2 = ReparametrizedMixtureSameFamily(mixture_distribution=mixture,
                                          component_distribution=components2)
    torch.manual_seed(123456)
    X1 = mog.rsample(sample_shape=(5,))
    Z1 = mog._distributional_transform(X1)
    torch.manual_seed(123456)
    X2 = mog2.rsample(sample_shape=(5,))
    Z2 = mog2._distributional_transform(X2)

    assert torch.allclose(X1, X2.squeeze())
    assert torch.allclose(Z1, Z2.squeeze())

    # Check if multivariate runs
    mixture3 = distr.Categorical(probs=mixture_probs[0])
    components3 = Normal(loc=locs.T, scale=stds.T)
    components3 = distr.Independent(components3, 1)
    mog3 = ReparametrizedMixtureSameFamily(mixture_distribution=mixture3,
                                           component_distribution=components3)

    X3 = mog3.rsample(sample_shape=(5,))
    # TODO: add test for multivariate
