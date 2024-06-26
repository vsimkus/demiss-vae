import math

import numpy as np
import torch
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
from torch.distributions.kl import register_kl


# https://github.com/pytorch/pytorch/issues/72525


def _log_xexp_ratio(x):
    """
    Compute log(x * (exp(x) + 1) / (exp(x) - 1)) in a numerically stable way.
    Ported from: https://github.com/tensorflow/probability/blob/HEAD/tensorflow_probability/python/distributions/continuous_bernoulli.py#L33-L76
    """
    dtype = x.dtype
    device = x.device
    eps = torch.finfo(x.dtype).eps

    # This function is even, hence we use abs(x) everywhere.
    x = torch.abs(x)

    # For x near zero, we have the Taylor series:
    # log(2) + x**2 / 12 - 7 x**4 / 1440 + 31 x**6 / 90720 + O(x**8)
    # whose coefficients decrease in magnitude monotonically

    # For x large in magnitude, the ratio (exp(x) + 1) / (exp(x) - 1)) tends to
    # sign(x), so thus this function should tend to log(abs(x))

    # Finally for x medium in magnitude, we can use the naive expression. Thus,
    # we generate 2 cutofs.

    # Use the first 3 non-zero terms of the Taylor series when
    # |x| < small_cutoff.
    small_cutoff = math.pow(eps * 90720. / 31, 1 / 6.)

    # Use log(abs(x)) when |x| > large_cutoff
    large_cutoff = -math.log(eps)

    x_squared = torch.square(x)
    x_pow4 = torch.square(x_squared)
    # x_pow6 = torch.pow(x_squared, 3)

    result = (math.log(2.) + x_squared / 12.
              - 7 * x_pow4 / 1440.
            #   + 31 * x_pow6 / 90720
              )
    middle_region = (x > small_cutoff) & (x < large_cutoff)
    safe_x_medium = torch.where(middle_region, x, torch.tensor(1., dtype=dtype, device=device))
    result = torch.where(
        middle_region,
        (torch.log(safe_x_medium) + torch.nn.functional.softplus(safe_x_medium) -
        torch.log(torch.expm1(safe_x_medium))),
        result)

    # We can do this by choosing a cutoff when x > log(1 / machine eps)
    safe_x_large = torch.where(x >= large_cutoff, x, torch.tensor(1., dtype=dtype, device=device))
    result = torch.where(x >= large_cutoff, torch.log(safe_x_large), result)
    return result


def log1mexp(x):
    """Compute `log(1 - exp(-|x|))` elementwise in a numerically stable way.

    Ported from: https://github.com/tensorflow/probability/blob/v0.15.0/tensorflow_probability/python/math/generic.py#L648-L673

    Args:
    x: Float `Tensor`.

    Returns:
    log1mexp: Float `Tensor` of `log1mexp(x)`.

    #### References

    [1]: Machler, Martin. Accurately computing log(1 - exp(-|a|))
        https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """

    x = torch.abs(x)
    return torch.where(
        # This switching point is recommended in [1].
        x < math.log(2), torch.log(-torch.expm1(-x)),
        torch.log1p(-torch.exp(-x)))


class ContinuousBernoulliPatched(ContinuousBernoulli):
    def entropy(self):
        raise NotImplementedError('Numerically stable entropy is not implemented yet.')

    def cdf(self, value):
        raise NotImplementedError('Numerically stable cdf is not implemented yet.')

    @property
    def variance(self):
        raise NotImplementedError('Numerically stable variance is not implemented yet.')

    def _cont_bern_log_norm(self):
        """Override with a more numerically stable implementation

        Ported from: https://github.com/tensorflow/probability/blob/HEAD/tensorflow_probability/python/distributions/continuous_bernoulli.py#L199-L213"""
        # The normalizer is 2 * atanh(1 - 2 * probs) / (1 - 2 * probs), with the
        # removable singularity at probs = 0.5 removed (and replaced with 2).
        # We do this computation in logit space to be more numerically stable.
        # Note that 2 * atanh(1 - 2 / (1 + exp(-logits))) = -logits.
        # Thus we end up with
        # -logits / (1 - 2 / (1 + exp(-logits))) =
        # logits / ((-exp(-logits) + 1) / (exp(-logits) + 1)) =
        # (exp(-logits) + 1) * logits / (-exp(-logits) + 1) =
        # (1 + exp(logits)) * logits / (exp(logits) - 1)

        return _log_xexp_ratio(self.logits)

    @property
    def mean(self):
        """Override with a more numerically stable implementation

        Ported from: https://github.com/tensorflow/probability/blob/HEAD/tensorflow_probability/python/distributions/continuous_bernoulli.py#L307-L340
        """
        # The mean is probs / (2 * probs - 1) + 1 / (2 * arctanh(1 - 2 * probs))
        # with the removable singularity at 0.5 removed.
        # We write this in logits space.
        # The first term becomes
        # 1 / (1 + exp(-logits)) / (2 / (1 + exp(-logits)) - 1) =
        # 1 / (2 - 1 - exp(-logits)) =
        # 1 / (1 - exp(-logits))
        # The second term becomes - 1 / logits.
        # Thus we have mean = 1 / (1 - exp(-logits)) - 1 / logits.

        # When logits is close to zero, we can compute the Laurent series for the
        # first term as:
        # 1 / x + 1 / 2 + x / 12 - x**3 / 720 + x**5 / 30240 + O(x**7).
        # Thus we get the pole at zero canceling out with the second term.

        # For large negative logits, the denominator (1 - exp(-logits)) in
        # the first term yields inf values. Whilst the ratio still returns
        # zero as it should, the gradients of this ratio become nan.
        # Thus, noting that 1 / (1 - exp(-logits)) quickly tends towards 0
        # for large negative logits, the mean tends towards - 1 / logits.

        logits = self.logits

        dtype = logits.dtype
        device = logits.device
        eps = torch.finfo(logits.dtype).eps

        small_cutoff = math.pow(eps * 30240, 1 / 5.)
        result = 0.5 + logits / 12. - logits * torch.square(logits) / 720

        large_cutoff = -math.log(eps)

        safe_logits_mask = (torch.abs(logits) > small_cutoff) & (logits > -large_cutoff)
        safe_logits = torch.where(
            safe_logits_mask,
            logits,
            torch.tensor(1., dtype=dtype, device=device))

        result  = torch.where(
            safe_logits_mask,
            # NOTE: this is yields unstable gradients for large negative logits
            -(torch.reciprocal(
                torch.expm1(-safe_logits)) +
            torch.reciprocal(safe_logits)),
            result)

        # NOTE: this is new (not in tensorflow)

        large_neg_mask = logits <= -large_cutoff
        logits_large_neg = torch.where(large_neg_mask, logits, torch.tensor(1., dtype=dtype, device=device))
        return torch.where(large_neg_mask, -torch.reciprocal(logits_large_neg), result)

    def icdf(self, value):
        """Override with a more numerically stable implementation

        Ported from: https://github.com/tensorflow/probability/blob/HEAD/tensorflow_probability/python/distributions/continuous_bernoulli.py#L392-L426
        """
        logits = self.logits
        dtype = logits.dtype
        device = logits.device

        p = value
        logp = torch.log(p)
        # The expression for the quantile function is:
        # log(1 + (e^s - 1) * p) / s, where s is `logits`. When s is large,
        # the e^s sub-term becomes increasingly ill-conditioned.  However,
        # since the numerator tends to s, we can reformulate the s > 0 case
        # as a offset from 1, which is more accurate.  Coincidentally,
        # this eliminates a ratio of infinities problem when `s == +inf`.

        safe_negative_logits = torch.where(logits < 0., logits, torch.tensor(-1., dtype=dtype, device=device))
        safe_positive_logits = torch.where(logits > 0., logits, torch.tensor(1., dtype=dtype, device=device))
        result = torch.where(
            logits > 0.,
            1. + torch.logaddexp(
                logp + log1mexp(safe_positive_logits),
                -safe_positive_logits) / safe_positive_logits,
            torch.log1p(
                torch.expm1(safe_negative_logits) * p) / safe_negative_logits)

        # When logits is zero, we can simplify
        # log(1 + (e^s - 1) * p) / s ~= log(1 + s * p) / s ~= s * p / s = p
        # Specifically, when logits is zero, the naive computation produces a NaN.
        result = torch.where(logits == 0., p, result)

        # Finally, handle the case where `logits` and `p` are on the boundary,
        # as the above expressions can result in ratio of `infs` in that case as
        # well.
        return torch.where(
            ((logits == -np.inf) & (logp == 0.)) |
            ((logits == np.inf) & (logp == np.inf)),
            torch.ones_like(logits),
            result)


@register_kl(ContinuousBernoulliPatched, ContinuousBernoulliPatched)
def _kl_continuous_bernoulli_continuous_bernoulli_patched(p, q):
    p_log_probs0, p_log_probs1 = -torch.nn.functional.softplus(p.logits), -torch.nn.functional.softplus(-p.logits)
    q_log_probs0, q_log_probs1 = -torch.nn.functional.softplus(q.logits), -torch.nn.functional.softplus(-q.logits)
    p_mean = p.mean
    p_log_norm = p._cont_bern_log_norm()
    q_log_norm = q._cont_bern_log_norm()
    # t1 = p.mean * (p_log_probs1 + q_log_probs0
    #                - p_log_probs0 - q_log_probs1)
    # t2 = p._cont_bern_log_norm() + p_log_probs0
    # t3 = - q._cont_bern_log_norm() - q_log_probs0
    # return t1 + t2 + t3
    kl = (
        p_mean * (p_log_probs1 + q_log_probs0 - p_log_probs0 - q_log_probs1)
        + p_log_norm - q_log_norm + p_log_probs0 - q_log_probs0)
    # return torch.clamp(kl, min=0.)
    return kl


if __name__ == '__main__':
    # Check that the overriden functions are the same when parametrised via the probs argument
    A = torch.rand(2000)
    B = torch.rand(2000)

    a = ContinuousBernoulliPatched(probs=A)
    b = ContinuousBernoulliPatched(probs=B)
    a_orig = ContinuousBernoulli(probs=A)
    b_orig = ContinuousBernoulli(probs=B)

    U = torch.rand(2000)
    assert torch.allclose(a.mean, a_orig.mean)
    assert torch.allclose(a._cont_bern_log_norm(), a_orig._cont_bern_log_norm())
    assert torch.allclose(a.icdf(U), a_orig.icdf(U))
    assert torch.allclose(a.log_prob(U), a_orig.log_prob(U))
    assert torch.allclose(b.mean, b_orig.mean)
    assert torch.allclose(b.icdf(U), b_orig.icdf(U))

    assert np.allclose(torch.distributions.kl_divergence(a, b), torch.distributions.kl_divergence(a_orig, b_orig))

    # https://github.com/pytorch/pytorch/issues/72525

    A = torch.randn(2000)*200
    B = torch.randn(2000)

    a = ContinuousBernoulliPatched(logits=A)
    b = ContinuousBernoulliPatched(logits=B)

    assert torch.all(torch.distributions.kl_divergence(a, b) >= 0)
