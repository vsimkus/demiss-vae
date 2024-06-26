import math
import torch
import numpy as np

from einops import rearrange, asnumpy, repeat

from vgiwae.shared.vae_imputation import VAESampler_Base


def compute_conditional_mog_parameters(X, M, comp_log_probs, means, covs):
    """
    Computes the parameters of the conditional mixture of Gaussians
    p(x) = \sum_c pi_c N(x | mu_c, sigma_c)
    p(xo | xm)  = \frac{\sum_c pi_c N(xo, xm | mu_c, sigma_c)}{\sum_k pi_k N(xo | k)}
                = \sum_c ( \frac{pi_c N(xo | mu_c, sigma_c)}{\sum_k pi_k N(xo | mu_k, sigma_k)} ) N(xm | mu_c, sigma_c) )
                = \sum_c pi_c_|xo N(xm | mu_c, sigma_c)
    where pi_c_|xo = \frac{pi_c N(xo | mu_c, sigma_c)}{\sum_k pi_k N(xo | mu_k, sigma_k)}
    """
    X = X.unsqueeze(1)
    M = M.unsqueeze(1)
    M_not = ~M

    covs = covs.unsqueeze(0)
    means = means.unsqueeze(0)

    sigma_mm = covs * M_not.unsqueeze(-1) * M_not.unsqueeze(-2)
    sigma_mo = covs * M_not.unsqueeze(-1) * M.unsqueeze(-2)
    sigma_oo = covs * M.unsqueeze(-1) * M.unsqueeze(-2)

    M_eye = torch.eye(X.shape[-1], device=X.device).unsqueeze(0).unsqueeze(0) * M_not.unsqueeze(-1)
    sigma_oo_with_ones_in_missing_diag_entries = sigma_oo + M_eye

    # sigma_oo_inv = torch.inverse(sigma_oo_with_ones_in_missing_diag_entries) - M_eye
    # sigma_mo_oo_inv = sigma_mo @ sigma_oo_inv

    xo_minus_mean = (X - means)*M
    sigma_oo_inv_mult_xo_minus_mean = torch.linalg.solve(sigma_oo_with_ones_in_missing_diag_entries, xo_minus_mean)

    # Compute N(xm | xo; k) for each k and each xo

    # means_m_given_o = means*M_not + (sigma_mo_oo_inv @ xo_minus_mean.unsqueeze(-1)).squeeze(-1)
    means_m_given_o = means*M_not + (sigma_mo @ sigma_oo_inv_mult_xo_minus_mean.unsqueeze(-1)).squeeze(-1)
    # covs_m_given_o = sigma_mm - sigma_mo_oo_inv @ sigma_mo.transpose(-1, -2)
    covs_m_given_o = sigma_mm - sigma_mo @ torch.linalg.solve(sigma_oo_with_ones_in_missing_diag_entries, sigma_mo.transpose(-1, -2))

    # Compute the component coefficients pi_k given obs

    D = M.sum(-1)
    # cond_log_probs_oo = 0.5*(-D * torch.log(torch.tensor(2*np.pi))
    #             -torch.logdet(sigma_oo_with_ones_in_missing_diag_entries)
    #             -(xo_minus_mean.unsqueeze(-2) @ sigma_oo_inv @ xo_minus_mean.unsqueeze(-1)).squeeze()
    # )
    cond_log_probs_oo = 0.5*(-D * torch.log(torch.tensor(2*np.pi))
                -torch.logdet(sigma_oo_with_ones_in_missing_diag_entries)
                -(xo_minus_mean.unsqueeze(-2) @ sigma_oo_inv_mult_xo_minus_mean.unsqueeze(-1)).squeeze()
    )

    joint_log_probs_oo = comp_log_probs + cond_log_probs_oo
    comp_log_probs_given_o = joint_log_probs_oo - torch.logsumexp(joint_log_probs_oo, dim=-1, keepdim=True)

    return comp_log_probs_given_o, means_m_given_o, covs_m_given_o

def sample_sparse_mog(num_samples, M, comp_log_probs, cond_means, cond_covs, *, sampling_batch_size=None):
    M_eye = torch.eye(cond_covs.shape[-1], device=comp_log_probs.device).unsqueeze(0).unsqueeze(0) * (M).unsqueeze(1).unsqueeze(-1)
    cond_covs = cond_covs + M_eye
    # NOTE: in pytorch Cholesky decomp is only implemented for positive definite matrices
    L, info = torch.linalg.cholesky_ex(cond_covs)
    # errors = info > 0
    # if errors.sum() > 0:
    #     breakpoint()
    # Instead we can implement Cholesky decomposition for positive semi-definite matrices using eigen decomposition
    # L = semi_definite_symmetric_cholesky(cond_covs)

    # Sample component index
    comp_probs = torch.exp(comp_log_probs)
    comp_distr = torch.distributions.Categorical(probs=comp_probs, validate_args=True)
    component_idx = comp_distr.sample(sample_shape=(num_samples,))
    component_idx = component_idx.T

    if sampling_batch_size is None:
        # Select covariance (L)
        L_ = L[torch.arange(M.shape[0])[:, None], component_idx]

        # Select mean
        means_ = cond_means[torch.arange(M.shape[0])[:, None], component_idx]

        distr = torch.distributions.MultivariateNormal(loc=means_, scale_tril=L_,
                                                    validate_args=False) # Turn off validation, since it does not like negative diagonals
        X = distr.sample()*(~M).unsqueeze(1)
    else:
        X = []
        for b in range(math.ceil(num_samples/sampling_batch_size)):
            component_idx_b = component_idx[:, b*sampling_batch_size:min((b+1)*sampling_batch_size, num_samples)]

            # Select covariance (L)
            L_b = L[torch.arange(M.shape[0])[:, None], component_idx_b]

            # Select mean
            means_ = cond_means[torch.arange(M.shape[0])[:, None], component_idx_b]

            distr = torch.distributions.MultivariateNormal(loc=means_, scale_tril=L_b,
                                                        validate_args=False)

            X_b = distr.sample()*(~M).unsqueeze(1)
            X.append(X_b)
        X = torch.cat(X, dim=1)

    return X.float()

def batched_compute_joint_log_probs_sparse_mogs(X, M, comp_log_probs, means, covs, use_solver=False):
    """
    Computes the log probability of the data xo given the mogs parameters
    p(xm, c) = pi_c p(xm|c)
    """
    M = M.unsqueeze(1)
    M_not = ~M

    sigma_mm = covs * M_not.unsqueeze(-1) * M_not.unsqueeze(-2)

    M_not_eye = torch.eye(X.shape[-1]).unsqueeze(0).unsqueeze(0) * M.unsqueeze(-1)
    sigma_mm_with_ones_in_missing_diag_entries = sigma_mm + M_not_eye
    if not use_solver:
        sigma_mm_inv = torch.inverse(sigma_mm_with_ones_in_missing_diag_entries)# - M_not_eye
        # NOTE: Cholesky inverse should be more stable for SPD matrices than the standard inverse (above)
        # sigma_mm_inv = torch.cholesky_inverse(torch.linalg.cholesky(sigma_mm_with_ones_in_missing_diag_entries))
    # NOTE: standard pytorch cholesky requires SPD matrices (above), instead we use our own implementation that accepts semi-definite matrices too
    # sigma_mm_inv = torch.cholesky_inverse(semi_definite_symmetric_cholesky(sigma_mm_with_ones_in_missing_diag_entries))
    # L = semi_definite_symmetric_cholesky(sigma_mm_with_ones_in_missing_diag_entries)
    # L_inv = torch.inverse(L)
    # sigma_mm_inv = L_inv @ L_inv.transpose(-1, -2)

    if not use_solver:
        sigma_mm_inv = sigma_mm_inv.unsqueeze(1)
    sigma_mm_with_ones_in_missing_diag_entries = sigma_mm_with_ones_in_missing_diag_entries.unsqueeze(1)

    xm_minus_mean = ((X.unsqueeze(-2) - means.unsqueeze(1))*M_not.unsqueeze(1))

    # NOTE: torch.linalg.solve should be numerically more stable _and_ faster,
    # *BUT* it is slow in this case because it broadcasts the sigma to the large dimensions of xm
    # Hence, it is faster to invert the matrix first (as above)
    if use_solver:
        sigma_mm_inv_mult_xm_minus_mean = torch.linalg.solve(sigma_mm_with_ones_in_missing_diag_entries, xm_minus_mean.unsqueeze(-1))

    D = M_not.sum(-1)
    if not use_solver:
        cond_log_probs_mm = 0.5*(-D.unsqueeze(1) * torch.log(torch.tensor(2*np.pi))
                                 -torch.logdet(sigma_mm_with_ones_in_missing_diag_entries)
                                 -(xm_minus_mean.unsqueeze(-2) @ (sigma_mm_inv @ xm_minus_mean.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        )
    else:
        cond_log_probs_mm = 0.5*(-D.unsqueeze(1) * torch.log(torch.tensor(2*np.pi))
                                -torch.logdet(sigma_mm_with_ones_in_missing_diag_entries)
                                -(xm_minus_mean.unsqueeze(-2) @ sigma_mm_inv_mult_xm_minus_mean).squeeze(-1)
        )

    joint_log_probs = comp_log_probs.unsqueeze(1) + cond_log_probs_mm

    return joint_log_probs


def compute_kl_div_for_sparse_mogs(params1, params2, M, N=1000000, use_solver=False, return_jsd_midpoint=False):
    """Estimates the KL divergence between two mixtures of Gaussians"""
    comp_log_probs1 = params1['comp_log_probs']
    means1 = params1['means']
    covs1 = params1['covs']

    comp_log_probs2 = params2['comp_log_probs']
    means2 = params2['means']
    covs2 = params2['covs']

    if comp_log_probs2.dtype != comp_log_probs1.dtype:
        comp_log_probs2 = comp_log_probs2.to(comp_log_probs1.dtype)
        means2 = means2.to(means1.dtype)
        covs2 = covs2.to(covs1.dtype)

    X = sample_sparse_mog(num_samples=N, M=M, comp_log_probs=comp_log_probs1, cond_means=means1, cond_covs=covs1)
    log_p1 = batched_compute_joint_log_probs_sparse_mogs(X, M, comp_log_probs=comp_log_probs1, means=means1, covs=covs1, use_solver=use_solver)
    log_p1 = torch.logsumexp(log_p1, dim=-1)
    log_p2 = batched_compute_joint_log_probs_sparse_mogs(X, M, comp_log_probs=comp_log_probs2, means=means2, covs=covs2, use_solver=use_solver)
    log_p2 = torch.logsumexp(log_p2, dim=-1)

    kl = (log_p1 - log_p2).mean(dim=1)

    if not return_jsd_midpoint:
        return kl
    else:
        # log-avg-exp to compute log_pm, where pm is the midpoint distribution 0.5(p1 + p2)
        log_pm = torch.logsumexp(rearrange([log_p1, log_p2], 'p ... -> p ...'), dim=0) - torch.log(torch.tensor(2, device=log_p1.device, dtype=log_p1.dtype))
        jsd_term = (log_p1 - log_pm).mean(dim=1)

        return kl, jsd_term

def compute_kl_divs_and_jsd_for_sparse_mogs(params1, params2, M, num_kl_samples=1000000):
    kldivs1, jsd_term1 = compute_kl_div_for_sparse_mogs(params1=params1,
                                                        params2=params2,
                                                        M=M,
                                                        N=num_kl_samples,
                                                        return_jsd_midpoint=True,
                                                        # use_solver=self.hparams.use_solver,)
                                                        )
    kldivs2, jsd_term2 = compute_kl_div_for_sparse_mogs(params1=params2,
                                                        params2=params1,
                                                        M=M,
                                                        N=num_kl_samples,
                                                        return_jsd_midpoint=True,
                                                        # use_solver=self.hparams.use_solver,)
                                                        )

    kldivs_fow = kldivs1
    kldivs_rev = kldivs2
    jsds = 0.5*(jsd_term1 + jsd_term2)

    return kldivs_fow, kldivs_rev, jsds


def linear_interpolate_sparse_mogs(params1, params2, *, alpha: float):
    assert 0 <= alpha <= 1, 'Linear interpolation coefficient must be between 0 and 1.'
    comp_log_probs1 = params1['comp_log_probs']
    means1 = params1['means']
    covs1 = params1['covs']

    comp_log_probs2 = params2['comp_log_probs']
    means2 = params2['means']
    covs2 = params2['covs']

    comp_log_probs = torch.logsumexp(rearrange([torch.log(torch.tensor(alpha)) + comp_log_probs1,
                                                torch.log(torch.tensor(1-alpha)) + comp_log_probs2],
                                               'p ... -> p ...'), dim=0)
    means = alpha*means1 + (1-alpha)*means2
    covs = alpha*covs1 + (1-alpha)*covs2

    return {'comp_log_probs': comp_log_probs,
            'means': means,
            'covs': covs}

def wasserstein_interpolate_sparse_mogs(M, params1, params2, *, alpha: float):
    assert 0 <= alpha <= 1, 'Interpolation coefficient must be between 0 and 1.'
    comp_log_probs1 = params1['comp_log_probs']
    means1 = params1['means']
    covs1 = params1['covs']

    comp_log_probs2 = params2['comp_log_probs']
    means2 = params2['means']
    covs2 = params2['covs']

    comp_log_probs = torch.logsumexp(rearrange([torch.log(torch.tensor(alpha)) + comp_log_probs1,
                                                torch.log(torch.tensor(1-alpha)) + comp_log_probs2],
                                               'p ... -> p ...'), dim=0)
    means = alpha*means1 + (1-alpha)*means2

    # Interpolate the covariance matrices using the Wasserstein distance
    # Based on Chet at al 2018: Optimal transport for Gaussian mixture models
    # https://arxiv.org/abs/1710.07876

    def sparse_matrix_sqrt(covs):
        L, Q = torch.linalg.eigh(covs)
        L[L < 0] = 0 # Set negative eigenvalues (due to matrix sparsity) to 0
        L_sqrt = torch.diag_embed(torch.sqrt(L))
        covs_sqrt = Q @ L_sqrt @ torch.linalg.inv(Q)
        return covs_sqrt

    covs1_sqrt = sparse_matrix_sqrt(covs1)
    M_eye = torch.eye(M.shape[-1], device=M.device).unsqueeze(0) * (M).unsqueeze(-1)
    M_eye = M_eye.unsqueeze(1)
    covs1_sqrt_with_ones_in_obs_diag_entries = covs1_sqrt + M_eye
    covs1_sqrt_inv = torch.linalg.inv(covs1_sqrt_with_ones_in_obs_diag_entries)
    covs1_sqrt_inv = covs1_sqrt_inv - M_eye

    inner_term = alpha*covs1 + (1-alpha)*sparse_matrix_sqrt(covs1_sqrt @ covs2 @ covs1_sqrt)
    innter_term_sq = inner_term @ inner_term
    covs = covs1_sqrt_inv @ innter_term_sq @ covs1_sqrt_inv

    return {'comp_log_probs': comp_log_probs,
            'means': means,
            'covs': covs}

def get_mean_and_cov_from_mog(comp_log_probs, means, covs):
    """
    Computes the first two moments of the MoG distribution.
    I.e. fits a Gaussian to the MoG.
    """
    comp_probs = np.exp(comp_log_probs)
    # Compute the mean of the MoG
    mean = (means*comp_probs[:, None]).sum(0)
    # Compute the covariance of the MoG
    dif = means - mean
    cov = covs + dif[..., None] @ dif[:, None, :]
    cov *= comp_probs[:, None, None]
    cov = cov.sum(0)

    return mean, cov

def construct_target_distribution(target, X, M, *,
                                  comp_log_probs, means, covs,
                                  comp_log_probs_given_o, means_m_given_o, covs_m_given_o,
                                  batch_size, device):
    if target == 'indep_marginal_gaussian':
        target_mean, target_cov = get_mean_and_cov_from_mog(asnumpy(comp_log_probs.exp()), asnumpy(means), asnumpy(covs))
        target_mean = torch.tensor(target_mean)
        target_cov = torch.tensor(target_cov)

        # Set the off-diagonal elements of cov to 0.
        target_cov *= torch.eye(target_cov.shape[-1])

        # Create a MoG with the same mean and covariance for each component
        # target_comp_log_probs = torch.log(torch.ones(comp_log_probs) / len(comp_log_probs))
        target_means = repeat(target_mean, 'd -> 1 c d', c=len(comp_log_probs))
        target_covs = repeat(target_cov, 'd1 d2 -> 1 c d1 d2 ', c=len(comp_log_probs))

        target_means = target_means.to(device)
        target_covs = target_covs.to(device)
        # Intentionally leaving the comp_log_probs the same
        target_comp_logprobs = comp_log_probs_given_o
    elif target == 'marginal_gaussian':
        target_mean, target_cov = get_mean_and_cov_from_mog(asnumpy(comp_log_probs.exp()), asnumpy(means), asnumpy(covs))
        target_mean = torch.tensor(target_mean)
        target_cov = torch.tensor(target_cov)

        # Create a MoG with the same mean and covariance for each component
        # target_comp_log_probs = torch.log(torch.ones(comp_log_probs) / len(comp_log_probs))
        target_means = repeat(target_mean, 'd -> 1 c d', c=len(comp_log_probs))
        target_covs = repeat(target_cov, 'd1 d2 -> 1 c d1 d2 ', c=len(comp_log_probs))

        target_means = target_means.to(device)
        target_covs = target_covs.to(device)
        # Intentionally leaving the comp_log_probs the same
        target_comp_logprobs = comp_log_probs_given_o
    elif target == 'indep_marginal_mog':
        # Not conditional on x_obs but all x_mis are independent
        target_means = repeat(means, 'c d -> 1 c d', c=len(comp_log_probs))
        target_covs = repeat(covs, 'c d1 d2 -> 1 c d1 d2', c=len(comp_log_probs))
        target_comp_logprobs = repeat(comp_log_probs, 'c -> b c', b=batch_size, c=len(comp_log_probs))

        # Set the off-diagonal elements of cov to 0.
        target_covs *= torch.eye(target_covs.shape[-1], device=target_covs.device)
    elif target == 'marginal_mog':
        # Not conditional on x_obs
        target_means = repeat(means, 'c d -> 1 c d', c=len(comp_log_probs))
        target_covs = repeat(covs, 'c d1 d2 -> 1 c d1 d2', c=len(comp_log_probs))
        target_comp_logprobs = repeat(comp_log_probs, 'c -> b c', b=batch_size, c=len(comp_log_probs))
    elif target == 'conditional_mog_with_mode_collapse_to_maxprob_mode':
        # Conditional on x_obs, but change the component probabilities such that only one mode (the one that initially had maximum probability) is non-zero
        target_means = means_m_given_o
        target_covs = covs_m_given_o
        non_zero_probs = torch.exp(comp_log_probs_given_o).max(-1)[1]
        target_comp_logprobs = torch.zeros_like(comp_log_probs_given_o)
        target_comp_logprobs[torch.arange(target_comp_logprobs.shape[0]), non_zero_probs] = 1.
        target_comp_logprobs = target_comp_logprobs.log()
    elif target == 'conditional_mog_with_mode_collapse_to_secondlargestprob_mode':
        # Conditional on x_obs, but change the component probabilities such that only one mode is non-zero
        target_means = means_m_given_o
        target_covs = covs_m_given_o

        second_largest = torch.topk(torch.exp(comp_log_probs_given_o), k=2, dim=-1)[1]
        second_largest = second_largest[:, 1]

        target_comp_logprobs = torch.zeros_like(comp_log_probs_given_o)
        target_comp_logprobs[torch.arange(target_comp_logprobs.shape[0]), second_largest] = 1.
        target_comp_logprobs = target_comp_logprobs.log()
    elif target == 'conditional_mog_with_mode_collapse_to_thirdlargestprob_mode':
        # Conditional on x_obs, but change the component probabilities such that only one mode is non-zero
        target_means = means_m_given_o
        target_covs = covs_m_given_o

        third_largest = torch.topk(torch.exp(comp_log_probs_given_o), k=3, dim=-1)[1]
        third_largest = third_largest[:, 2]

        target_comp_logprobs = torch.zeros_like(comp_log_probs_given_o)
        target_comp_logprobs[torch.arange(target_comp_logprobs.shape[0]), third_largest] = 1.
        target_comp_logprobs = target_comp_logprobs.log()
    elif target == 'conditional_mog_with_mode_collapse_to_fourthlargestprob_mode':
        # Conditional on x_obs, but change the component probabilities such that only one mode is non-zero
        target_means = means_m_given_o
        target_covs = covs_m_given_o

        fourth_largest = torch.topk(torch.exp(comp_log_probs_given_o), k=4, dim=-1)[1]
        fourth_largest = fourth_largest[:, 3]

        target_comp_logprobs = torch.zeros_like(comp_log_probs_given_o)
        target_comp_logprobs[torch.arange(target_comp_logprobs.shape[0]), fourth_largest] = 1.
        target_comp_logprobs = target_comp_logprobs.log()
    elif target == 'conditional_mog_with_mode_collapse_to_fifthlargestprob_mode':
        # Conditional on x_obs, but change the component probabilities such that only one mode is non-zero
        target_means = means_m_given_o
        target_covs = covs_m_given_o

        fifth_largest = torch.topk(torch.exp(comp_log_probs_given_o), k=5, dim=-1)[1]
        fifth_largest = fifth_largest[:, 4]

        target_comp_logprobs = torch.zeros_like(comp_log_probs_given_o)
        target_comp_logprobs[torch.arange(target_comp_logprobs.shape[0]), fifth_largest] = 1.
        target_comp_logprobs = target_comp_logprobs.log()
    elif target == 'conditional_mog_with_mode_collapse_to_firstnonzero_mode':
        # Conditional on x_obs, but change the component probabilities such that only one mode (the first that is non-zero initially) is non-zero
        target_means = means_m_given_o
        target_covs = covs_m_given_o

        def first_nonzero(x, axis=-1):
            nonz = (x > 0)
            return ((nonz.cumsum(axis) == 1) & nonz).max(axis)

        non_zero_probs = first_nonzero(torch.exp(comp_log_probs_given_o))[1]
        target_comp_logprobs = torch.zeros_like(comp_log_probs_given_o)
        target_comp_logprobs[torch.arange(target_comp_logprobs.shape[0]), non_zero_probs] = 1.
        target_comp_logprobs = target_comp_logprobs.log()
    elif target == 'conditional_mog_with_equal_component_probabilities':
        # Conditional on x_obs, but change the component probabilities (that were non-zero initially) have equal probability
        target_means = means_m_given_o
        target_covs = covs_m_given_o

        non_zero_probs = torch.exp(comp_log_probs_given_o) > 0.
        count_nonzero = non_zero_probs.sum(-1)
        target_comp_logprobs = torch.ones_like(comp_log_probs_given_o) * (1./count_nonzero.unsqueeze(-1))
        target_comp_logprobs *= non_zero_probs
        target_comp_logprobs = target_comp_logprobs.log()
    elif target == 'conditional_independent_mog_with_var_shrinkage_to_0.01var':
        target_means = means_m_given_o
        target_covs = covs_m_given_o * (0.01*torch.eye(covs_m_given_o.shape[-1], device=covs_m_given_o.device))
        target_comp_logprobs = comp_log_probs_given_o
    elif target == 'conditional_independent_mog':
        target_means = means_m_given_o
        target_covs = covs_m_given_o * (torch.eye(covs_m_given_o.shape[-1], device=covs_m_given_o.device))
        target_comp_logprobs = comp_log_probs_given_o
    elif target == 'conditional_independent_mog_created_from_true_indep_mog':
        # Set the off-diagonal elements of cov to 0.
        covs_indep = covs * torch.eye(covs.shape[-1], device=covs.device)
        target_comp_logprobs, target_means, target_covs = compute_conditional_mog_parameters(X, M, comp_log_probs, means, covs_indep)
    else:
        raise NotImplementedError(f'{target =} not implemented.')

    return target_comp_logprobs, target_means, target_covs

def construct_interpolated_conditional_mogs_from_true_mog(X, M, *, start_distribution,
                                                          interpolate_method, interpolate_target, alpha: float,
                                                          comp_log_probs, means, covs):
    B = X.shape[0]

    # Compute the conditional distribution for each data-point in the batch
    comp_log_probs_given_o, means_m_given_o, covs_m_given_o = compute_conditional_mog_parameters(X, M, comp_log_probs, means, covs)

    # Construct the target distribution
    target_comp_logprobs, target_means, target_covs = construct_target_distribution(
                                                            interpolate_target,
                                                            X, M,
                                                            comp_log_probs=comp_log_probs,
                                                            means=means,
                                                            covs=covs,
                                                            comp_log_probs_given_o=comp_log_probs_given_o,
                                                            means_m_given_o=means_m_given_o,
                                                            covs_m_given_o=covs_m_given_o,
                                                            batch_size=B,
                                                            device=X.device)

    # Ensure observed variable dimensions are zeroed in the target distribution
    M_not = ~M
    target_means = target_means * M_not.unsqueeze(1)
    target_covs = target_covs * M_not.unsqueeze(1).unsqueeze(-1) * M_not.unsqueeze(1).unsqueeze(-2)

    if start_distribution is None or start_distribution == 'true_conditional':
        # Do nothing
        pass
    else:
        # Override the true mog
        comp_log_probs_given_o, means_m_given_o, covs_m_given_o = construct_target_distribution(
                                    start_distribution,
                                    X, M,
                                    comp_log_probs=comp_log_probs,
                                    means=means, covs=covs,
                                    comp_log_probs_given_o=comp_log_probs_given_o,
                                    means_m_given_o=means_m_given_o,
                                    covs_m_given_o=covs_m_given_o,
                                    batch_size=B,
                                    device=X.device)

        # Ensure observed variable dimensions are zeroed in the target distribution
        means_m_given_o = means_m_given_o * M_not.unsqueeze(1)
        covs_m_given_o = covs_m_given_o * M_not.unsqueeze(1).unsqueeze(-1) * M_not.unsqueeze(1).unsqueeze(-2)


    # True conditional mog parameters
    cond_mog_params = {'comp_log_probs': comp_log_probs_given_o,
                        'means': means_m_given_o,
                        'covs': covs_m_given_o}
    # Anneal from the true MoG conditional to the target
    if interpolate_method == 'linear':
        annealed_params = linear_interpolate_sparse_mogs(cond_mog_params,
                                                        {'comp_log_probs': target_comp_logprobs,
                                                        'means': target_means,
                                                        'covs': target_covs},
                                                        alpha=alpha)
    elif interpolate_method == 'wasserstein':
        annealed_params = wasserstein_interpolate_sparse_mogs(M,
                                                              cond_mog_params,
                                                              {'comp_log_probs': target_comp_logprobs,
                                                               'means': target_means,
                                                               'covs': target_covs},
                                                              alpha=alpha)
    else:
        raise NotImplementedError(f'{interpolate_method =} not implemented.')

    return annealed_params

class VAESampler_MoG_Oracle(VAESampler_Base):
    def __init__(self,
                 start_distribution: str = 'true_conditional',
                 interpolate_method: str = 'linear',
                 interpolate_target: str = 'indep_marginal_gaussian',
                 interpolate_alpha: float = 1.0,
                 clip_imputations=False,
                 clipping_mode=None,
                 data_channel_dim=None):
        self.start_distribution = start_distribution
        self.interpolate_method = interpolate_method
        self.interpolate_target = interpolate_target
        self.interpolate_alpha = interpolate_alpha
        assert 0 <= self.interpolate_alpha <= 1, 'Interpolation coefficient must be between 0 and 1.'

        self.clip_imputations = clip_imputations
        self.clipping_mode = clipping_mode
        assert clip_imputations is False, 'Clip imputations is not implemented for MoG oracle sampler.'
        self.data_channel_dim = data_channel_dim

    def __call__(self, X, M, *, model, clip_values_min=None, clip_values_max=None):
        assert clip_values_min is None and clip_values_max is None, 'Clipping not implemented for MoG oracle sampler.'

        B, K = X.shape[0], X.shape[1]
        # Only using the first imputation (the observed values are repeated in the other imps,
        # and we don't need the past missing value imputations here)
        X = X[:, 0]
        M = M[:, 0]

        train_dataset = model.datamodule.train_data_core
        comp_log_probs = torch.tensor(train_dataset.data_file['comp_probs']).log().squeeze(0).to(X.device)
        means = torch.tensor(train_dataset.data_file['means']).to(X.device)
        covs = torch.tensor(train_dataset.data_file['covs']).to(X.device)

        annealed_params = construct_interpolated_conditional_mogs_from_true_mog(X, M,
                                                            start_distribution=self.start_distribution,
                                                            interpolate_method=self.interpolate_method,
                                                            interpolate_target=self.interpolate_target,
                                                            alpha=self.interpolate_alpha,
                                                            comp_log_probs=comp_log_probs,
                                                            means=means,
                                                            covs=covs)

        X_imp = sample_sparse_mog(K, M,
                              annealed_params['comp_log_probs'],
                              annealed_params['means'],
                              annealed_params['covs'],
                              sampling_batch_size=None)

        # Update values
        X = X.unsqueeze(1)*M.unsqueeze(1) + X_imp * (~M).unsqueeze(1)
        return X
