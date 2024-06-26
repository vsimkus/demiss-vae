import numpy as np
import torch
from einops import rearrange, reduce

from vgiwae.shared.neural_nets import FullyConnectedNetwork, ResidualFCNetwork
from vgiwae.shared.vae_imputation import VAESampler_Base

def create_2D_grid(min_v, max_v, bins=100):
    """Creates the grid for integrating in 2D"""
    x0, x1 = np.mgrid[min_v:max_v:(max_v-min_v)/bins,
                    min_v:max_v:(max_v-min_v)/bins]
    pos = np.empty(x0.shape + (2,))
    pos[:, :, 0] = x0
    pos[:, :, 1] = x1
    return x0, x1, torch.tensor(pos).float()


def create_Z(latent_dim, device=torch.device('cpu'), *, grid_min=None, grid_max=None, grid_steps=None):
    """Creates the "grid" for integrated in 1D or 2D"""
    if latent_dim == 1:
        min, max = -15, 15
        steps = 4000
        if grid_min is not None:
            min = grid_min
        if grid_max is not None:
            max = grid_max
        if grid_steps is not None:
            steps = grid_steps
        Z = rearrange(torch.linspace(min, max, steps=steps), 'z -> z 1')
        Z = Z.to(device)
        # "Volume" of each integrand
        dz = (max-min)/steps
        grid = None
    elif latent_dim == 2:
        min, max = -10, 10
        steps = 500
        if grid_min is not None:
            min = grid_min
        if grid_max is not None:
            max = grid_max
        if grid_steps is not None:
            steps = grid_steps
        z0, z1, Z = create_2D_grid(min, max, bins=steps)
        Z = rearrange(Z, 'z1 z2 d -> (z1 z2) d')
        Z = Z.to(device)
        # "Volume" of each integrand
        dz = ((max-min)/steps)**2
        grid = (z0, z1)
    else:
        raise NotImplementedError('Function not implemented for latent dimensionality greater than 2.')
    return Z, dz, grid

# def logTrapezoid2DExp(log_X, dx):
#     """Performs log-trapezoid-exp in a numerically safe way for 2D grids"""
#     # See e.g. https://math.stackexchange.com/questions/2891298/derivation-of-2d-trapezoid-rule
#     # Apply the 1D trapz two times.
#     steps = 500
#     log_X = rearrange(log_X, '(z1 z2) d -> z1 z2 d', z1=steps, z2=steps)
#     log_X[0, ...] -= torch.log(torch.tensor(2))
#     log_X[-1, ...] -= torch.log(torch.tensor(2))
#     log_X = torch.logsumexp(log_X, dim=0) - torch.log(torch.tensor(dx))
#     log_X[0, ...] -= torch.log(torch.tensor(2))
#     log_X[-1, ...] -= torch.log(torch.tensor(2))
#     log_X = torch.logsumexp(log_X, dim=0) - torch.log(torch.tensor(dx))
#     return log_X

def get_marginal_logprob(self, batch, *, compute_complete=False, marginal_eval_batchsize=-1, grid_min=None, grid_max=None, grid_steps=None):
    # Prepare grid
    latent_dim = None
    if isinstance(self.generator_network, FullyConnectedNetwork):
        latent_dim = self.generator_network.layer_dims[0]
    elif isinstance(self.generator_network, ResidualFCNetwork):
        latent_dim = self.generator_network.input_dim

    assert latent_dim <= 2,\
        'Cannot numerically integrate for dims > 2!'

    # Create the grid
    device = batch[0].device
    Z, dz, _ = create_Z(latent_dim, device=device, grid_min=grid_min, grid_max=grid_max, grid_steps=grid_steps)

    out =  get_marginal_logprob_for_Zgrid(self, batch, Z, dz,
                                          compute_complete=compute_complete,
                                          marginal_eval_batchsize=marginal_eval_batchsize)

    if compute_complete:
        return out['marginal_logprob_pxo'], out['complete_marginal_logprob_px']
    else:
        return out['marginal_logprob_pxo']

def get_posterior_divs(self, batch, imputed_batch, *, marginal_eval_batchsize=-1, grid_min=None, grid_max=None, grid_steps=None):
    # Prepare grid
    latent_dim = None
    if isinstance(self.generator_network, FullyConnectedNetwork):
        latent_dim = self.generator_network.layer_dims[0]
    elif isinstance(self.generator_network, ResidualFCNetwork):
        latent_dim = self.generator_network.input_dim

    assert latent_dim <= 2,\
        'Cannot numerically integrate for dims > 2!'

    # Create the grid
    device = batch[0].device
    Z, dz, _ = create_Z(latent_dim, device=device, grid_min=grid_min, grid_max=grid_max, grid_steps=grid_steps)

    # Compute shared parameters from Z
    # Compute prior logprob
    prior_dist = self.get_prior()
    prior_logprob = prior_dist.log_prob(Z)
    prior_logprob = reduce(prior_logprob, 'z ... d -> z 1 ...', 'sum')

    # Compute the parameters of the generator
    generator_params = self.generator_network(Z)
    generator_params = rearrange(generator_params, 'z ... pd -> z 1 ... pd')

    # Compute p(z|x)
    # out =  get_marginal_logprob_for_Zgrid(self, batch, Z, dz,
    #                                       compute_complete=True,
    #                                       return_joint=True,
    #                                       marginal_eval_batchsize=marginal_eval_batchsize)
    out = get_marginal_logprob_for_priorlogprobs_and_generator_params(
        self, batch, Z, dz, prior_logprob, generator_params,
        compute_complete=True, return_joint=True, marginal_eval_batchsize=marginal_eval_batchsize)

    complete_marginal_logprob_px = out['complete_marginal_logprob_px']
    complete_joint_logprob_pxz = out['complete_joint_logprob_pxz']
    log_pz_given_x = complete_joint_logprob_pxz - complete_marginal_logprob_px

    # Compute q(z|x)
    X, M = batch[:2]
    var_latent_params = self.predict_var_latent_params(X, M)
    var_latent_distribution = self.create_distribution(var_latent_params, self.hparams.var_latent_distribution)

    log_qz_given_x = var_latent_distribution.log_prob(Z.unsqueeze(1))
    log_qz_given_x = reduce(log_qz_given_x, 'z b d -> z b', 'sum')

    # Compute KL(q(z|x) || p(z|x))
    complete_kl_rev = torch.sum(torch.exp(log_qz_given_x)*(log_qz_given_x - log_pz_given_x)*dz, dim=0)
    # Compute KL(p(z|x) || q(z|x))
    complete_kl_fow = torch.sum(torch.exp(log_pz_given_x)*(log_pz_given_x - log_qz_given_x)*dz, dim=0)

    # Compute p(z|xo)
    marginal_logprob_px = out['marginal_logprob_pxo']
    joint_logprob_pxz = out['joint_logprob_pxoz']
    log_pz_given_xo = joint_logprob_pxz - marginal_logprob_px

    # Compute q(z|xo)
    X_imp, M_imp = imputed_batch[:2]
    var_latent_params = self.predict_var_latent_params(X_imp, M_imp)
    var_latent_distribution = self.create_distribution(var_latent_params, self.hparams.var_latent_distribution)

    log_qz_given_ximps = var_latent_distribution.log_prob(Z.unsqueeze(1).unsqueeze(1))
    log_qz_given_ximps = reduce(log_qz_given_ximps, 'z b k d -> z b k', 'sum')
    log_qz_given_xo = torch.logsumexp(log_qz_given_ximps, dim=2) - torch.log(torch.tensor(log_qz_given_ximps.shape[2], device=log_qz_given_ximps.device))

    # Compute KL(q(z|xo) || p(z|xo))
    incomplete_kl_rev = torch.sum(torch.exp(log_qz_given_xo)*(log_qz_given_xo - log_pz_given_xo)*dz, dim=0)
    # Compute KL(p(z|xo) || q(z|xo))
    incomplete_kl_fow = torch.sum(torch.exp(log_pz_given_xo)*(log_pz_given_xo - log_qz_given_xo)*dz, dim=0)

    # Compute JSD(q(z|x) || p(z|x))
    # log-avg-exp to compute log_pm, where pm is the midpoint distribution 0.5(p1 + p2)
    log_mz_given_x = torch.logsumexp(rearrange([log_qz_given_x, log_pz_given_x], 'p ... -> p ...'), dim=0) - torch.log(torch.tensor(2, device=log_pz_given_x.device, dtype=log_pz_given_x.dtype))
    complete_jsd = 0.5*(torch.sum(torch.exp(log_qz_given_x)*(log_qz_given_x - log_mz_given_x)*dz, dim=0) + torch.sum(torch.exp(log_pz_given_x)*(log_pz_given_x - log_mz_given_x)*dz, dim=0))

    # Compute JSD(q(z|xo) || p(z|xo))
    # log-avg-exp to compute log_pm, where pm is the midpoint distribution 0.5(p1 + p2)
    log_mz_given_xo = torch.logsumexp(rearrange([log_qz_given_xo, log_pz_given_xo], 'p ... -> p ...'), dim=0) - torch.log(torch.tensor(2, device=log_pz_given_xo.device, dtype=log_pz_given_xo.dtype))
    incomplete_jsd = 0.5*(torch.sum(torch.exp(log_qz_given_xo)*(log_qz_given_xo - log_mz_given_xo)*dz, dim=0) + torch.sum(torch.exp(log_pz_given_xo)*(log_pz_given_xo - log_mz_given_xo)*dz, dim=0))

    # Compute posterior conditional on imputed data
    # out_imputed =  get_marginal_logprob_for_Zgrid(self, imputed_batch, Z.unsqueeze(1), dz,
    #                                               compute_complete=True,
    #                                               return_joint=True,
    #                                               marginal_eval_batchsize=marginal_eval_batchsize)
    out_imputed = get_marginal_logprob_for_priorlogprobs_and_generator_params(
        self, imputed_batch, Z.unsqueeze(1), dz, prior_logprob.unsqueeze(1), generator_params.unsqueeze(1),
        compute_complete=True, return_joint=True, marginal_eval_batchsize=marginal_eval_batchsize)
    complete_marginal_logprob_pximps = out_imputed['complete_marginal_logprob_px']
    complete_joint_logprob_pximpsz = out_imputed['complete_joint_logprob_pxz']
    log_pz_given_ximps = complete_joint_logprob_pximpsz - complete_marginal_logprob_pximps

    # Compute KL(q(z|x_imp) || p(z|x_imp))
    complete_imps_kl_rev = torch.sum(torch.exp(log_qz_given_ximps)*(log_qz_given_ximps - log_pz_given_ximps)*dz, dim=0)
    # Compute KL(p(z|x_imp) || q(z|x_imp))
    complete_imps_kl_fow = torch.sum(torch.exp(log_pz_given_ximps)*(log_pz_given_ximps - log_qz_given_ximps)*dz, dim=0)
    # Compute JSD(q(z|x) || p(z|x))
    # log-avg-exp to compute log_pm, where pm is the midpoint distribution 0.5(p1 + p2)
    log_mz_given_ximps = torch.logsumexp(rearrange([log_qz_given_ximps, log_pz_given_ximps], 'p ... -> p ...'), dim=0) - torch.log(torch.tensor(2, device=log_pz_given_ximps.device, dtype=log_pz_given_ximps.dtype))
    complete_imps_jsd = 0.5*(torch.sum(torch.exp(log_qz_given_ximps)*(log_qz_given_ximps - log_mz_given_ximps)*dz, dim=0) + torch.sum(torch.exp(log_pz_given_ximps)*(log_pz_given_ximps - log_mz_given_ximps)*dz, dim=0))

    return complete_kl_fow, complete_kl_rev, complete_jsd, incomplete_kl_fow, incomplete_kl_rev, incomplete_jsd, complete_imps_kl_fow, complete_imps_kl_rev, complete_imps_jsd

def get_marginal_logprob_for_priorlogprobs_and_generator_params(self, batch, Z, dz, prior_logprob, generator_params, *, compute_complete=False, return_joint=False, marginal_eval_batchsize=-1):
    # Compute the conditional log-likelihood of each data-point
    generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution,
                                               validate_args=False) # Setting validate_args=False, because we don't want to check if the parameters are valid - if they aren't then the result will show.

    X, M = batch[:2]
    if marginal_eval_batchsize == -1:
        # Eval all in one go
        comp_cond_logprob = generator_distr.log_prob(X)
        cond_logprob = comp_cond_logprob*M
        cond_logprob = reduce(cond_logprob, 'z b ... d -> z b ...', 'sum')

        # Compute the marginal log_probability
        joint_logprob = prior_logprob + cond_logprob
        marginal_logprob = torch.logsumexp(joint_logprob + torch.log(torch.tensor(dz)), dim=0)
        # marginal_logprob = logTrapezoid2DExp(prior_logprob + cond_logprob, dx=dz**0.5)

        out = {'marginal_logprob_pxo': marginal_logprob}

        if return_joint:
            out['joint_logprob_pxoz'] = joint_logprob

        if compute_complete:
            # Compute marginal log_probability on complete data too
            comp_cond_logprob = reduce(comp_cond_logprob, 'z b ... d -> z b ...', 'sum')
            comp_joint_logprob = prior_logprob + comp_cond_logprob
            complete_marginal_logprob = torch.logsumexp(comp_joint_logprob + torch.log(torch.tensor(dz)), dim=0)

            out['complete_marginal_logprob_px'] = complete_marginal_logprob

            if return_joint:
                out['complete_joint_logprob_pxz'] = comp_joint_logprob

        return out
    else:
        # Eval in batches
        marginal_logprob = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
        if return_joint:
            joint_logprob = torch.empty(Z.shape[0], X.shape[0], dtype=X.dtype, device=X.device)
        if compute_complete:
            complete_marginal_logprob = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
            if return_joint:
                complete_joint_logprob = torch.empty(Z.shape[0], X.shape[0], dtype=X.dtype, device=X.device)

        indices = rearrange(torch.arange(X.shape[0] - X.shape[0] % marginal_eval_batchsize), '(b k) -> b k', k=marginal_eval_batchsize)
        for i in range(len(indices)):
            idx = indices[i]
            Xi, Mi = X[idx], M[idx]

            comp_cond_logprob_i = generator_distr.log_prob(Xi)
            cond_logprob_i = comp_cond_logprob_i*Mi
            cond_logprob_i = reduce(cond_logprob_i, 'z b ... d -> z b ...', 'sum')

            # Compute the marginal log_probability
            joint_logprob_i = prior_logprob + cond_logprob_i
            marginal_logprob[idx] = torch.logsumexp(joint_logprob_i + torch.log(torch.tensor(dz)), dim=0)
            if return_joint:
                joint_logprob[:, idx] = joint_logprob_i

            if compute_complete:
                # Compute marginal log_probability on complete data too
                comp_cond_logprob_i = reduce(comp_cond_logprob_i, 'z b ... d -> z b ...', 'sum')
                comp_joint_logprob_i = prior_logprob + comp_cond_logprob_i
                complete_marginal_logprob[idx] = torch.logsumexp(comp_joint_logprob_i + torch.log(torch.tensor(dz)), dim=0)
                if return_joint:
                    complete_joint_logprob[:, idx] = comp_joint_logprob_i

        # Eval for the rest of datapoints
        if X.shape[0] % marginal_eval_batchsize != 0:
            idx = torch.arange(X.shape[0] - X.shape[0] % marginal_eval_batchsize, X.shape[0])
            Xi, Mi = X[idx], M[idx]

            comp_cond_logprob_i = generator_distr.log_prob(Xi)
            cond_logprob_i = comp_cond_logprob_i*Mi
            cond_logprob_i = reduce(cond_logprob_i, 'z b ... d -> z b ...', 'sum')

            # Compute the marginal log_probability
            joint_logprob_i = prior_logprob + cond_logprob_i
            marginal_logprob[idx] = torch.logsumexp(joint_logprob_i + torch.log(torch.tensor(dz)), dim=0)
            if return_joint:
                joint_logprob[:, idx] = joint_logprob_i

            if compute_complete:
                # Compute marginal log_probability on complete data too
                comp_cond_logprob_i = reduce(comp_cond_logprob_i, 'z b ... d -> z b ...', 'sum')
                comp_joint_logprob_i = prior_logprob + comp_cond_logprob_i
                complete_marginal_logprob[idx] = torch.logsumexp(comp_joint_logprob_i + torch.log(torch.tensor(dz)), dim=0)
                if return_joint:
                    complete_joint_logprob[:, idx] = comp_joint_logprob_i

        out = {'marginal_logprob_pxo': marginal_logprob}
        if return_joint:
            out['joint_logprob_pxoz'] = joint_logprob
        if compute_complete:
            out['complete_marginal_logprob_px'] = complete_marginal_logprob
            if return_joint:
                out['complete_joint_logprob_pxz'] = complete_joint_logprob

        return out

def get_marginal_logprob_for_Zgrid(self, batch, Z, dz, *, compute_complete=False, return_joint=False, marginal_eval_batchsize=-1):
    # Compute prior logprob
    prior_dist = self.get_prior()
    prior_logprob = prior_dist.log_prob(Z)
    prior_logprob = reduce(prior_logprob, 'z ... d -> z 1 ...', 'sum')

    # Compute the parameters of the generator
    generator_params = self.generator_network(Z)
    generator_params = rearrange(generator_params, 'z ... pd -> z 1 ... pd')

    out = get_marginal_logprob_for_priorlogprobs_and_generator_params(
        self, batch, Z, dz, prior_logprob, generator_params,
        compute_complete=compute_complete, return_joint=return_joint, marginal_eval_batchsize=marginal_eval_batchsize)
    return out

def sample_pz_given_xo_with_rejection_sampling(self, X, M, num_samples, joint_logprob_max_envelope, Z_dim, Z_batchsize, Z_min, Z_max):
    # TODO: implement this more efficiently.
    samples = torch.empty(joint_logprob_max_envelope.shape[0], num_samples, Z_dim, device=self.device)
    for i in range(len(joint_logprob_max_envelope)):
        found_samples = 0
        while found_samples < num_samples:
            Z = torch.rand(Z_batchsize, Z_dim, device=self.device)*(Z_max-Z_min) + Z_min

            # Compute prior logprob
            prior_dist = self.get_prior()
            prior_logprob = prior_dist.log_prob(Z)
            prior_logprob = reduce(prior_logprob, 'z d -> z', 'sum')

            # Compute the parameters of the generator
            generator_params = self.generator_network(Z)
            generator_params = rearrange(generator_params, 'z pd -> z pd')
            # Compute the conditional log-likelihood of each data-point
            generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)
            comp_cond_logprob = generator_distr.log_prob(X[i])
            cond_logprob = comp_cond_logprob*M[i]
            cond_logprob = cond_logprob.sum(dim=-1)

            # Compute the joint
            joint_logprob = prior_logprob + cond_logprob
            joint_prob = torch.exp(joint_logprob)

            # Rejection sampling
            u = torch.rand(Z_batchsize, device=self.device)*torch.exp(joint_logprob_max_envelope[i])
            accepted = (u < joint_prob)

            #
            Z_accepted = Z[accepted]
#             print('Accepted:', Z_accepted.shape)
            num_accepted_samples = Z_accepted.shape[0]

            samples[i, found_samples:min(found_samples+num_accepted_samples, num_samples)] = \
                Z_accepted[:num_samples-found_samples]
            found_samples += num_accepted_samples
    return samples

class VAESampler_Rejection(VAESampler_Base):
    def __init__(self,
                 Z_batchsize,
                 envelope_multiplier=1.0,
                 grid_min=None,
                 grid_max=None,
                 grid_steps=None,
                 clip_imputations=False,
                 clipping_mode=None,
                 data_channel_dim=None):
        self.Z_batchsize = Z_batchsize
        self.envelope_multiplier = envelope_multiplier

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_steps = grid_steps

        self.clip_imputations = clip_imputations
        self.clipping_mode = clipping_mode
        assert clip_imputations is False, 'Clip imputations is not implemented for rejection sampler.'
        self.data_channel_dim = data_channel_dim

    def __call__(self, X, M, *, model, clip_values_min=None, clip_values_max=None):
        assert clip_values_min is None and clip_values_max is None, 'Clipping not implemented for rejection sampler.'

        B, K = X.shape[0], X.shape[1]
        # Only using the first imputation (the observed values are repeated in the other imps,
        # and we don't need the past missing value imputations here)
        X = X[:, 0]
        M = M[:, 0]

        latent_dim = None
        if isinstance(model.generator_network, FullyConnectedNetwork):
            latent_dim = model.generator_network.layer_dims[0]
        elif isinstance(model.generator_network, ResidualFCNetwork):
            latent_dim = model.generator_network.input_dim

        # logprob_out = get_marginal_logprob(model, (X, M), compute_complete=True, marginal_eval_batchsize=-1,
        #                                    grid_min=self.grid_min, grid_max=self.grid_max, grid_steps=self.grid_steps)
        # Create the grid
        Z, dz, _ = create_Z(latent_dim, device=X.device,
                            grid_min=self.grid_min, grid_max=self.grid_max, grid_steps=self.grid_steps)
        logprob_out = get_marginal_logprob_for_Zgrid(model, (X, M), Z, dz,
                                                     compute_complete=False, return_joint=True, marginal_eval_batchsize=-1)

        # Compute the maximum envelope of the joint log-probability
        joint_logprob_max_envelope = torch.max(logprob_out['joint_logprob_pxoz'], dim=0)[0]
        # Multiply the envelope by the multiplier (in probability space, so sum in log-space)
        joint_logprob_max_envelope = joint_logprob_max_envelope + torch.log(torch.tensor(self.envelope_multiplier, device=joint_logprob_max_envelope.device, dtype=joint_logprob_max_envelope.dtype))

        # Sample p(z|x_o) with rejection sampling
        Z_samples = sample_pz_given_xo_with_rejection_sampling(model, X, M, K, joint_logprob_max_envelope,
                                                               Z_dim=latent_dim, Z_batchsize=self.Z_batchsize,
                                                               Z_min=self.grid_min, Z_max=self.grid_max)

        # Compute the parameters of the generator
        generator_params = model.generator_network(Z_samples)
        generator_distr = model.create_distribution(generator_params, model.hparams.generator_distribution)
        X_imp = generator_distr.sample()

        # Update values
        X = X.unsqueeze(1)*M.unsqueeze(1) + X_imp * (~M).unsqueeze(1)
        return X
