import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from einops import rearrange, reduce, asnumpy

from vgiwae.shared.neural_nets import FullyConnectedNetwork, ResidualFCNetwork
from vgiwae.shared.vae_marginal_logprob import create_Z


class VAE_plot_posterior:
    def plot_posterior(self, batch):
        X, M = batch[:2]

        latent_dim = self.latent_dim

        assert latent_dim <= 2,\
            'Cannot numerically integrate for dims > 2!'

        # Create the grid
        Z, dz, grid = create_Z(latent_dim, device=X.device)

        # Compute prior logprob
        prior_dist = torch.distributions.Normal(loc=0, scale=1.)
        prior_logprob = prior_dist.log_prob(Z)
        prior_logprob = reduce(prior_logprob, 'z d -> z 1', 'sum')

        # Compute the parameters of the generator
        generator_mean, generator_logvar = self.compute_generator_params(Z)
        raise NotImplementedError('This needs to be updated to used softplus activatios.')
        generator_mean = rearrange(generator_mean, 'z d -> z 1 d')
        generator_logvar = rearrange(generator_logvar, 'z d -> z 1 d')

        # Compute the conditional log-likelihood of each data-point
        generator_distr = torch.distributions.Normal(loc=generator_mean, scale=torch.exp(generator_logvar/2))
        cond_logprob = generator_distr.log_prob(X)*M
        cond_logprob = reduce(cond_logprob, 'z b d -> z b', 'sum')

        # Compute the marginal log_probability
        marginal_logprob = torch.logsumexp(prior_logprob + cond_logprob + torch.log(torch.tensor(dz)), dim=0)

        # Compute posterior logprob
        posterior_logprob = cond_logprob - marginal_logprob
        posterior_prob = torch.exp(posterior_logprob)

        if latent_dim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(9,9))
            posterior_prob = rearrange(posterior_prob, '(z0 z1) b -> z0 z1 b', z0=grid[0].shape[0])
            posterior_prob = asnumpy(posterior_prob)

            cmap = mpl.cm.get_cmap('gist_ncar')
            colors = cmap(np.linspace(0.05, 1, posterior_prob.shape[2]))
            for p in range(posterior_prob.shape[2]):
                ax.contour(grid[0], grid[1], posterior_prob[:, :, p], levels=3, colors=mpl.colors.to_hex(colors[p], keep_alpha=False))

        else:
            raise NotImplementedError('Function not implemented for latent dimensionality greater than 2 or less than 1.')

        return fig

