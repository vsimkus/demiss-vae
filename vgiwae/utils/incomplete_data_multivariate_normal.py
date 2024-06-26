import math

import torch


class IncompleteDataMultivariateNormal(torch.distributions.MultivariateNormal):
    def log_prob_mis(self, value: torch.Tensor, mis: torch.Tensor):
        """
        Computes the marginal log-probability of the observed data, where the observed
        values are indicated by a missingness mask.

        NOTE: The implementation can be slow and less numerically accurate since it
        doesn't use cholesky decomposition but works with the covariance matrix for
        ease of implementation

        TODO: See if there are cholesky decomposition algorithms that can decompose sparse
        matrices with 0-columns and 0-rows.

        Args:
            value:  A tensor of values where some may be missing
            mis:    A boolean tensor indicating which values are missing in the value tensor.
                    If 1 then observed, else if 0 then missing.
        """
        if self._validate_args:
            self._validate_sample(value)
            self._validate_sample(mis)

        # Get the number of observed dims
        num_obs_dims = mis.sum(-1)

        not_mis = ~mis

        # Get the difference vector of observed values,
        # set missing values to zero
        value_obs = value * mis
        loc_obs = self.loc * mis
        diff_obs = value_obs - loc_obs

        # Get the observed covariance matrix for each datapoint
        # We replace missing columns/rows with zeros
        # and then create an auxilliary matrix with 1s in the missing diagonal places to ensure non-singularity
        # this does not affect the log-determinant or the inverse.
        sigma_obs = self.covariance_matrix * mis[..., None, :] * mis[..., :, None]
        sigma_obs_aux = sigma_obs + torch.eye(self.event_shape[0])[(None,)*len(self.batch_shape)]*not_mis[..., None, :] * not_mis[..., :, None]

        # Triangular solve would probably be more accurate if we had a Cholesky decomposition of the covariance matrix for each datapoint
        mahalanobis = (diff_obs[..., None, :] @ torch.linalg.solve(sigma_obs_aux, diff_obs[..., :, None]))[..., 0, 0]

        return -0.5 * (num_obs_dims*math.log(2 * math.pi) + torch.logdet(sigma_obs_aux) + mahalanobis)


if __name__ == '__main__':
    # TODO: Refactor into unit tests

    # Quick test for the case where we have # distributions == # datapoints

    # Create distributions
    D = 5
    means = torch.randn(11, D)
    covs = torch.randn(11, D, D)
    covs = covs @ covs.transpose(-1, -2)
    norm = IncompleteDataMultivariateNormal(loc=means, covariance_matrix=covs)

    # Create some data
    M = torch.rand(11, D) > 0.6
    X = norm.sample()

    # Eval implementation
    logprob = norm.log_prob_mis(X, M)

    # Check each value is correct for each datapoint
    for i in range(len(X)):
        xi, mi = X[i], M[i]

        meani = means[i]
        covi = covs[i]

        xi_obs = xi[mi]
        meani_obs = meani[mi]
        covi_obs = covi[mi, :][:, mi]

        normi = torch.distributions.MultivariateNormal(loc=meani_obs, covariance_matrix=covi_obs)

        logprobi = normi.log_prob(xi_obs) if mi.any() else torch.tensor(0.)

        assert torch.allclose(logprobi, logprob[i]),\
            'IncompleteDataMultivariateNormal gives different result that MultivariateNormal.'

    # Quick test where the # distributions < # datapoints

    # Create distributions
    D = 5
    mean = torch.randn(D)
    cov = torch.randn(D, D)
    cov = cov @ cov.T
    norm = IncompleteDataMultivariateNormal(loc=mean, covariance_matrix=cov)

    # Create some data
    M = torch.rand(11, D) > 0.6
    X = torch.randn(11, D)

    # Eval implementation
    logprob = norm.log_prob_mis(X, M)

    # Check each value is correct for each datapoint
    for i in range(len(X)):
        xi, mi = X[i], M[i]

        xi_obs = xi[mi]
        mean_obs = mean[mi]
        cov_obs = cov[mi, :][:, mi]

        norm_obs = torch.distributions.MultivariateNormal(loc=mean_obs, covariance_matrix=cov_obs)

        logprobi = norm_obs.log_prob(xi_obs) if mi.any() else torch.tensor(0.)

        assert torch.allclose(logprobi, logprob[i]),\
            'IncompleteDataMultivariateNormal gives different result that MultivariateNormal.'

    # Quick test where # distributions > # datapoints

    # Create distributions
    D = 5
    means = torch.randn(11, D)
    covs = torch.randn(11, D, D)
    covs = covs @ covs.transpose(-1, -2)
    norm = IncompleteDataMultivariateNormal(loc=means, covariance_matrix=covs)

    # Create some data
    M = torch.rand(D) > 0.6
    X = torch.randn(D)

    # Eval implementation
    logprob = norm.log_prob_mis(X, M)

    # Check each value is correct for each distribution
    for i in range(len(means)):
        x_obs = X[M]
        meani_obs = means[i][M]
        covi_obs = covs[i][M, :][:, M]

        normi_obs = torch.distributions.MultivariateNormal(loc=meani_obs, covariance_matrix=covi_obs)

        logprobi = normi_obs.log_prob(x_obs) if M.any() else torch.tensor(0.)

        assert torch.allclose(logprobi, logprob[i]),\
            'IncompleteDataMultivariateNormal gives different result that MultivariateNormal.'
