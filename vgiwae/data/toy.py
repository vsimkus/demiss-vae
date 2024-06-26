
import os
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import scipy.stats
import torch
import torch.utils.data as data

from vgiwae.utils.incomplete_data_multivariate_normal import IncompleteDataMultivariateNormal


def generate_mog_model(num_components: int, dims: int, standardise: bool = True):
    """
    Generates a random Mixture of Gaussians model

    Args:
        num_components:   Number of components in the mixture
        dims:             Dimensionality of the data
        standardise:      If True standardise the marginal distribution of the mixture to unit variance
    Returns:

    """
    # Generate component mixture
    comp_probs_truth = scipy.stats.dirichlet.rvs(
        np.ones(num_components), size=1)[0]
    # comp_logits_truth = np.log(comp_probs_truth)

    # Generate Gaussians
    covs_truth = scipy.stats.invwishart.rvs(
        df=dims, scale=np.eye(dims), size=num_components)
    means_truth = np.random.randn(num_components, dims)*3

    if standardise:
        # Compute the mean of the MoG
        mean = (means_truth*comp_probs_truth[:, None]).sum(0)
        # Compute the covariance of the MoG
        dif = means_truth - mean
        cov = covs_truth + dif[..., None] @ dif[:, None, :]
        cov *= comp_probs_truth[:, None, None]
        cov = cov.sum(0)

        # Get standard deviation of marginals
        std = np.diagonal(cov)**(0.5)

        # Standardise the covariance matrices
        L = (np.linalg.cholesky(covs_truth)/std[None, :, None])
        covs_truth = L @ L.transpose((0, 2, 1))

        # Standardise the means
        means_truth = (means_truth-mean)/std

    comp_probs_truth = torch.tensor(comp_probs_truth).float()
    means_truth = torch.tensor(means_truth).float()
    covs_truth = torch.tensor(covs_truth).float()

    return comp_probs_truth, means_truth, covs_truth

def sample_mog(num_samples, comp_probs, means, *, covs=None, scale_trils=None):
    mix = torch.distributions.Categorical(probs=comp_probs)
    multi_norms = torch.distributions.MultivariateNormal(
        loc=means, covariance_matrix=covs, scale_tril=scale_trils)
    comp = torch.distributions.Independent(multi_norms, 0)
    mog = torch.distributions.MixtureSameFamily(mix, comp)

    return mog.sample(sample_shape=(num_samples,))

def mog_log_prob_miss(X, M, *, comp_probs, means, covs=None, scale_trils=None):
    X = X.unsqueeze(-2)
    M = M.unsqueeze(-2)

    mix = torch.distributions.Categorical(probs=comp_probs)
    incomp_multi_norms = IncompleteDataMultivariateNormal(means, covariance_matrix=covs, scale_tril=scale_trils)

    log_prob_x = incomp_multi_norms.log_prob_mis(X, M)
    log_mix_prob = torch.log_softmax(mix.logits, dim=-1)
    return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)


def create_and_save_dataset(num_samples, dims, num_components, root_dir='./data', filename='data_mog.mat'):
    # Generate a Mixture-of-Gaussians distribution
    comp_probs, means, covs = generate_mog_model(num_components, dims, standardise=True)

    # Generate samples
    data_train = sample_mog(int(num_samples*0.9), comp_probs, means, covs=covs)
    data_val = sample_mog(int(num_samples*0.1), comp_probs, means, covs=covs)
    data_test = sample_mog(num_samples, comp_probs, means, covs=covs)

    data = {
        "train": data_train.numpy(),
        "val": data_val.numpy(),
        "test": data_test.numpy(),
        "comp_probs": comp_probs.numpy(),
        "means": means.numpy(),
        "covs": covs.numpy()
    }
    filename = os.path.join(root_dir, 'toy', filename)

    # Save and return samples
    sio.savemat(file_name=filename, mdict=data)

    return data

def create_and_save_larger_dataset(num_samples_train, original_filename, root_dir='./data', filename='data_mog_large.mat'):
    # Generate a Mixture-of-Gaussians distribution
    data_file = sio.loadmat(os.path.join(root_dir, 'toy', original_filename))
    comp_probs = torch.tensor(data_file['comp_probs']).flatten()
    means = torch.tensor(data_file['means'])
    covs = torch.tensor(data_file['covs'])

    # Generate samples
    data_train = sample_mog(int(num_samples_train), comp_probs, means, covs=covs)

    data = {
        "train": data_train.numpy(),
        "val": data_file['val'],
        "test": data_file['test'],
        "comp_probs": comp_probs.numpy(),
        "means": means.numpy(),
        "covs": covs.numpy()
    }
    filename = os.path.join(root_dir, 'toy', filename)

    # Save and return samples
    sio.savemat(file_name=filename, mdict=data)

    return data

def plot_data_pairwise(data):
    dims = data.shape[-1]
    min_v, max_v = data.min(), data.max()

    fig = plt.figure(figsize=(9,9))
    grid_spec = mpl.gridspec.GridSpec(ncols=dims-1, nrows=dims-1, figure=fig)
    for i in range(0, dims):
        for j in range(i+1, dims):
            # Plot pairs of dimensions
            ax = fig.add_subplot(grid_spec[i, j-1])

            ax.scatter(data[:, i], data[:, j], alpha=0.7)

            # Set common limits
            ax.set_ylim(min_v, max_v)
            ax.set_xlim(min_v, max_v)

            if i == 0:
                ax.set_title(f'dim={j}', fontsize=12)
            if j == 1:
                ax.set_ylabel(f'dim={i}', fontsize=12)

        # Create dummy axes to add labels on the left-hand side
        if 0 < i < (dims-1):
            ax = fig.add_subplot(dims-1, dims-1, i*(dims-1)+1)
            ax.set_ylabel(f'dim={i}', fontsize=12)
            # Hide the dummy axes
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)
            ax.patch.set_visible(False)

    grid_spec.tight_layout(fig)

    return fig


class ToyDataset(data.Dataset):
    """
    A dataset wrapper
    """

    def __init__(self, root: str, filename='data_mog',
                 split: str = 'train',
                 rng: torch.Generator = None):
        """
        Args:
            root:       root directory that contains all data
            filename:   filename of the data file
            split:      data split, e.g. train, val, test
            rng:        random number generator
        """
        super().__init__()

        root = os.path.expanduser(root)
        filename = os.path.join(root, 'toy', f'{filename}.mat')

        # Load Toy dataset
        self.data_file = sio.loadmat(filename)
        self.data = self.data_file[split]

        self.data_min = np.min(self.data, axis=0)
        self.data_max = np.max(self.data, axis=0)

    def __getitem__(self, index):
        """
        Args:
            index: index of sample
        Returns:
            image: dataset sample
        """
        return self.data[index]

    def __setitem__(self, key, value):
        """
        Args:
            key: index of sample
        """
        self.data[key] = value

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    filename='data_mog2'
    dataset = ToyDataset(root='./data', filename=filename, split='train')
    X = dataset[:]

    print(X.shape)

    fig = plot_data_pairwise(X)
    plt.show()

    # plt.savefig(f'./data/toy/{filename}_pairwise_plots.pdf')
