import numpy as np
import torch
import scipy

from vgiwae.shared.neural_nets import ResidualFCNetwork
from vgiwae.shared.vae_resnet import ResNetEncoder


def compute_fid_score(refs, samples, *, inception_model):
    """
    Computes the FID score between two sets of samples.

    Args:
        refs:               Reference samples.
        samples:            Samples from the model to compare to the reference.
        inception_model:    The inception model to use for feature extraction.
    """
    with torch.inference_mode():
        ref_feat_mean, ref_feat_cov = compute_inception_gaussian(refs, inception_model=inception_model)
        samples_feat_mean, samples_feat_cov = compute_inception_gaussian(samples, inception_model=inception_model)

        fid = gaussian_fid_score(ref_feat_mean, ref_feat_cov, samples_feat_mean, samples_feat_cov)

    return fid

def compute_inception_gaussian(samples, *, inception_model, batch_size=5000):
    """
    Computes the Gaussian distribution of the features given by an inception model.

    Args:
        samples:            Samples.
        inception_model:    The inception model to use for feature extraction.
    """
    with torch.inference_mode():
        feats_samples = []
        for b in range(0, samples.shape[0], batch_size):
            samples_b = samples[b:min(b+batch_size, samples.shape[0])]
            if isinstance(inception_model, ResidualFCNetwork):
                feats_samples_b = inception_model(samples_b)
            # elif isinstance(inception_model, ResNetClassifier):
            #     _, feats_samples_b = inception_model(samples_b)
            elif isinstance(inception_model, ResNetEncoder):
                feats_samples_b = inception_model(samples_b)
            else:
                raise NotImplementedError(f'Unknown inception model type: {type(inception_model)}')
            feats_samples.append(feats_samples_b)
        feats_samples = torch.cat(feats_samples, dim=0)

        samples_feat_mean, samples_feat_cov = torch.mean(feats_samples, dim=0), torch.cov(feats_samples.T)

    return samples_feat_mean, samples_feat_cov

def gaussian_fid_score(ref_mean, ref_cov, samples_mean, samples_cov):
    """
    Computes the FID score two Gaussians.
    """
    # FID is equivalent to squared Wasserstein distance between Gaussians
    sqrt = scipy.linalg.sqrtm(samples_cov @ ref_cov)
    if sqrt.dtype in (np.complex64, np.complex128):
        sqrt = sqrt.real
    sqrt = torch.tensor(sqrt)
    fid = torch.norm(ref_mean - samples_mean, p=2)**2 + torch.trace(ref_cov + samples_cov - 2*sqrt)

    return fid
