# Improving Variational Autoencoder Estimation from Incomplete Data with Mixture Variational Families

This repository contains the research code for

> Vaidotas Simkus, Michael U. Gutmann. Improving Variational Autoencoder Estimation from Incomplete Data with Mixture Variational Families. Transactions on Machine Learning Research, 2024.

The paper can be found here: <https://openreview.net/forum?id=lLVmIvZfry>.

The code is shared for reproducibility purposes and is not intended for production use.

## Abstract

We consider the task of estimating variational autoencoders (VAEs) when the training data is incomplete. We show that missing data increases the complexity of the model’s posterior distribution over the latent variables compared to the fully-observed case. The increased complexity may adversely affect the fit of the model due to a mismatch between the variational and model posterior distributions. We introduce two strategies based on (i) finite variational-mixture and (ii) imputation-based variational-mixture distributions to address the increased posterior complexity. Through a comprehensive evaluation of the proposed approaches, we show that variational mixtures are effective at improving the accuracy of VAE estimation from incomplete data.

## Dependencies

Install python dependencies from conda and the `vgiwae` project package with

```bash
conda env create -f environment.yml
conda activate vgiwae
python setup.py develop
```

If the dependencies in `environment.yml` change, update dependencies with

```bash
conda env update --file environment.yml
```

## Organisation of the code

* `./vgiwae/data/` contains data loaders and missingness generators.
* `./vgiwae/models/` contains the model implementations.
  * `mvbvae.py` contains the implementations of the DeMissVAE method in the paper.
  * `vae.py` and `iwae.py` contains the implementations of MVAE, MIWAE, MissVAE, and MissIWAE
  * `multiple_vae.py` and `multiple_iwae.py` contains the implementaions of MissSVAE and MissSIWAE.
* `./configs/` contains the yaml configuration files containing all the information about each experiment.
* `./notebooks/` contain analysis notebooks that produce the figures in the paper.

## Running the code

Activate the conda environment

```bash
conda activate vgiwae
```

### VAE training

To train the VAE, which we use for sampling run e.g.

```bash
python train.py --config=configs/uci_gas/mis50/iwae_i5_encm_stl.yaml
```

### VAE marginal log-likelihood estimation

Use `refit_encoder_using_iwae_and_estimate_loglik.py` to estimate the marginal log-likelihood

```bash
python refit_encoder_using_iwae_and_estimate_loglik.py --config=configs/uci_gas/refit_encoder_testcomp/mis50/iwae_i5_encm_stl.yaml
```
