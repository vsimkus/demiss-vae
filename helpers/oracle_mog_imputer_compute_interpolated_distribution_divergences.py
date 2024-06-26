import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningArgumentParser
from pytorch_lightning.utilities.seed import seed_everything, reset_seed

import os
import tqdm
import torch
import numpy as np

from vgiwae.data import MissingDataModule
from vgiwae.utils.arg_utils import construct_experiment_subdir
from vgiwae.utils.mog_utils import (construct_interpolated_conditional_mogs_from_true_mog,
                                    compute_conditional_mog_parameters,
                                    compute_kl_divs_and_jsd_for_sparse_mogs)

def build_argparser():
    parser = LightningArgumentParser('Computes the interpolated distribution divergences for the oracle imputer',
                                     parse_as_dict=False)

    # Add general arguments
    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)
    parser.add_argument('--experiment_subdir_base', type=str, required=True,
        help='Experiment subdirectory.')

    parser.add_argument('--start_distribution', type=str, default='true_conditional',
                        help=('The distribution to start the interpolation from'))
    parser.add_argument('--interpolate_method', type=str, required=True,
                        help=('The method to use for the interpolation'))
    parser.add_argument('--interpolate_target', type=str, required=True,
                        help=('The target distribution for the interpolation'))
    parser.add_argument('--interpolate_alpha', type=float, required=True,
                        help=('The alpha parameter for the interpolation'))
    parser.add_argument('--divergence_num_samples', type=int, required=True,
                        help=('The number of samples to use for the divergence computation'))

    # Add class arguments
    parser.add_lightning_class_args(MissingDataModule, 'data')
    # Note use `python train.py --model=vgiwae.models.VAE --print_config`
    # to print a config for a specific model class

    return parser

def run(hparams):
    # Set random seed
    # NOTE: this must be done before any class initialisation,
    # hence also before the call to parser.instantiate_classes()
    seed_everything(hparams.seed_everything, workers=True)

    # Construct the experiment directory
    experiment_subdir = construct_experiment_subdir(hparams)
    experiment_dir = f'./{experiment_subdir}/lightning_logs/'
    # if hparams.trainer.default_root_dir is None:
    #     experiment_dir = f'./{experiment_subdir}'
    # else:
    #     experiment_dir = f'{hparams.trainer.default_root_dir}/{experiment_subdir}'
    print('Experiment directory:', experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    # Instantiate dynamic object classes
    hparams = parser.instantiate_classes(hparams)

    # Get the instantiated data
    datamodule = hparams.data

    datamodule.setup('fit')
    datamodule.setup('test')

    # The instantiation steps might be different for different models.
    # Hence we reset the seed before training such that the seed at the start of lightning setup is the same.
    reset_seed()

    train_dataset = datamodule.train_data_core
    comp_log_probs = torch.tensor(train_dataset.data_file['comp_probs']).log().squeeze(0)
    means = torch.tensor(train_dataset.data_file['means'])
    covs = torch.tensor(train_dataset.data_file['covs'])

    dataloader = datamodule.train_dataloader()
    all_kldivs_fow = []
    all_kldivs_rev = []
    all_jsds = []
    for batch in tqdm.tqdm(dataloader):
        X, M = batch[:2]

        # Compute the ground-truth conditionals
        comp_log_probs_given_o, means_m_given_o, covs_m_given_o = \
            compute_conditional_mog_parameters(X, M,
                                               comp_log_probs=comp_log_probs,
                                               means=means,
                                               covs=covs)

        # Compute the oracle approximations
        cond_params_approx = \
            construct_interpolated_conditional_mogs_from_true_mog(X, M,
                                                                  start_distribution=hparams.start_distribution,
                                                                  interpolate_method=hparams.interpolate_method,
                                                                  interpolate_target=hparams.interpolate_target,
                                                                  alpha=hparams.interpolate_alpha,
                                                                  comp_log_probs=comp_log_probs,
                                                                  means=means,
                                                                  covs=covs)

        cond_params_true = {
            'comp_log_probs': comp_log_probs_given_o,
            'means': means_m_given_o,
            'covs': covs_m_given_o
        }
        # cond_params_approx ={
        #     'comp_log_probs': comp_log_probs_given_o_approx,
        #     'means': means_m_given_o_approx,
        #     'covs': covs_m_given_o_approx
        # }
        # Compute the divergences between the ground-truth and oracle approximations
        kldivs_fow, kldivs_rev, jsds = compute_kl_divs_and_jsd_for_sparse_mogs(cond_params_approx, cond_params_true, M,
                                                                               num_kl_samples=hparams.divergence_num_samples)

        all_kldivs_fow.append(kldivs_fow)
        all_kldivs_rev.append(kldivs_rev)
        all_jsds.append(jsds)

    all_kldivs_fow = torch.concat(all_kldivs_fow)
    all_kldivs_rev = torch.concat(all_kldivs_rev)
    all_jsds = torch.concat(all_jsds)

    kldiv_fow = all_kldivs_fow.mean()
    kldiv_rev = all_kldivs_rev.mean()
    jsd = all_jsds.mean()

    np.savez(experiment_dir + '/oracle_imputer_interpolated_distribution_divergences.npz',
             kldiv_fow=kldiv_fow,
             kldiv_rev=kldiv_rev,
             jsd=jsd)


if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
