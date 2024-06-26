
import os
from enum import Enum, auto
from typing import Optional, List

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader

from .dataset_with_indices import DatasetWithIndices
from .fully_missing_filter_dataset import FullyMissingDataFilter
from .mnist import MNIST
from .omniglot import Omniglot
from .toy import ToyDataset
from .uniform_missingness_provider import UniformMissingnessDataset
from .tophalf_missingness_provider import TopHalfMissingnessDataset
from .quadrant_missingness_provider import QuadrantMissingnessDataset
from .uci_gas import UCI_GAS
from .uci_power import UCI_POWER
from .uci_hepmass import UCI_HEPMASS
from .uci_miniboone import UCI_MINIBOONE

class DATASET(Enum):
    toy_mog = auto()
    toy_vae = auto()
    toy_mog2 = auto()
    toy_mog2_large = auto()
    toy_mog3_large = auto()
    mnist = auto()
    mnist_bin = auto()
    mnist_14x14 = auto()
    mnist_5x5 = auto()
    mnist_7x7 = auto()
    uci_gas = auto()
    uci_power = auto()
    uci_hepmass = auto()
    uci_miniboone = auto()
    omniglot_bin = auto()
    omniglot_28x28_bin = auto()

    def __call__(self, *args, **kwargs):
        if self.name == 'toy_mog':
            return ToyDataset(*args, filename='data_mog', **kwargs)
        elif self.name == 'toy_vae':
            return ToyDataset(*args, filename='data_vae', **kwargs)
        elif self.name == 'toy_mog2':
            return ToyDataset(*args, filename='data_mog2', **kwargs)
        elif self.name == 'toy_mog2_large':
            return ToyDataset(*args, filename='data_mog2_large', **kwargs)
        elif self.name == 'toy_mog3_large':
            return ToyDataset(*args, filename='data_mog3_large', **kwargs)
        elif self.name == 'mnist':
            return MNIST(*args, **kwargs)
        elif self.name == 'mnist_bin':
            return MNIST(*args, binarise=True, **kwargs)
        elif self.name == 'mnist_14x14':
            return MNIST(*args, transform=torchvision.transforms.Resize((14,14)), **kwargs)
        elif self.name == 'mnist_5x5':
            return MNIST(*args, transform=torchvision.transforms.Resize((5,5)), **kwargs)
        elif self.name == 'mnist_7x7':
            return MNIST(*args, transform=torchvision.transforms.Resize((7,7)), **kwargs)
        elif self.name == 'omniglot_bin':
            return Omniglot(*args, binarise_fixed=True, **kwargs)
        elif self.name == 'omniglot_28x28_bin':
            return Omniglot(*args, transform=torchvision.transforms.Resize((28,28)), binarise_fixed=True, **kwargs)
        elif self.name == 'uci_gas':
            return UCI_GAS(*args, **kwargs)
        elif self.name == 'uci_power':
            return UCI_POWER(*args, **kwargs)
        elif self.name == 'uci_hepmass':
            return UCI_HEPMASS(*args, **kwargs)
        elif self.name == 'uci_miniboone':
            return UCI_MINIBOONE(*args, **kwargs)
        else:
            raise NotImplementedError()

class MISSINGNESS(Enum):
    uniform = 'uniform'
    top_half = 'top_half'
    quadrants = 'quadrants'

train_missingness_fn = {
    "uniform": lambda dataset, hparams, rng: \
        UniformMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_train, rng=rng),
    "top_half": lambda dataset, hparams, rng: \
        TopHalfMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_train, rng=rng),
    "quadrants": lambda dataset, hparams, rng: \
        QuadrantMissingnessDataset(dataset, target_idx=0, img_dims=hparams.img_dims, total_miss=hparams.total_miss_train, rng=rng),
}

val_missingness_fn = {
    "uniform": lambda dataset, hparams, rng: \
        UniformMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_val, rng=rng),
    "top_half": lambda dataset, hparams, rng: \
        TopHalfMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_val, rng=rng),
    "quadrants": lambda dataset, hparams, rng: \
        QuadrantMissingnessDataset(dataset, target_idx=0, img_dims=hparams.img_dims, total_miss=hparams.total_miss_val, rng=rng),
}

test_missingness_fn = {
    "uniform": lambda dataset, hparams, rng: \
        UniformMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_test, rng=rng),
    "top_half": lambda dataset, hparams, rng: \
        TopHalfMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_test, rng=rng),
    "quadrants": lambda dataset, hparams, rng: \
        QuadrantMissingnessDataset(dataset, target_idx=0, img_dims=hparams.img_dims, total_miss=hparams.total_miss_test, rng=rng),
}

def impute_with_mean(dataset, rng=None):
    """
    Computes the empirical mean of each dimension, using only observed values.
    Then, replaces the missing values with the empirical mean for that
    dimension.
    """
    # Compute observed mean for each dimension
    # Use observed unaugmented data
    X, M = dataset[:][:2]
    count_observed = M.sum(axis=0)
    sum_observed = (X * M).sum(axis=0)
    means = sum_observed / count_observed

    # Replace missing values with empirical means
    X = X*M + means*~M
    dataset[:] = X

    return means

def impute_with_zero(dataset, rng=None):
    X, M = dataset[:][:2]
    X = X*M
    dataset[:] = X

def impute_with_empirical_distribution_sample(dataset, rng=None):
    """
    Computes the empirical distribution of each dimension, using only observed
    values. Then, replaces the missing values with random sample from the
    empirical distribution.
    """
    X, M = dataset[:][:2]

    M_not = ~M
    count_missing_per_dim = (M_not).sum(axis=0)
    # Impute each dimension
    for i in range(M.shape[-1]):
        if count_missing_per_dim[i] == 0:
            continue
        # samples = np.random.choice(X[:, i][M[:, i]],
        #                            size=count_missing_per_dim[i])
        samples = torch.multinomial(M[:, i], count_missing_per_dim[i], generator=rng)
        samples = X[:, i][samples]
        X[M_not[:, i], i] = samples

    # Set imputed data
    dataset[:] = X


class PRE_IMPUTATION(Enum):
    oracle = 'oracle'
    mean = 'mean'
    zero = 'zero'
    empirical = 'empirical'

pre_imputation_fn = {
    'oracle': lambda dataset, rng: None,
    'mean': impute_with_mean,
    'zero': impute_with_zero,
    'empirical': impute_with_empirical_distribution_sample,
}

class MissingDataModule(pl.LightningDataModule):
    """
    Missing Data provider module adding synthetic missingness to complete data.

    Args:
        dataset:                        Which dataset to load.
        batch_size:                     Size of the training and validation mini-batches.
        missingess:                     Missingness type.
        total_miss_train:               Total missing fraction in training data.
        total_miss_val:                 Total missing fraction in validation data.
        pre_imputation:                 What to do with missing values at the start.
        preimpute_val:                  Whether to preimpute missing values in validation data or not.
        filter_fully_missing_train:     Remove fully missing datapoints from the training data.
        filter_fully_missing_val:       Remove fully missing datapoints from the validation data.
        use_test_instead_val:           Loads test data instead of val data in "validation" loader.
        img_dims:                       Image dimensions. Used in some missingness providers.
        data_root:                      Root directory of all datasets.
        setup_seed:                     Random seed for the setup.
        num_workers:                    The number of dataloader workers. If None, chooses the smaller
                                        between 8 and the number of CPU cores on the machine.

        total_miss_test:                Total missing fraction in test data.
        use_test_instead_train:         Loads test data instead of train data in "train" loader.
        filter_fully_missing_test:      Remove fully missing datapoints from the test data.
        test_batch_size:                Size of the test mini-batches.
    """
    def __init__(self,
                 dataset: DATASET,
                 batch_size: int,
                 missingness: MISSINGNESS, # TODO: load by subclass instead?
                 total_miss_train: float,         # TODO: load by subclass instead?
                 total_miss_val: float,         # TODO: load by subclass instead?
                 pre_imputation: PRE_IMPUTATION,
                 pre_impute_val: bool,
                 filter_fully_missing_train: bool = True,
                 filter_fully_missing_val: bool = True,
                 use_test_instead_val: bool = False,
                 img_dims: List[int] = None,
                 data_root: str = "./data",
                 setup_seed: int = None,
                 num_workers: int = None,
                 total_miss_test: float = 0.,         # TODO: load by subclass instead?
                 use_test_instead_train: bool = False,
                 filter_fully_missing_test: bool = True,
                 test_batch_size: int = None,
                 ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        rng = torch.Generator()
        rng = rng.manual_seed(self.hparams.setup_seed)

        if stage == 'fit':
            # Load train and validation splits
            rng_dataset = torch.Generator()
            rng_dataset.manual_seed(int(self.hparams.setup_seed*2.33/2))
            self.train_data = self.hparams.dataset(self.hparams.data_root, split='train', rng=rng_dataset)
            self.train_data_core = self.train_data
            val_split = 'val'
            if self.hparams.use_test_instead_val:
                print('Using test data in val dataloader!')
                val_split = 'test'
            self.val_data = self.hparams.dataset(self.hparams.data_root, split=val_split, rng=rng_dataset)
            self.val_data_core = self.val_data

            # Initialise missingness
            train_init_miss = train_missingness_fn[self.hparams.missingness.value]
            self.train_data = train_init_miss(self.train_data, self.hparams, rng=rng)
            val_init_miss = val_missingness_fn[self.hparams.missingness.value]
            self.val_data = val_init_miss(self.val_data, self.hparams, rng=rng)

            # Filter fully missing datapoints
            if self.hparams.filter_fully_missing_train:
                self.train_data = FullyMissingDataFilter(self.train_data, miss_mask_idx=1)
            if self.hparams.filter_fully_missing_val:
                self.val_data = FullyMissingDataFilter(self.val_data, miss_mask_idx=1)

            # Augment datapoints
            self.train_data = DatasetWithIndices(self.train_data)
            self.val_data = DatasetWithIndices(self.val_data)

            # Initialise missing values with pre-imputation
            pre_imputation_method = pre_imputation_fn[self.hparams.pre_imputation.value]
            pre_imputation_method(self.train_data, rng=rng)
            if self.hparams.pre_impute_val:
                pre_imputation_method(self.val_data, rng=rng)

            print('Train data size:', len(self.train_data))
            print('Validation data size:', len(self.val_data))

            if self.hparams.use_test_instead_train:
                print('Using test data in train dataloader!')
                self.setup(stage='test')
                self.train_dat_core = self.test_data_core
                self.train_data = self.test_data

        elif stage == 'test':
            # Load test split
            rng_dataset = torch.Generator()
            rng_dataset.manual_seed(int(self.hparams.setup_seed*2.33/2))
            self.test_data = self.hparams.dataset(self.hparams.data_root, split='test', rng=rng_dataset)
            self.test_data_core = self.test_data

            # Initialise missingness
            test_init_miss = test_missingness_fn[self.hparams.missingness.value]
            self.test_data = test_init_miss(self.test_data, self.hparams, rng=rng)

            if self.hparams.filter_fully_missing_test:
                self.test_data = FullyMissingDataFilter(self.test_data, miss_mask_idx=1)

            # Augment datapoints
            self.test_data = DatasetWithIndices(self.test_data)

            print('Test data size:', len(self.test_data))
        else:
            raise NotImplementedError(f'{stage=} is not implemented.')

    def train_dataloader(self):
        num_workers = min(8, os.cpu_count()) if self.hparams.num_workers is None else self.hparams.num_workers
        return DataLoader(self.train_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          shuffle=True,
        )

    def val_dataloader(self):
        num_workers = min(8, os.cpu_count()) if self.hparams.num_workers is None else self.hparams.num_workers
        return DataLoader(self.val_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          shuffle=False,
        )

    def test_dataloader(self):
        num_workers = min(8, os.cpu_count()) if self.hparams.num_workers is None else self.hparams.num_workers
        batch_size = self.hparams.test_batch_size if self.hparams.test_batch_size is not None else self.hparams.batch_size
        return DataLoader(self.test_data,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=False,
        )
