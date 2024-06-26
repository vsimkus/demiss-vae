from enum import Enum
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat, asnumpy


class IMPUTATION_INIT(Enum):
    oracle = 'oracle'
    mean = 'mean'
    zero = 'zero'
    empirical = 'empirical'
    empirical_parallel = 'empirical_parallel'


class ImputationDistribution(nn.Module):
    """
    An empirical distribution representing imputations of the data.
    Requires the original dataset to provide datapoint indices, such
    that this module can map those indices to imputations.

    Args:
        dataset:            The original dataset.
        target_idx:         The index of the element in the tuple provided by the
                            dataset that represents the data we want to impute.
        mis_idx:            The index of the element in the tuple provided by the
                            dataset that represents the missingness mask.
        index_idx:          The index of the datapoint index in the tuple provided
                            by the dataset.
        num_imputations:    The number of imputations to maintain for each datapoint.
        imputation_init:    Initialisation method for imputations.
    """
    def __init__(self,
                 dataset: Tuple[torch.utils.data.Dataset, tuple],
                 target_idx: int,
                 mis_idx: int,
                 index_idx: int,
                 num_imputations: int,
                 imputation_init: IMPUTATION_INIT):
        super().__init__()
        assert not isinstance(dataset, torch.utils.data.IterableDataset),\
            'Iterable-type datasets are not supported!'

        self.dataset = dataset
        self.target_idx = target_idx
        self.mis_idx = mis_idx
        self.index_idx = index_idx
        self.num_imputations = num_imputations
        self.imputation_init = imputation_init

        # Get the original data
        # TODO: handle datasets which don't support slicing
        if isinstance(dataset, torch.utils.data.Dataset):
            all_data = dataset[:]
        elif isinstance(dataset, tuple):
            all_data = dataset
        else:
            raise NotImplementedError()
        data = all_data[target_idx]
        index = all_data[index_idx]
        assert index.dtype == torch.long or index.dtype == np.int64,\
            'The dataset index is not of type long!'

        # Sort the data using the index in case it comes shuffled.
        data = data[index]

        # Convert numpy arrays to tensors
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)

        # Repeat each datapoint #num_imputations times
        # NOTE: this also repeats complete datapoints, however in higher dimensions
        # most of the datapoints will have some incomplete dimensions and hence it won't
        # create a lot of overhead
        self.data_imp = repeat(data, 'n d -> (n k) d', k=num_imputations)
        self.idx_imp = rearrange(torch.arange(0, len(data)*num_imputations),
                                 '(n k) -> n k', k=num_imputations)
        # We will append the imputation index at the end of the tuple
        self.imp_idx = len(all_data)

    def save_imputations(self, path):
        # Workaround for storing a tuple of numpy arrays
        all_data = self.dataset[:]
        all_data = {f'dataset_{i}': t for i, t in enumerate(all_data)}
        np.savez_compressed(path,
                            **all_data,
                            target_idx=self.target_idx,
                            mis_idx=self.mis_idx,
                            index_idx=self.index_idx,
                            num_imputations=self.num_imputations,
                            imputation_init=self.imputation_init,
                            data_imps=self.data_imp,
                            imp_idx=self.imp_idx,
                            )

    @staticmethod
    def load_from_save(path):
        data = np.load(path, allow_pickle=True)
        dataset_keys = [k for k in data.keys() if k.startswith('dataset_')]
        dataset_index = [int(k.split('dataset_')[1]) for k in dataset_keys]
        sorted_index = sorted(dataset_index)
        dataset_keys = [f'dataset_{i}' for i in sorted_index]
        dataset_tuple = tuple(torch.tensor(data[k]) for k in dataset_keys)
        imp_distr = ImputationDistribution(
            dataset=dataset_tuple,
            target_idx=data['target_idx'].item(),
            mis_idx=data['mis_idx'].item(),
            index_idx=data['index_idx'].item(),
            num_imputations=data['num_imputations'].item(),
            imputation_init=data['imputation_init']
        )
        imp_distr.data_imp = torch.tensor(data['data_imps'])
        imp_distr.imp_idx = torch.tensor(data['imp_idx'])
        return imp_distr

    def init_imputations(self):
        # import time

        # start_time = time.time()
        # for _ in range(10000):
        #     impute_with_empirical_distribution_sample(self.dataset, self, self.target_idx, self.mis_idx)
        # end_time = time.time()

        # print((end_time-start_time)*1e3, 'ms')

        # start_time = time.time()
        # for _ in range(10000):
        #     impute_with_empirical_distribution_sample_parallel(self.dataset, self, self.target_idx, self.mis_idx)
        # end_time = time.time()

        # print((end_time-start_time)*1e3, 'ms')
        # breakpoint()

        imputation_fn = imputation_init_fn[self.imputation_init.value]
        imputation_fn(self.dataset, self, self.target_idx, self.mis_idx)

    def get_imputed_datapoints(self, batch: Tuple[torch.Tensor, ...], num_imputations: int = None) -> Tuple[torch.Tensor, ...]:
        # Get target elements from the batch tuple
        data = batch[self.target_idx]
        index = batch[self.index_idx]

        # Get the imputation indices from the original index
        idx_imp = self.idx_imp[index]
        if num_imputations is not None:
            batch_size = index.shape[0]
            sampled_i = torch.multinomial(torch.ones((batch_size, self.num_imputations), device=self.data_imp.device),
                                          num_samples=num_imputations,
                                          replacement=False)
            idx_imp = idx_imp[torch.arange(batch_size)[:, None], sampled_i]

        # Get the imputed data
        data_imp = self.data_imp[rearrange(idx_imp, 'b k -> (b k)')]
        data_imp = rearrange(data_imp, '(b k) ... -> b k ...',
                             b=len(data),
                             k=num_imputations if num_imputations is not None else self.num_imputations)

        # Put the imputation tensors on the same device as the original batch
        if isinstance(data, torch.Tensor):
            data_imp = data_imp.to(data.device)
        if isinstance(index, torch.Tensor):
            idx_imp = idx_imp.to(index.device)

        # Replace the target data element with augmented data
        # repeat the other elements #num_imputations times
        # and append augmentation index at the end
        imputed_batch = list(batch)
        for i in range(len(imputed_batch)):
            if i == self.target_idx:
                imputed_batch[i] = data_imp
                continue
            imputed_batch[i] = repeat(imputed_batch[i], 'b ... -> b k ...',
                                      k=num_imputations if num_imputations is not None else self.num_imputations)
        imputed_batch.insert(self.imp_idx, idx_imp)
        return tuple(imputed_batch)

    def set_imputed_datapoints(self, batch: Tuple[torch.Tensor, ...]):
        # Get the imputed data
        data_imp = batch[self.target_idx].detach().cpu()
        data_imp = rearrange(data_imp, 'b k ... -> (b k) ...')
        # Get the imputation indices (they were appended to the end of the batch tuple)
        idx_imp = batch[self.imp_idx].detach().cpu()
        idx_imp = rearrange(idx_imp, 'b k -> (b k)')

        # Set the updated data imputations
        self.data_imp[idx_imp] = data_imp

    # def get_imputed_datapoints_by_sample(self, batch: Tuple[torch.Tensor, ...], num_imputations: int = None) -> Tuple[torch.Tensor, ...]:
    #     imputed_batch = self.get_imputed_datapoints(batch, num_imputations=num_imputations)
    #     return self.rearrange_imputed_batch_by_sample(imputed_batch, batch_size=len(batch[0]))

    # def set_imputed_datapoints_by_sample(self, batch: Tuple[torch.Tensor, ...]):
    #     imputed_batch = self.derearrange_imputed_batch(batch)
    #     self.set_imputed_datapoints(imputed_batch)

    # def rearrange_imputed_batch_by_sample(self,
    #                                       batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
    #                                       *,
    #                                       batch_size: int):
    #     rearranged_batch = []
    #     for tensor in batch:
    #         rearranged_batch.append(rearrange(tensor, '(b k) ... -> b k ...', b=batch_size))

    #     return tuple(rearranged_batch)

    # def derearrange_imputed_batch(self, batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]):
    #     rearranged_batch = []
    #     for tensor in batch:
    #         rearranged_batch.append(rearrange(tensor, 'b k ... -> (b k) ...'))

    #     return tuple(rearranged_batch)

class ImputationDistributionWithLatents(ImputationDistribution):
    """
    Adds an additional tensor for latents imputations.

    Args:
        latent_dim: Dimensionality of the latent variables.
    """
    def __init__(self,
                 *args,
                 latent_dim: int = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim

        # Create latent imputations, default to standard Normal initialisation for simplicity
        self.latent_imp = torch.randn(self.data_imp.shape[0], self.latent_dim, dtype=self.data_imp.dtype)

        # We will append the latents after the imputation index in the batch tuple
        self.latent_idx = self.imp_idx + 1

    def get_imputed_datapoints(self, batch: Tuple[torch.Tensor, ...], num_imputations: int = None) -> Tuple[torch.Tensor, ...]:
        # Run parent class method
        imputed_batch = super().get_imputed_datapoints(batch, num_imputations=num_imputations)

        # Get the imputation indices from the imputed batch tuple
        idx_imp = imputed_batch[self.imp_idx]
        # Get the imputed latents
        latent_imp = self.latent_imp[rearrange(idx_imp, 'b k -> (b k)')]
        latent_imp = rearrange(latent_imp, '(b k) ... -> b k ...',
                             b=len(batch[self.target_idx]),
                             k=num_imputations if num_imputations is not None else self.num_imputations)

        # Put the imputation tensors on the same device as the original batch
        if isinstance(imputed_batch[0], torch.Tensor):
            latent_imp = latent_imp.to(imputed_batch[0].device)

        # Append the latents at the end
        imputed_batch = list(imputed_batch)
        imputed_batch.insert(self.latent_idx, latent_imp)
        return tuple(imputed_batch)

    def set_imputed_datapoints(self, batch: Tuple[torch.Tensor, ...]):
        super().set_imputed_datapoints(batch)

        # Get the imputed latents
        latent_imp = batch[self.latent_idx].detach().cpu()
        latent_imp = rearrange(latent_imp, 'b k ... -> (b k) ...')
        # Get the imputation indices (they were appended to the end of the batch tuple)
        idx_imp = batch[self.imp_idx].detach().cpu()
        idx_imp = rearrange(idx_imp, 'b k -> (b k)')

        # Set the updated data imputations
        self.latent_imp[idx_imp] = latent_imp


#
# Initial imputation methods
# TODO: Move these elsewhere, also resolve duplication with the missing_data_module.py if possible
#

def impute_with_mean(original_dataset: torch.utils.data.Dataset,
                     imp_distribution: ImputationDistribution,
                     target_idx: int,
                     mis_idx: int):
    """
    Computes the empirical mean of each dimension, using only observed values.
    Then, replaces the missing values with the empirical mean for that
    dimension.
    """
    # Compute observed mean for each dimension
    all_data = original_dataset[:]
    data, mis = all_data[target_idx], all_data[mis_idx]
    count_observed = mis.sum(axis=0)
    sum_observed = (data * mis).sum(axis=0)
    means = sum_observed / count_observed
    if isinstance(means, np.ndarray):
        means = torch.tensor(means)
    # means = rearrange(means, 'b ... -> b 1 ...')

    # Replace missing imputation values with empirical means
    imputed_batch = imp_distribution.get_imputed_datapoints(all_data)
    data_imp, mis_imp = imputed_batch[target_idx], imputed_batch[mis_idx]
    data_imp = data_imp*mis_imp + means*~mis_imp
    imp_distribution.set_imputed_datapoints(imputed_batch)

    return means

def impute_with_zero(original_dataset: torch.utils.data.Dataset,
                     imp_distribution: ImputationDistribution,
                     target_idx: int,
                     mis_idx: int):
    all_data = original_dataset[:]
    imputed_batch = imp_distribution.get_imputed_datapoints(all_data)
    data_imp, mis_imp = imputed_batch[target_idx], imputed_batch[mis_idx]
    data_imp = data_imp*mis_imp
    imp_distribution.set_imputed_datapoints(imputed_batch)

def impute_with_empirical_distribution_sample(original_dataset: torch.utils.data.Dataset,
                                              imp_distribution: ImputationDistribution,
                                              target_idx: int,
                                              mis_idx: int):
    """
    Computes the empirical distribution of each dimension, using only observed
    values. Then, replaces the missing values with random sample from the
    empirical distribution.
    """
    all_data = original_dataset[:]
    data, mis = all_data[target_idx], all_data[mis_idx]
    imputed_batch = imp_distribution.get_imputed_datapoints(all_data)
    data_imp, mis_imp = imputed_batch[target_idx], imputed_batch[mis_idx]
    batch_size = len(data_imp)
    data_imp = rearrange(data_imp, 'b k ... -> (b k) ...')
    mis_imp = rearrange(mis_imp, 'b k ... -> (b k) ...')

    mis_imp_not = ~mis_imp
    count_missing_per_dim = (mis_imp_not).sum(axis=0)
    # Impute each dimension
    for i in range(data.shape[-1]):
        if count_missing_per_dim[i] == 0:
            continue
        samples = np.random.choice(data[:, i][mis[:, i]],
                                   size=count_missing_per_dim[i])
        if isinstance(samples, np.ndarray):
            samples = torch.tensor(samples)
        data_imp[mis_imp_not[:, i], i] = samples

    # Set imputed data
    imputed_batch = list(imputed_batch)
    imputed_batch[target_idx] = rearrange(data_imp, '(b k) ... -> b k ...', b=batch_size)
    imp_distribution.set_imputed_datapoints(imputed_batch)

def impute_with_empirical_distribution_sample_parallel(original_dataset: torch.utils.data.Dataset,
                                                       imp_distribution: ImputationDistribution,
                                                       target_idx: int,
                                                       mis_idx: int):
    """
    Computes the empirical distribution of each dimension, using only observed
    values. Then, replaces the missing values with random sample from the
    empirical distribution.

    Does the same as `impute_with_empirical_distribution_sample` but parallelises the loop,
    might be faster for higher dim data (it is not faster for low-dim data)
    """
    all_data = original_dataset[:]
    data, mis = all_data[target_idx], all_data[mis_idx]
    imputed_batch = imp_distribution.get_imputed_datapoints(all_data)
    data_imp, mis_imp = imputed_batch[target_idx], imputed_batch[mis_idx]
    batch_size = len(data_imp)
    data_imp = rearrange(data_imp, 'b k ... -> (b k) ...')
    mis_imp = rearrange(mis_imp, 'b k ... -> (b k) ...')

    # Impute each dimension
    mis = torch.tensor(mis)
    # Create uniform distribution over the observed values for each dimension
    cat = torch.distributions.Categorical(probs=(mis/mis.sum(0)).T)
    # Sample it N*K times
    idx = cat.sample(sample_shape=(len(data_imp),))  # (N*K, D)
    imps = torch.gather(torch.tensor(data), 0, idx)  # (N*K, D)
    # assert torch.tensor(mis)[idx, torch.arange(data_imp.shape[-1])].sum() == mis.numel()
    data_imp = data_imp*mis_imp + imps*(~mis_imp)

    # Set imputed data
    imputed_batch = list(imputed_batch)
    imputed_batch[target_idx] = rearrange(data_imp, '(b k) ... -> b k ...', b=batch_size)
    imp_distribution.set_imputed_datapoints(imputed_batch)

imputation_init_fn = {
    'oracle': lambda *args: None,
    'mean': impute_with_mean,
    'zero': impute_with_zero,
    'empirical': impute_with_empirical_distribution_sample,
    'empirical_parallel': impute_with_empirical_distribution_sample_parallel,
}
