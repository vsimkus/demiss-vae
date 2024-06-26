import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange, repeat


# Collate function to concat copies of sample into batch
def collate_augmented_samples(batch):
    return tuple((torch.cat([torch.as_tensor(sample)
                             for sample in samples])
                  for samples in zip(*batch)))


class DataAugmentation(Dataset):
    """
    Wrapper that augments incomplete samples.
    Maintains the order of input dataset, but the incomplete samples
    are repeated num_copies times.
    """
    def __init__(self, dataset, num_copies, augment_complete=True):
        self.dataset = dataset
        self.augment_complete = augment_complete

        # Get the original data
        all_data = dataset[:]
        I = np.arange(len(dataset))
        X, M = all_data[0], all_data[1]

        if self.augment_complete:
            self.aug_data = repeat(X, 'n d -> (k n) d', k=num_copies)
            self.missing_mask = repeat(M, 'n d -> (k n) d', k=num_copies)
            self.original_idx = repeat(I, 'n -> (k n)', k=num_copies)
            # self.sample_incomp_mask = np.tile(sample_incomp_mask, (num_copies))

            self.aug_idx = rearrange(np.arange(0, self.aug_data.shape[0]), '(k n) -> n k', k=num_copies)
            self.value_mask = np.ones_like(self.aug_idx, dtype=np.bool)
        else:
            # Set the observed data
            self.aug_data = X
            self.missing_mask = M
            self.original_idx = I

            sample_incomp_mask = M.sum(axis=1) != X.shape[-1]
            if isinstance(sample_incomp_mask, torch.Tensor):
                sample_incomp_mask = sample_incomp_mask.numpy()

            # Create augmented copies
            X_augmented = repeat(X[sample_incomp_mask, ...], 'n d -> (k n) d', k=num_copies-1)
            M_augmented = repeat(M[sample_incomp_mask, ...], 'n d -> (k n) d', k=num_copies-1)
            I_augmented = repeat(I[sample_incomp_mask], 'n -> (k n)', k=num_copies-1)
            # incomp_mask_augmented = np.ones_like(I_augmented, dtype=np.bool)
            self.aug_data = np.append(self.aug_data, X_augmented, axis=0)
            self.missing_mask = np.append(self.missing_mask, M_augmented, axis=0)
            self.original_idx = np.append(self.original_idx, I_augmented, axis=0)

            # Augmentation reference index
            # So that we can lookup all copies of a datapoint
            # E.g. for a dataset
            #        X                 M
            # [[0, 1, 2, 3],    [[1, 1, 1, 0],
            #  [1, 1, 1, 5],     [0, 0, 1, 1],
            #  [6, 9, 1, 2]]     [1, 1, 1, 1]]
            #
            #    aug_data
            # [[0, 1, 2, 3],
            #  [1, 1, 1, 5],
            #  [6, 9, 1, 2],
            #  [0, 1, 2, 3],
            #  [1, 1, 1, 5]
            #  [0, 1, 2, 3],
            #  [1, 1, 1, 5]]
            #
            # create aug_idx:
            # [[0, 3, 5], # The copies for 0th sample are in indices 0, 3, 5
            #  [1, 4, 6], # The copies for 1st sample are in indices 1, 4, 6
            #  [2, X, X]] # The 2nd sample doesn't have copies, where
            # X will be larger than the dataset, so that using it would cause
            # an error.
            self.placeholder_val = self.aug_data.shape[0] + 1
            self.aug_idx = np.full((X.shape[0], num_copies),
                                # Set initial value to larger than the dataset
                                self.placeholder_val,
                                dtype=np.long)
            self.aug_idx[:, 0] = np.arange(X.shape[0])

            if num_copies > 1:
                # self.aug_idx[sample_incomp_mask, 1:] = \
                #     np.arange(X.shape[0],
                #             X.shape[0] + sample_incomp_mask.sum()*(num_copies-1))\
                #     .reshape(-1, num_copies-1, order='F')
                self.aug_idx[sample_incomp_mask, 1:] = \
                    rearrange(np.arange(X.shape[0], X.shape[0] + sample_incomp_mask.sum()*(num_copies-1))
                              , '(k n) -> n k', k=num_copies-1)


            # Cache a mask, which shows which values in the aug_idx are _not_
            # placeholders. I.e. shows that the element is either true value,
            # or an augmentation, but not a placeholder index.
            self.value_mask = self.aug_idx != self.placeholder_val

            # Store a cache mask indicating which samples are incomplete
            # self.sample_incomp_mask = np.append(sample_incomp_mask, incomp_mask_augmented, axis=0)


    def __getitem__(self, idx):
        # Augmented indices of this sample
        if self.augment_complete:
            indices = self.aug_idx[idx].flatten()
        else:
            mask = self.value_mask[idx]
            indices = self.aug_idx[idx][mask].flatten()

        return (self.aug_data[indices],
                self.missing_mask[indices],
                indices,
                self.original_idx[indices])

    def __setitem__(self, key, value):
        self.aug_data[key] = value

    def __len__(self):
        return len(self.dataset)

    def augmented_len(self):
        return len(self.aug_data)

    def unaugmented_data(self):
        indices = self.aug_idx[:, 0]
        return (self.aug_data[indices],
                self.missing_mask[indices],
                indices,
                self.original_idx[indices])
