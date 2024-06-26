import torch
from torch.utils.data import Dataset


class TopHalfMissingnessDataset(Dataset):
    """
    Generates missing data in the given dataset.
    The missingness is denoted by adding additional tensor M,
    whose size is the same as X.

    Args:
        dataset:        Fully-observed PyTorch dataset
        target_idx:     If the dataset returns tuples, then this should be the index
                        of the target data in the tuple for which the missing mask is added.
        total_miss:     Total fraction of values to be made missing
        rng:            Torch random number generator.
    """
    def __init__(self,
                 dataset: Dataset,
                 target_idx: int = 0,
                 total_miss: float = 0.00,
                 rng: torch.Generator = None):
        super().__init__()
        self.dataset = dataset
        self.target_idx = target_idx
        self.total_miss = total_miss
        self.rng = rng

        data = self._get_target_data()
        self.miss_mask = self.generate_mis_mask(data)

    def _get_target_data(self):
        data = self.dataset[:]
        if isinstance(data, tuple):
            # Get the data for which the missing masks are generated
            data = data[self.target_idx]

        # if isinstance(data, np.ndarray):
        #     data = torch.tensor(data)

        return data

    def generate_mis_mask(self, data):
        dim = data[0].shape[-1]

        miss_mask = torch.ones(*(data.shape), dtype=torch.bool)
        miss_mask[:, :int(dim*self.total_miss)] = 0

        return miss_mask.numpy()

    def __getitem__(self, idx):
        data = self.dataset[idx]
        miss_mask = self.miss_mask[idx]
        if isinstance(data, tuple):
            # Insert missingness mask after the target_idx tensor to which it corresponds
            data = (data[:self.target_idx+1]
                    + (miss_mask,)
                    + data[self.target_idx+1:])
        else:
            data = (data, miss_mask)

        return data

    def __setitem__(self, key, value):
        self.dataset[key] = value

    def __len__(self):
        return len(self.dataset)
