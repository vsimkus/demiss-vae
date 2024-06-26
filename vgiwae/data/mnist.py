import os.path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image


def save_splits():
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True)
    data = dataset.data
    targets = dataset.targets

    N_train = int(0.9 * len(data))
    data_train, targets_train = data[:N_train], targets[:N_train]
    data_val, targets_val = data[N_train:], targets[N_train:]

    train = {
        'data': data_train,
        'targets': targets_train
    }

    val = {
        'data': data_val,
        'targets': targets_val
    }

    # Save split data
    np.savez_compressed(os.path.join('./data', 'MNIST', 'train.npz'), **train)
    np.savez_compressed(os.path.join('./data', 'MNIST', 'val.npz'), **val)

    test_dataset = torchvision.datasets.MNIST('./data', train=False)

    # Save test data
    np.savez_compressed(os.path.join('./data', 'MNIST', 'test.npz'),
                        data=test_dataset.data, targets=test_dataset.targets)


class MNIST(data.Dataset):
    """
    A dataset wrapper for MNIST
    """

    def __init__(self, root: str, split: str = 'train',
                 transform: Optional[Callable] = None,
                 binarise: bool = False,
                 rng: torch.Generator = None):
        """
        Args:
            root:       root directory that contains all data
            split:      data split, e.g. train, val, test
            transforms: torchvision transforms to apply to the data
            binarise:   binarise the dataset
            rng:        random number generator used for noise
        """
        super().__init__()

        filename = os.path.join(root, 'MNIST', f'{split}.npz')
        data = np.load(filename)
        self.data = data['data']
        self.targets = data['targets']

        self.transform = transform
        self.binarise = binarise

        # Preprocess in advance, so that we can modify the data after
        self.preprocess(rng=rng)

    def preprocess(self, *, rng: torch.Generator):
        # Transform one sample to find out the shape of the transformed image
        temp = self.data[0]
        temp = Image.fromarray(temp, mode='L')
        if self.transform is not None:
            temp = self.transform(temp)
        temp = torchvision.transforms.ToTensor()(temp)
        temp = temp.flatten()

        # Transform all data
        original_data = self.data
        self.data = np.empty((len(original_data), *temp.shape), dtype=temp.numpy().dtype)
        for i, img in enumerate(original_data):
            img = Image.fromarray(img, mode='L')

            if self.transform is not None:
                img = self.transform(img)

            img = torchvision.transforms.ToTensor()(img).numpy()
            # We will use flatenned arrays
            img = img.flatten()

            self.data[i] = img

        if self.binarise:
            # Binarise to {-1, 1}
            self.data = (torch.rand(*self.data.shape, generator=rng).numpy() <= self.data).astype(self.data.dtype)
            self.data = self.data*2 - 1
        else:
            # Add noise and rescale to [0, 1]
            self.data = self.data*255
            self.data += torch.rand(*self.data.shape, generator=rng).numpy().astype(self.data.dtype)
            self.data /= 256

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __setitem__(self, key, value):
        """
        Args:
            key: index of sample
        """
        self.data[key] = value

    def __len__(self) -> int:
        return len(self.data)
