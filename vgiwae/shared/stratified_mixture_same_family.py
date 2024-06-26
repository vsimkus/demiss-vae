import torch
from torch.distributions.mixture_same_family import MixtureSameFamily

class StratifiedMixtureSameFamily(MixtureSameFamily):
    """
    A mixture (same-family) distribution with stratified sampling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sample_stratified = True

    def set_sample_not_stratified(self):
        self.sample_stratified = False

    def rsample(self, sample_shape=torch.Size()):
        if self.sample_stratified:
            comp = self.component_distribution
            Z = comp.rsample(sample_shape=sample_shape)
            return Z
        else:
            return super().rsample(sample_shape=sample_shape)

    def sample(self, sample_shape=torch.Size()):
        if self.sample_stratified:
            comp = self.component_distribution
            Z = comp.sample(sample_shape=sample_shape)
            return Z
        else:
            return super().sample(sample_shape=sample_shape)

    def sample_not_stratified(self, sample_shape=torch.Size()):
        return super().sample(sample_shape=sample_shape)
