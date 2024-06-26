import torch
from torch.distributions.bernoulli import Bernoulli

class Bernoulli_m1_p1(Bernoulli):
    """
    Bernoulli distribution with support {-1, 1} instead of {0, 1}.
    """
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def variance(self):
        raise NotImplementedError()

    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        samples = samples*2 - 1
        return samples

    def log_prob(self, value):
        value = (value+1)/2
        return super().log_prob(value)
