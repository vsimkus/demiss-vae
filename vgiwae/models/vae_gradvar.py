from typing import Tuple, Type

from torch.optim import Optimizer

from vgiwae.shared.vae_grad_var import (gradvar_on_before_optimizer_step,
                                        gradvar_on_train_epoch_start,
                                        gradvar_training_epoch_end)

from .iwae import IWAE
from .mvbiwae import MVBIWAE
from .mvbvae import MVBVAE
from .vae import VAE
from .multiple_iwae import MultipleIWAE
from .multiple_vae import MultipleVAE


def create_gradvar_subclass(classname: str, parent_classes: Tuple[Type[object]]) -> Type[object]:
    """Estimates gradient variance during learning."""
    # def __init__(self, *args, **kwargs):
    #     super(type(self), self).__init__(*args, **kwargs)

    def on_train_epoch_start(self):
        out = super(type(self), self).on_train_epoch_start()
        gradvar_on_train_epoch_start(self)
        return out

    def on_train_epoch_end(self) -> None:
        out = super(type(self), self).on_train_epoch_end()
        gradvar_training_epoch_end(self)
        return out

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        out = super(type(self), self).on_before_optimizer_step(optimizer, optimizer_idx)
        gradvar_on_before_optimizer_step(self, optimizer, optimizer_idx)
        return out

    overrides = {
        # '__init__': __init__,
        'on_train_epoch_start': on_train_epoch_start,
        'on_train_epoch_end': on_train_epoch_end,
        'on_before_optimizer_step': on_before_optimizer_step
    }

    subclass = type(classname, parent_classes, overrides)

    # Let the docstring of the new class be the one from the parent class + description of subclass
    subclass.__doc__ = parent_classes[0].__doc__ + '\n\nEstimates gradient variance during learning.'

    return subclass

VAEGradVar = create_gradvar_subclass('VAEGradVar', (VAE,))
IWAEGradVar = create_gradvar_subclass('IWAEGradVar', (IWAE,))
MVBVAEGradVar = create_gradvar_subclass('MVBVAEGradVar', (MVBVAE,))
MVBIWAEGradVar = create_gradvar_subclass('MVBIWAEGradVar', (MVBIWAE,))

MultipleVAEGradVar = create_gradvar_subclass('MultipleVAEGradVar', (MultipleVAE,))
MultipleIWAEGradVar = create_gradvar_subclass('MultipleIWAEGradVar', (MultipleIWAE,))
