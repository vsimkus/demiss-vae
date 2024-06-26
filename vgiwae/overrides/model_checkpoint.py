import os
from typing import Any, Optional, Union
import pytorch_lightning as pl

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ModelCheckpoint(ModelCheckpoint):
    """
    Adds another option to clear store custom epoch/step checkpoints.
    """
    def __init__(self, ckpt_epochs=[], ckpt_steps=[], del_old_chpts=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.del_old_chpts = del_old_chpts
        self.ckpt_epochs = ckpt_epochs
        self.ckpt_steps = ckpt_steps

        if del_old_chpts:
            dir_name = os.path.split(kwargs['filepath'])[0]
            for f in os.listdir(dir_name):
                self._del_model(os.path.join(dir_name, f))

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        out = super().on_validation_end(trainer, pl_module)

        epoch = trainer.current_epoch
        if self.ckpt_epochs is not None and epoch in self.ckpt_epochs:
            filepath = os.path.join(self.dirpath,
                                    f'custom_ckpt_by_epoch_{epoch}.ckpt')
            trainer.save_checkpoint(filepath, self.save_weights_only)

        return out

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[STEP_OUTPUT, None], batch: Any, batch_idx: int) -> None:
        out = super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        step = trainer.global_step
        if self.ckpt_steps is not None and step in self.ckpt_steps:
            filepath = os.path.join(self.dirpath,
                                    f'custom_ckpt_by_step_{step}.ckpt')
            trainer.save_checkpoint(filepath, self.save_weights_only)

        return out

    rank_zero_only
    def on_validation_batch_end(
        self, trainer: Trainer,
        pl_module: LightningModule,
        outputs: Union[STEP_OUTPUT, None],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        out = super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        step = trainer.global_step
        if self.ckpt_steps is not None and step in self.ckpt_steps:
            filepath = os.path.join(self.dirpath,
                                    f'custom_ckpt_by_step_{step}.ckpt')
            trainer.save_checkpoint(filepath, self.save_weights_only)

        return out
