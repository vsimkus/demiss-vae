import math

from pytorch_lightning.utilities import rank_zero_warn

# Adapted backport from https://github.com/PyTorchLightning/pytorch-lightning/pull/11599

class PLEstimateTotalTrainSteps:
    @property
    def estimated_num_training_steps(self) -> int:
        """Total training steps inferred from dataloaders and distributed setup."""
        # infinite training
        if self.trainer.max_epochs == -1 and self.trainer.max_steps == -1:
            return float("inf")

        # max steps training
        if (self.trainer.max_epochs is None or self.trainer.max_epochs == -1) and self.trainer.max_steps is not None:
            return self.trainer.max_steps

        if self.train_dataloader is None or self.trainer.num_training_batches == float('inf'):
            rank_zero_warn("Loading `train_dataloader` to estimate number of training steps.")
            self.trainer.reset_train_dataloader()

        total_batches = self.trainer.num_training_batches

        # iterable dataset
        if total_batches == float("inf"):
            return self.trainer.max_steps

        self.accumulate_grad_batches = self.trainer.accumulation_scheduler.get_accumulate_grad_batches(self.current_epoch)
        effective_batch_size = self.accumulate_grad_batches
        max_estimated_steps = math.ceil(total_batches / effective_batch_size) * (
            self.trainer.max_epochs if self.trainer.max_epochs != -1 else 1
        )

        max_estimated_steps = min(max_estimated_steps, self.trainer.max_steps) if self.trainer.max_steps != -1 else max_estimated_steps
        return max_estimated_steps
