from typing import List, Tuple, Type, Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .vae import VAE
from .iwae import IWAE
from .mvbvae import MVBVAE
from .mvbiwae import MVBIWAE

from .multiple_iwae import MultipleIWAE
from .multiple_vae import MultipleVAE


from vgiwae.shared.fid import compute_inception_gaussian, gaussian_fid_score


def create_fid_subclass(classname: str, parent_classes: Tuple[Type[object]]) -> Type[object]:
    """Computes FID score using an inception model."""
    def __init__(self, *args,
                 fid_eval_step_freq:int = 1,
                 inception_model_type: str = None,
                 inception_model_path: str = None,
                 reference_samples: List[str] = None,
                 num_samples_per_batch: int = None,
                 **kwargs):
        super(type(self), self).__init__(*args, **kwargs)
        # With this dynamic subclassing save_hyperparameters does not automatically save the arguments of the subclass, so save them manually
        self.hparams.fid_eval_step_freq = fid_eval_step_freq
        self.hparams.num_samples_per_batch = num_samples_per_batch

        if inception_model_type is not None:
            if inception_model_type  == 'VAE_encoder':
                inception_model = VAE.load_from_checkpoint(checkpoint_path=inception_model_path)
                inception_model = inception_model.to('cpu')
                inception_model.freeze()
                self.inception_model = inception_model.var_latent_network
                # NOTE: make sure this is not stored in the checkpoints!
                # This is done below in `on_save_checkpoint`
            else:
                raise NotImplementedError(f'Unknown inception model type: {inception_model_type}')

        self.hparams.reference_samples = reference_samples

    def on_train_start(self):
        out = super(type(self), self).on_train_start()

        # Get reference samples
        self.fid_ref_dataset_sizes = []
        self.fid_ref_gaussians = []
        datamodule = self.trainer.datamodule
        for ref in self.hparams.reference_samples:
            if ref == 'dataset_test':
                if not datamodule.has_setup_test:
                    print('Setting up test dataloader for FID evaluation.')
                    datamodule.setup(stage='test')
                refs_dataset = datamodule.test_data_core[:]
            elif ref == 'dataset_val':
                if not datamodule.has_setup_validate:
                    print('Setting up validation dataloader for FID evaluation.')
                    datamodule.setup(stage='fit')
                refs_dataset = datamodule.val_data_core[:]
            elif ref == 'dataset_train':
                if not datamodule.has_setup_fit:
                    print('Setting up train dataloader for FID evaluation.')
                    datamodule.setup(stage='fit')
                refs_dataset = datamodule.train_data_core[:]
            else:
                raise NotImplementedError(f'Unknown {ref=}')

            if isinstance(refs_dataset, tuple):
                refs_dataset = refs_dataset[0]

            refs_dataset = torch.tensor(refs_dataset).to(self.device)
            self.fid_ref_dataset_sizes.append(refs_dataset.shape[0])

            # Compute the features of the reference samples and their Gaussian distribution
            with torch.inference_mode():
                if self.hparams.num_samples_per_batch is None:
                    num_samples_per_batch = 5000
                else:
                    num_samples_per_batch = self.hparams.num_samples_per_batch
                total = 0
                fid_ref_gaussian_mean, fid_ref_gaussian_cov = \
                    compute_inception_gaussian(refs_dataset, inception_model=self.inception_model,
                                               batch_size=min(num_samples_per_batch,5000))
                fid_ref_gaussian_mean = fid_ref_gaussian_mean.cpu()
                fid_ref_gaussian_cov = fid_ref_gaussian_cov.cpu()

                self.fid_ref_gaussians.append((fid_ref_gaussian_mean, fid_ref_gaussian_cov))

        return out

    def on_save_checkpoint(self, checkpoint):
        super(type(self), self).on_save_checkpoint(checkpoint=checkpoint)
        # NOTE: Don't store the inception model in the checkpoints
        for state_key in list(checkpoint['state_dict'].keys()):
            if state_key.startswith('inception_model'):
                checkpoint['state_dict'].pop(state_key)

        return checkpoint

    def eval_fid_score(self, ref_id):
        num_samples = self.fid_ref_dataset_sizes[ref_id]
        # Limit the number of samples
        num_samples = min(num_samples, 20000)

        with torch.inference_mode():
            # Generate samples from the model
            samples = []
            if self.hparams.num_samples_per_batch is None:
                num_samples_per_batch = num_samples
            else:
                num_samples_per_batch = self.hparams.num_samples_per_batch
            total = 0
            for b in range(0, num_samples, num_samples_per_batch):
                num_samples_b = min(num_samples_per_batch, num_samples - total)
                total += num_samples_b
                samples_b = self.sample_vae(num_samples_b)
                samples.append(samples_b)
            samples = torch.cat(samples, dim=0)

            # Compute the features of the samples and their Gaussian distribution
            samples_feat_mean, samples_feat_cov = \
                compute_inception_gaussian(samples, inception_model=self.inception_model, batch_size=min(num_samples_per_batch,5000))
            samples_feat_mean = samples_feat_mean.cpu()
            samples_feat_cov = samples_feat_cov.cpu()

            # Compute the FID score
            fid_score = gaussian_fid_score(self.fid_ref_gaussians[ref_id][0], self.fid_ref_gaussians[ref_id][1],
                                           samples_feat_mean, samples_feat_cov)

        return fid_score

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        out = super(type(self), self).on_train_batch_end(outputs, batch, batch_idx)

        step = self.trainer.global_step
        if step % self.hparams.fid_eval_step_freq == 0 or self.trainer.global_step == self.trainer.max_steps:
            # Eval FID and log
            for ref_id, ref in enumerate(self.hparams.reference_samples):
                fid_score = self.eval_fid_score(ref_id)
                # self.log('fid_score/test', fid_score, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                # Log directly in tensorboard to avoid excessive loggin during steps when we are not computing the FID
                tensorboard = self.logger.experiment
                dataset = None
                if ref == 'dataset_test':
                    dataset = 'test'
                elif ref == 'dataset_train':
                    dataset = 'train'
                elif ref == 'dataset_val':
                    dataset = 'val'
                else:
                    raise NotImplementedError(f'Unknown {self.hparams.reference_samples=}')
                tensorboard.add_scalar(f'fid_score/{dataset}', fid_score, global_step=step)

        return out

    overrides = {
        '__init__': __init__,
        'on_save_checkpoint': on_save_checkpoint,
        'on_train_start': on_train_start,
        'on_train_batch_end': on_train_batch_end,
        'eval_fid_score': eval_fid_score,
    }

    subclass = type(classname, parent_classes, overrides)

    # Let the docstring of the new class be the one from the parent class + description of subclass
    subclass.__doc__ = parent_classes[0].__doc__ + '\n\nComputes FID score using an inception model.'

    return subclass

VAE_FID = create_fid_subclass('VAE_FID', (VAE,))
IWAE_FID = create_fid_subclass('IWAE_FID', (IWAE,))
MVBVAE_FID = create_fid_subclass('MVBVAE_FID', (MVBVAE,))
MVBIWAE_FID = create_fid_subclass('MVBIWAE_FID', (MVBIWAE,))

MultipleIWAE_FID = create_fid_subclass('MultipleIWAE_FID', (MultipleIWAE,))
MultipleVAE_FID = create_fid_subclass('MultipleVAE_FID', (MultipleVAE,))
