from typing import List, Optional, Tuple, Union
from enum import Enum, auto

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vgiwae.shared.vae_enums import DISTRIBUTION
from vgiwae.utils.estimate_total_train_steps import PLEstimateTotalTrainSteps

from vgiwae.models.iwae import estimate_iwelbo
from vgiwae.models.multiple_iwae import estimate_multiple_iwelbo
from vgiwae.models.vae import estimate_elbo
from vgiwae.models.multiple_vae import estimate_multiple_elbo

from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE
if _OMEGACONF_AVAILABLE:
    from omegaconf import Container

class RefitEncoderClass(Enum):
    standard = auto()
    stratified = auto()


class RefitEncoderUsingIWAE(pl.LightningModule, PLEstimateTotalTrainSteps):
    def __init__(self,
                 refit_encoder_class: RefitEncoderClass,
                 var_latent_STL: bool = False,
                 var_latent_DREG: bool = False,
                 num_latent_samples: int = 1,
                 num_importance_samples: int = 1,

                 num_test_latent_samples: int = 1,
                 num_test_importance_samples: int = 1,

                 lr_latent: float = 0.,
                 amsgrad_latent: bool = False,
                 use_lr_scheduler: bool = False,
                 max_scheduler_steps: int = -1,

                 add_mask_layer_to_existing_encoder: bool = False,

                 use_looser_siwelbo_bound: bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters()

    # Override
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint=checkpoint)
        if self.model.hparams:
            if hasattr(self.model, "_hparams_name"):
                checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_NAME] = self.model._hparams_name
            # dump arguments
            if _OMEGACONF_AVAILABLE and isinstance(self.model.hparams, Container):
                checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY] = self.model.hparams
                checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_TYPE] = type(self.model.hparams)
            else:
                checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY] = dict(self.model.hparams)

    # Override
    def state_dict(self):
        """Returns model state."""
        model = self.model
        state_dict = model.state_dict()
        # state_dict_new = {}
        # # Remove the 'model.' prefix from param names
        # for name, param in state_dict.items():
        #     state_dict_new[name.split('model.')[1]] = param
        return state_dict

    def set_model(self, model: pl.LightningModule):
        self.model = model

        if self.hparams.add_mask_layer_to_existing_encoder:
            self.model.var_latent_network.add_mask_inputs_to_existing_network()
            self.model.hparams.encoder_use_mis_mask = True

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def vae_forward(self, *args, **kwargs):
        return self.model.vae_forward(*args, **kwargs)

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        if self.hparams.refit_encoder_class is RefitEncoderClass.standard:
            elbo = estimate_iwelbo(self, batch,
                                var_latent_STL=self.hparams.var_latent_STL,
                                var_latent_DREG=self.hparams.var_latent_DREG,
                                num_latent_samples=self.hparams.num_latent_samples,
                                num_importance_samples=self.hparams.num_importance_samples)
        elif self.hparams.refit_encoder_class is RefitEncoderClass.stratified:
            # Re-compute the number of importance samples
            num_importance_samples = self.hparams.num_importance_samples
            num_importance_samples = num_importance_samples_for_stratified_mixtures(
                                                num_importance_samples,
                                                self.model.hparams.var_latent_distribution)

            elbo = estimate_multiple_iwelbo(self, batch,
                                var_latent_STL=self.hparams.var_latent_STL,
                                var_latent_DREG=self.hparams.var_latent_DREG,
                                num_latent_samples=self.hparams.num_latent_samples,
                                num_importance_samples_for_each_component=num_importance_samples,
                                use_looser_bound=self.hparams.use_looser_siwelbo_bound)
        else:
            raise NotImplementedError(f'Unknown refit encoder class: {self.refit_encoder_class}')
        # if self.hparams.refit_encoder_class is RefitEncoderClass.standard:
        #     elbo = estimate_elbo(self, batch,
        #                         var_latent_STL=self.hparams.var_latent_STL,
        #                         kl_analytic=False,
        #                         num_latent_samples=self.hparams.num_latent_samples,
        #                         )
        # elif self.hparams.refit_encoder_class is RefitEncoderClass.stratified:
        #     # Re-compute the number of importance samples
        #     num_latent_samples = self.hparams.num_latent_samples
        #     num_latent_samples = num_importance_samples_for_stratified_mixtures(
        #                                         num_latent_samples,
        #                                         self.model.hparams.var_latent_distribution)

        #     elbo = estimate_multiple_elbo(self, batch,
        #                         var_latent_STL=self.hparams.var_latent_STL,
        #                         kl_analytic=False,
        #                         num_latent_samples=num_latent_samples,
        #                         )
        # else:
        #     raise NotImplementedError(f'Unknown refit encoder class: {self.refit_encoder_class}')

        loss = -elbo

        # logs metrics
        self.log('elbo/train', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        with torch.inference_mode():
            if self.hparams.refit_encoder_class is RefitEncoderClass.standard:
                elbo = estimate_iwelbo(self, batch,
                                    var_latent_STL=self.hparams.var_latent_STL,
                                    var_latent_DREG=self.hparams.var_latent_DREG,
                                    num_latent_samples=self.hparams.num_latent_samples,
                                    num_importance_samples=self.hparams.num_importance_samples)
            elif self.hparams.refit_encoder_class is RefitEncoderClass.stratified:
                # Re-compute the number of importance samples
                num_importance_samples = self.hparams.num_importance_samples
                num_importance_samples = num_importance_samples_for_stratified_mixtures(
                                                    num_importance_samples,
                                                    self.model.hparams.var_latent_distribution)

                elbo = estimate_multiple_iwelbo(self, batch,
                                    var_latent_STL=self.hparams.var_latent_STL,
                                    var_latent_DREG=self.hparams.var_latent_DREG,
                                    num_latent_samples=self.hparams.num_latent_samples,
                                    num_importance_samples_for_each_component=num_importance_samples,
                                    use_looser_bound=self.hparams.use_looser_siwelbo_bound)
            else:
                raise NotImplementedError(f'Unknown refit encoder class: {self.refit_encoder_class}')
            # if self.hparams.refit_encoder_class is RefitEncoderClass.standard:
            #     elbo = estimate_elbo(self, batch,
            #                          var_latent_STL=self.hparams.var_latent_STL,
            #                          kl_analytic=False,
            #                          num_latent_samples=self.hparams.num_latent_samples,
            #                          )
            # elif self.hparams.refit_encoder_class is RefitEncoderClass.stratified:
            #     # Re-compute the number of importance samples
            #     num_latent_samples = self.hparams.num_latent_samples
            #     num_latent_samples = num_importance_samples_for_stratified_mixtures(
            #                                         num_latent_samples,
            #                                         self.model.hparams.var_latent_distribution)

            #     elbo = estimate_multiple_elbo(self, batch,
            #                         var_latent_STL=self.hparams.var_latent_STL,
            #                         kl_analytic=False,
            #                         num_latent_samples=num_latent_samples,
            #                         )
            # else:
            #     raise NotImplementedError(f'Unknown refit encoder class: {self.refit_encoder_class}')
            loss = -elbo

        # logs metrics
        self.log('elbo/val', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self,
                  batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                  batch_idx: int) -> Optional[STEP_OUTPUT]:
        with torch.inference_mode():
            if self.hparams.refit_encoder_class is RefitEncoderClass.standard:
                elbo = estimate_iwelbo(self, batch,
                                    var_latent_STL=self.hparams.var_latent_STL,
                                    var_latent_DREG=self.hparams.var_latent_DREG,
                                    num_latent_samples=self.hparams.num_test_latent_samples,
                                    num_importance_samples=self.hparams.num_test_importance_samples)
            elif self.hparams.refit_encoder_class is RefitEncoderClass.stratified:
                # Re-compute the number of importance samples
                # num_importance_samples = self.hparams.num_test_importance_samples
                # num_importance_samples = num_importance_samples_for_stratified_mixtures(
                #                                     num_importance_samples,
                #                                     self.model.hparams.var_latent_distribution)

                # elbo = estimate_multiple_iwelbo(self, batch,
                #                     var_latent_STL=self.hparams.var_latent_STL,
                #                     var_latent_DREG=self.hparams.var_latent_DREG,
                #                     num_latent_samples=self.hparams.num_test_latent_samples,
                #                     num_importance_samples_for_each_component=num_importance_samples,
                #                     use_looser_bound=self.hparams.use_looser_siwelbo_bound)
                elbo = estimate_iwelbo(self, batch,
                                    var_latent_STL=self.hparams.var_latent_STL,
                                    var_latent_DREG=self.hparams.var_latent_DREG,
                                    num_latent_samples=self.hparams.num_test_latent_samples,
                                    num_importance_samples=self.hparams.num_test_importance_samples,
                                    sample_stratifieddistr_without_stratification_norsample=True,
                                    )
            else:
                raise NotImplementedError(f'Unknown refit encoder class: {self.refit_encoder_class}')

        self.log('est_loglik/test', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.var_latent_network.parameters(),
             'amsgrad': self.hparams.amsgrad_latent,
             'lr': self.hparams.lr_latent}])

        opts = {
            'optimizer': optimizer
        }

        if self.hparams.use_lr_scheduler:
            max_steps = self.hparams.max_scheduler_steps if self.hparams.max_scheduler_steps != -1 else self.estimated_num_training_steps
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps, eta_min=0, last_epoch=-1)

            # NOTE: these configs are not available in MVB2 manual optimisation.
            # So if this changes, you should also check if it need to be changed in MVB2.
            opts['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
                'frequency': 1,
            }

        return opts

    def configure_gradient_clipping(self,
            optimizer: torch.optim.Optimizer,
            optimizer_idx: int,
            gradient_clip_val: Optional[Union[int, float]] = None,
            gradient_clip_algorithm: Optional[str] = None):
        # NOTE: We will only clip the gradients of the encoder network
        # Create a dummy optimiser object to work around pytorch-lightning
        # class DummyOpt(object):
        #     pass
        # dummy_opt = DummyOpt()
        # dummy_opt.param_groups = [{'params': self.var_latent_network.parameters()}]

        self.clip_gradients(optimizer, #dummy_opt,
                            gradient_clip_val=gradient_clip_val,
                            gradient_clip_algorithm=gradient_clip_algorithm)

def num_importance_samples_for_stratified_mixtures(num_importance_samples, model_var_latent_distribution):
    if model_var_latent_distribution in (DISTRIBUTION.stratified_mixture1_normal_with_eps, DISTRIBUTION.stratified_mixture1_normal):
        num_components = 1
        num_importance_samples = int(num_importance_samples / num_components)
    elif model_var_latent_distribution in (DISTRIBUTION.stratified_mixture5_normal_with_eps, DISTRIBUTION.stratified_mixture5_normal):
        num_components = 5
        num_importance_samples = int(num_importance_samples / num_components)
    elif model_var_latent_distribution in (DISTRIBUTION.stratified_mixture15_normal_with_eps, DISTRIBUTION.stratified_mixture15_normal):
        num_components = 15
        num_importance_samples = int(num_importance_samples / num_components)
    elif model_var_latent_distribution in (DISTRIBUTION.stratified_mixture25_normal_with_eps, DISTRIBUTION.stratified_mixture25_normal):
        num_components = 25
        num_importance_samples = int(num_importance_samples / num_components)
    else:
        raise NotImplementedError(f'Unknown {model_var_latent_distribution=}')
    return num_importance_samples
