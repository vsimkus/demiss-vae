import torch
from einops import rearrange, reduce


def gradvar_on_train_epoch_start(self):
    self.generator_epoch_grads = []
    self.var_latent_epoch_grads = []

def gradvar_on_before_optimizer_step(self, optimizer, optimizer_idx):
    generator_grads = []
    for _, v in self.generator_network.named_parameters():
        generator_grads.append(v.grad.detach().clone().flatten())
    var_latent_grads = []
    for _, v in self.var_latent_network.named_parameters():
        var_latent_grads.append(v.grad.detach().clone().flatten())

    self.generator_epoch_grads.append(torch.concat(generator_grads))
    self.var_latent_epoch_grads.append(torch.concat(var_latent_grads))

def gradvar_training_epoch_end(self):
    generator_epoch_grads = rearrange(self.generator_epoch_grads, 'b g -> b g')
    generator_grad_std = torch.std(generator_epoch_grads, dim=0, unbiased=True)
    generator_grad_var = torch.sum(generator_grad_std**2)
    self.log('generator_grad_var/train', generator_grad_var, on_epoch=True, prog_bar=False, logger=True)

    var_latent_epoch_grads = rearrange(self.var_latent_epoch_grads, 'b g -> b g')
    encoder_grad_std = torch.std(var_latent_epoch_grads, dim=0, unbiased=True)
    encoder_grad_var = torch.sum(encoder_grad_std**2)
    self.log('encoder_grad_var/train', encoder_grad_var, on_epoch=True, prog_bar=False, logger=True)

    # Also record signal-to-noise as in Rainforth et al 2018 (Tighter bounds are not necessarily better)
    # NOTE: some gradients are always 0 (i.e. dead units), so remove them from computation
    generator_grad_mean = reduce(generator_epoch_grads, 'b g -> g', 'mean')
    generator_grad_snr = torch.abs(generator_grad_mean / generator_grad_std)
    generator_grad_snr = torch.sum(generator_grad_snr[torch.isfinite(generator_grad_snr)])
    self.log('generator_grad_snr/train', generator_grad_snr, on_epoch=True, prog_bar=False, logger=True)

    encoder_grad_mean = reduce(var_latent_epoch_grads, 'b g -> g', 'mean')
    encoder_grad_snr = torch.abs(encoder_grad_mean / encoder_grad_std)
    encoder_grad_snr = torch.sum(encoder_grad_snr[torch.isfinite(encoder_grad_snr)])
    self.log('encoder_grad_snr/train', encoder_grad_snr, on_epoch=True, prog_bar=False, logger=True)
