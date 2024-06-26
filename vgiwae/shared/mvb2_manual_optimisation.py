def mvb2_manual_optimisation_step(self, enc_loss, gen_loss):
    """Implements the custom optimisation for the double mvb objective"""
    opt = self.optimizers(use_pl_optimizer=True)
    opt.zero_grad()
    # Make sure that the gradients are taken wrt the generator network parameters
    self.manual_backward(gen_loss, retain_graph=True, inputs=list(self.generator_network.parameters()))
    # Make sure that the gradients are taken wrt the encoder network parameters
    self.manual_backward(enc_loss, inputs=list(self.var_latent_network.parameters()))

    # Clip gradients (if specified)
    self.configure_gradient_clipping(opt, -1,
                                     gradient_clip_val=self.hparams.mvb2_optim_gradclip_val,
                                     gradient_clip_algorithm=self.hparams.mvb2_optim_gradclip_alg)

    # Perform optimiser step
    opt.step()

    if self.hparams.use_lr_scheduler:
        # NOTE: this does not see the setting set in configure_optimizers.
        # So make sure it is the same as there to ensure consistency accross different methods.
        sch = self.lr_schedulers()
        sch.step()
