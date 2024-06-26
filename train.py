import os.path

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningArgumentParser
from pytorch_lightning.utilities.seed import seed_everything, reset_seed

from vgiwae.data import MissingDataModule
from vgiwae.utils.arg_utils import construct_experiment_subdir
from vgiwae.overrides.model_checkpoint import ModelCheckpoint


def build_argparser():
    parser = LightningArgumentParser('VGIWAE training experiment runner',
                                     parse_as_dict=False)

    # Add general arguments
    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)
    parser.add_argument('--experiment_subdir_base', type=str, required=True,
        help='Experiment subdirectory.')
    parser.add_argument('--add_checkpoint_callback', type=bool, default=False,
                        help='Adds additional checkpointing callback.')
    parser.add_argument('--save_custom_epochs',
                        type=int, nargs='*',
                        help='Epochs when to save a model')
    parser.add_argument('--save_custom_steps',
                        type=int, nargs='*',
                        help='Steps when to save a model')

    # Add class arguments
    parser.add_lightning_class_args(MissingDataModule, 'data')
    # Note use `python train.py --model=vgiwae.models.VAE --print_config`
    # to print a config for a specific model class
    parser.add_lightning_class_args(pl.LightningModule, 'model', subclass_mode=True)
    parser.add_lightning_class_args(pl.Trainer, 'trainer')

    return parser

def run(hparams):
    # Set random seed
    # NOTE: this must be done before any class initialisation,
    # hence also before the call to parser.instantiate_classes()
    seed_everything(hparams.seed_everything, workers=True)

    # Construct the experiment directory
    experiment_subdir = construct_experiment_subdir(hparams)
    if hparams.trainer.default_root_dir is None:
        experiment_dir = f'./{experiment_subdir}'
    else:
        experiment_dir = f'{hparams.trainer.default_root_dir}/{experiment_subdir}'
    print('Experiment directory:', experiment_dir)

    # Instantiate dynamic object classes
    hparams = parser.instantiate_classes(hparams)

    # Get the instantiated data
    datamodule = hparams.data

    # Get the instantiated model
    model = hparams.model
    model.set_datamodule(datamodule)

    # Instantiate the trainer
    trainer_args = { **(hparams.trainer.as_dict()), "default_root_dir": experiment_dir }
    checkpoint_callback = None
    if hparams.add_checkpoint_callback:
        # Always save best performing model by validation loss and last model
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              save_last=True,
                                              mode="min",
                                              save_weights_only=False,
                                              monitor="loss/val",
                                              # Save at custom epochs and steps
                                              ckpt_epochs=hparams.save_custom_epochs,
                                              ckpt_steps=hparams.save_custom_steps,)
        if trainer_args['callbacks'] is not None:
            trainer_args['callbacks'].append(checkpoint_callback)
        else:
            trainer_args['callbacks'] = [checkpoint_callback]

    trainer = pl.Trainer(**trainer_args)

    # The instantiation steps might be different for different models.
    # Hence we reset the seed before training such that the seed at the start of lightning setup is the same.
    reset_seed()

    # Begin fitting
    trainer.fit(model=model, datamodule=datamodule)

    # Store path to best model checkpoint
    if checkpoint_callback is not None:
        print('Best checkpoint path:', checkpoint_callback.best_model_path)
        with open(os.path.join(experiment_dir, 'best_val_loss_model_path.txt'), 'w') as f:
            f.write(checkpoint_callback.best_model_path)

if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
