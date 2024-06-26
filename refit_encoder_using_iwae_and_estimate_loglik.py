import os.path
import importlib

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningArgumentParser
from pytorch_lightning.utilities.seed import seed_everything, reset_seed

from vgiwae.data import MissingDataModule
from vgiwae.utils.arg_utils import construct_experiment_subdir
from vgiwae.overrides.model_checkpoint import ModelCheckpoint
from vgiwae.test.refit_encoder_using_iwae_and_estimate_loglik import RefitEncoderUsingIWAE


def build_argparser():
    parser = LightningArgumentParser('VGIWAE refit-encoder experiment runner',
                                     parse_as_dict=False)

    # Add general arguments
    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)
    parser.add_argument('--experiment_subdir_base', type=str, required=True,
        help='Experiment subdirectory.')
    parser.add_argument('--add_checkpoint_callback', type=bool, default=False,
                        help='Adds additional checkpointing callback.')

    parser.add_argument('--run_test', action='store_true', default=True,
                        help='Run the test set evaluation after training.')

    parser.add_argument('--model_class', type=str, required=True,
                        help='The model class to use for the experiment.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='The path to the pre-trained model.')
    parser.add_argument('--load_best_model', action='store_true',
                        help='Whether to load the best model from the checkpoint.')

    # Add class arguments
    parser.add_lightning_class_args(MissingDataModule, 'data')
    # Note use `python train.py --model=vgiwae.models.VAE --print_config`
    # to print a config for a specific model class
    parser.add_lightning_class_args(RefitEncoderUsingIWAE, 'refit_model')
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
    refit_model = hparams.refit_model
    refit_model.set_datamodule(datamodule)

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
                                            #   ckpt_epochs=hparams.save_custom_epochs,
                                            #   ckpt_steps=hparams.save_custom_steps,)
                                            )
        if trainer_args['callbacks'] is not None:
            trainer_args['callbacks'].append(checkpoint_callback)
        else:
            trainer_args['callbacks'] = [checkpoint_callback]
    # # Always save the last model
    # checkpoint_callback = ModelCheckpoint(save_top_k=0,
    #                                       save_last=True,
    #                                       save_weights_only=False,)
    # if trainer_args['callbacks'] is not None:
    #     trainer_args['callbacks'].append(checkpoint_callback)
    # else:
    #     trainer_args['callbacks'] = [checkpoint_callback]

    trainer = pl.Trainer(**trainer_args)

    # Load the pre-trained model
    module = importlib.import_module('vgiwae.models')
    my_class = getattr(module, hparams.model_class)

    # Find model path (choose latest version)
    model_path = hparams.model_path
    path_seed_part = experiment_dir.split('seed_')[1].split('/')[0]
    model_path = model_path.format(path_seed_part, '{}')

    # Gather latest stats
    versions = os.listdir(model_path.split('version_')[0])
    versions = sorted([int(v.split('version_')[1]) for v in versions], reverse=True)
    if len(versions) > 1:
        print(f'Multiple versions in {model_path}')
    version = versions[0]
    model_path = model_path.format(version)
    if hparams.load_best_model:
        print('Finding best model.')
        model_path = model_path.split('last.ckpt')[0]
        models = os.listdir(model_path)
        models = [m for m in models if not m.startswith('last.ckpt')]
        assert len(models) == 1, f'Found multiple models in {model_path}'
        model_path = os.path.join(model_path, models[0])
    print('Using pre-trained model from:', model_path)

    # Load the model
    pretrained_model = my_class.load_from_checkpoint(model_path)
    refit_model.set_model(pretrained_model)

    # The instantiation steps might be different for different models.
    # Hence we reset the seed before training such that the seed at the start of lightning setup is the same.
    reset_seed()

    # Begin fitting
    trainer.fit(model=refit_model, datamodule=datamodule)

    print('Finished fine-tuning encoder.')
    if hparams.run_test:
        print('Estimating test-log-likelihood.')
        trainer.test(model=refit_model, datamodule=datamodule)


if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
