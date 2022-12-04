import comet_ml
from types import SimpleNamespace
import os
import json

from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from src.config import hyperparams_dict, config
from src.datasets.dataset import CommonDataset
from src.models.pointnet_joint import PointNetJointModel, PointNetJointLoss
from src.pl.pl_wrapper import PLWrapper
from src.utils import set_seeds
from src.datasets.utils import get_dataloader, load_id_split


def dump_config(checkpoints_path, config):
    config_save_path = os.path.join(checkpoints_path, 'config.json')
    os.makedirs(checkpoints_path, exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)


def get_datasets(config, id_split):
    common_dataset_args = {
        'objects_path': config['data_path'],
        'num_points': config['num_points'],
        'affordances': config['affordances'],
        'object_affordance_labels': config['labels'],
        'num_slots': 7
    }

    train_dataset = CommonDataset(
        **common_dataset_args,
        object_ids=id_split['train'],
    )
    valid_dataset = CommonDataset(
        **common_dataset_args,
        object_ids=id_split['valid'],
    )
    test_dataset = CommonDataset(
        **common_dataset_args,
        object_ids=id_split['test'],
    )

    return train_dataset, valid_dataset, test_dataset


def init_experiment(tags, resume_id, disable_logging=False):
    if resume_id:
        experiment = comet_ml.ExistingExperiment(
            api_key=os.environ['COMET_API_KEY'],
            previous_experiment=resume_id,
            disabled=disable_logging)
    else:
        experiment = comet_ml.Experiment(api_key=os.environ['COMET_API_KEY'],
                                         project_name='object-affordances',
                                         disabled=disable_logging)
        for tag in tags:
            experiment.add_tag(tag)
    logger = CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                         rest_api_key=os.environ.get('COMET_API_KEY'),
                         save_dir='./comet_logs',
                         experiment_key=experiment.get_key(),
                         project_name="object-affordances")
    return experiment, logger


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-id')
    parser.add_argument('--no-logging', action='store_true', default=False)
    parser.add_argument('--no-checkpoints', action='store_true', default=False)
    parser.add_argument('--dev', action='store_true', default=False)
    args = parser.parse_args()
    hyperparams = SimpleNamespace(**hyperparams_dict)

    # Prepare environment
    set_seeds(1)
    torch.cuda.empty_cache()
    torch.set_num_threads(1)
    load_dotenv()

    # Define model
    model = PointNetJointModel(num_classes=len(config['affordances']),
                               num_slots=7)
    loss = PointNetJointLoss()

    # Load data
    id_split = load_id_split(config['data_path'],
                             config['train_object_classes'],
                             config['test_object_classes'])
    train_dataset, valid_dataset, _ = get_datasets(config, id_split)
    train_dataloader = get_dataloader(
        train_dataset,
        small=args.dev,
        batch_size=hyperparams.batch_size,
        weighted_sampling=hyperparams.weighted_sampling)
    valid_dataloader = get_dataloader(
        valid_dataset,
        small=args.dev,
        batch_size=hyperparams.batch_size,
        weighted_sampling=hyperparams.weighted_sampling)

    # Create lightning module
    pl_model = PLWrapper(
        model=model,
        loss=loss,
        hyperparams=hyperparams,
        index_affordance_map=train_dataset.index_affordance_map,
        dev=args.dev)

    # Initialize Comet experiment or continue logging to an existing experiment
    dev_tags = ['dev'] if args.dev else []
    experiment, comet_logger = init_experiment(
        tags=config['tags'] + dev_tags,
        resume_id=args.resume_id,
        disable_logging=args.no_logging,
    )

    # Save experiment config and hyperparameters
    options = {**hyperparams_dict, **config}
    comet_logger.log_hyperparams(options)
    checkpoints_path = os.path.join('data', 'checkpoints',
                                    experiment.get_key())
    if not args.no_checkpoints and not args.resume_id and not args.dev:
        dump_config(checkpoints_path, config)

    # Configure trainer and checkpointing
    checkpoint_cb = ModelCheckpoint(dirpath=checkpoints_path,
                                    save_top_k=3,
                                    monitor='val_loss',
                                    mode='max',
                                    save_last=True)
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.device_count() == 1 else None,
        logger=[comet_logger],
        callbacks=None if args.no_checkpoints else [checkpoint_cb],
        log_every_n_steps=1 if args.dev else 10,
        max_epochs=2 if args.dev else 50,
    )

    # Continue from checkpoint if specified
    ckpt_path = os.path.join('data', 'checkpoints', args.resume_id,
                             'last.ckpt') if args.resume_id else None

    # Start the training
    trainer.fit(pl_model,
                train_dataloader,
                valid_dataloader,
                ckpt_path=ckpt_path)
