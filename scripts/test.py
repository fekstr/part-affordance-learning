import comet_ml
import os
import json
from types import SimpleNamespace

from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from src.pl.pl_wrapper import PLWrapper
from src.models.pointnet_joint import PointNetJointModel, PointNetJointLoss
from src.utils import set_seeds
from src.datasets.dataset import get_datasets
from src.datasets.utils import get_dataloader, load_id_split


def load_options(checkpoints_path):
    with open(os.path.join(checkpoints_path, 'config.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join(checkpoints_path, 'hyperparams.json'), 'r') as f:
        hyperparams = json.load(f)
    return config, SimpleNamespace(**hyperparams)


def init_experiment(resume_id, disable_logging=False):
    experiment = comet_ml.ExistingExperiment(
        api_key=os.environ['COMET_API_KEY'],
        previous_experiment=resume_id,
        disabled=disable_logging)
    logger = CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                         rest_api_key=os.environ.get('COMET_API_KEY'),
                         save_dir='./comet_logs',
                         experiment_key=experiment.get_key(),
                         offline=disable_logging,
                         project_name="object-affordances")
    return experiment, logger


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--no-logging', action='store_true', default=False)
    parser.add_argument('--test-on-train', action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()

    # Prepare environment
    set_seeds(1)
    torch.cuda.empty_cache()
    torch.set_num_threads(1)
    load_dotenv()

    # Load config from training
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config, hyperparams = load_options(checkpoint_dir)

    # Define model
    model = PointNetJointModel(num_classes=len(config['affordances']),
                               num_slots=7)
    loss = PointNetJointLoss()

    # Load data
    id_split = load_id_split(config['data_path'],
                             config['train_object_classes'],
                             config['test_object_classes'],
                             test=True)
    train_dataset, _, test_dataset = get_datasets(config, hyperparams,
                                                  id_split)
    train_dataloader = get_dataloader(train_dataset,
                                      small=args.dev,
                                      batch_size=hyperparams.batch_size,
                                      weighted_sampling=True)
    test_dataloader = get_dataloader(test_dataset,
                                     small=args.dev,
                                     batch_size=hyperparams.batch_size)

    # Create lightning module
    pl_model = PLWrapper(
        model=model,
        loss=loss,
        index_affordance_map=test_dataset.index_affordance_map)

    experiment_id = os.path.dirname(args.checkpoint).split('/')[-1]

    experiment, comet_logger = init_experiment(disable_logging=args.no_logging,
                                               resume_id=experiment_id)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.device_count() == 1 else None,
        logger=[comet_logger])

    trainer.test(pl_model,
                 dataloaders=[train_dataloader]
                 if args.test_on_train else [test_dataloader],
                 ckpt_path=args.checkpoint)
