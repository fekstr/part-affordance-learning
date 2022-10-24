import comet_ml
from types import SimpleNamespace
import os
import json

from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.config import hyperparams_dict, config
from src.datasets.dataset import CommonDataset
from src.models.attention import AttentionModel
from src.pl.pl_wrapper import PLWrapper
from src.models.weak import WeakModel
from src.models.baseline import BaselineModel
from src.models.baseline2 import BaselineModel2
from src.models.baseline_object import BaselineObjectModel
from src.utils import set_seeds
from src.datasets.utils import get_dataloaders


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


def main(args, dataset, model, hyperparams, config):
    set_seeds(1)
    torch.set_num_threads(1)

    if args.resume_id:
        ckpt_path = os.path.join('checkpoints', args.resume_id, 'last.ckpt')
    else:
        ckpt_path = None

    dev_tags = ['dev'] if args.dev else []
    experiment, comet_logger = init_experiment(
        tags=[dataset.tag] + dev_tags,
        resume_id=args.resume_id,
        disable_logging=args.no_logging,
    )

    options = {**hyperparams, **config}
    comet_logger.log_hyperparams(options)
    hyperparams = SimpleNamespace(**hyperparams)

    train_dataloader, valid_dataloader, _ = get_dataloaders(
        dataset,
        small=args.dev,
        batch_size=hyperparams.batch_size,
        load_objects=config['item_type'] in ['object', 'all_part'])

    checkpoint_cb = ModelCheckpoint(dirpath=os.path.join(
        'checkpoints', experiment.get_key()),
                                    save_top_k=3,
                                    monitor='valid_loss',
                                    save_last=True)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.device_count() == 1 else None,
        logger=[comet_logger],
        callbacks=None if args.no_checkpoints else [checkpoint_cb],
        log_every_n_steps=1 if args.dev else 10,
        max_epochs=2 if args.dev else 50)

    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-id')
    parser.add_argument('--no-logging', action='store_true', default=False)
    parser.add_argument('--no-checkpoints', action='store_true', default=False)
    parser.add_argument('--dev', action='store_true', default=False)
    args = parser.parse_args()

    hyperparams = SimpleNamespace(**hyperparams_dict)

    dataset = CommonDataset(
        objects_path=config['data_path'],
        tag=config['tag'],
        num_points=config['num_points'],
        train_object_classes=config['train_object_classes'],
        test_object_classes=config['test_object_classes'],
        affordances=config['affordances'],
        manual_labels=config['labels'],
        item_type=config['item_type'],
        # force_new_split=True,
    )
    model = PLWrapper(AttentionModel(num_classes=dataset.num_class,
                                     affordances=['sit']),
                      hyperparams=hyperparams)

    torch.cuda.empty_cache()
    load_dotenv()
    main(args, dataset, model, hyperparams_dict, config)
