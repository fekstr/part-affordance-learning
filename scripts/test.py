import comet_ml
import os
import json

from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from src.models.baseline_object import BaselineObjectModel
from src.pl.pl_wrapper import PLWrapper
from src.models.baseline import BaselineModel
from src.models.baseline2 import BaselineModel2
from src.models.attention import AttentionModel
from src.models.attention_joint import JointAttentionModel, JointAttentionModelLoss
from src.models.attention_slot import JointSlotAttentionModel, JointSlotAttentionModelLoss
from src.datasets.utils import get_dataloaders
from src.utils import set_seeds
from src.datasets.dataset import CommonDataset


def load_config(checkpoints_path):
    with open(os.path.join(checkpoints_path, 'config.json'), 'r') as f:
        config = json.load(f)
    return config


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


def main(args, dataset, model):
    set_seeds(1)
    torch.set_num_threads(1)

    experiment_id = os.path.dirname(args.checkpoint).split('/')[-1]

    experiment, comet_logger = init_experiment(disable_logging=args.no_logging,
                                               resume_id=experiment_id)

    train_dataloader, _, test_dataloader = get_dataloaders(
        dataset,
        small=args.dev,
        batch_size=2 if args.dev else 8,
        load_objects=config['item_type'] in ['object', 'all_part'])
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.device_count() == 1 else None,
        logger=[comet_logger])
    trainer.test(model,
                 dataloaders=[train_dataloader]
                 if args.test_on_train else [test_dataloader],
                 ckpt_path=args.checkpoint)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--no-logging', action='store_true', default=False)
    parser.add_argument('--test-on-train', action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    config = load_config(checkpoint_dir)

    dataset = CommonDataset(
        objects_path=config['data_path'],
        tag=config['tag'],
        num_points=config['num_points'],
        train_object_classes=config['train_object_classes'],
        test_object_classes=config['test_object_classes'],
        affordances=config['affordances'],
        item_type=config['item_type'],
        manual_labels=config['labels'],
        test=True,
        use_cached_metas=True,
        num_slots=5)

    model = PLWrapper(model=JointSlotAttentionModel(
        num_classes=dataset.num_class,
        affordances=dataset.affordances,
        num_points=config['num_points'],
        num_slots=5),
                      loss=JointSlotAttentionModelLoss(),
                      index_affordance_map=dataset.index_affordance_map)

    load_dotenv()
    main(args, dataset, model)