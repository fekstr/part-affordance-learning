import comet_ml
import os
import json

from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from src.config import config
from src.models.baseline_object import BaselineObjectModel
from src.pl.pl_wrapper import PLWrapper
from src.models.baseline import BaselineModel
from src.models.baseline2 import BaselineModel2
from src.datasets.utils import get_dataloaders
from src.utils import set_seeds
from src.datasets.dataset import CommonDataset


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

    _, _, test_dataloader = get_dataloaders(dataset,
                                            small=args.dev,
                                            batch_size=8,
                                            load_objects=config['item_type']
                                            in ['object', 'all_part'])
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.device_count() == 1 else None,
        logger=[comet_logger])
    trainer.test(model,
                 dataloaders=[test_dataloader],
                 ckpt_path=args.checkpoint)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--no-logging', action='store_true', default=False)
    parser.add_argument('--checkpoint')

    dataset = CommonDataset(
        objects_path=config['data_path'],
        tag=config['tag'],
        num_points=config['num_points'],
        train_object_classes=config['train_object_classes'],
        test_object_classes=config['test_object_classes'],
        affordances=config['affordances'],
        item_type=config['item_type'],
        manual_labels=config['labels'],
        test=True)

    model = PLWrapper(BaselineObjectModel(num_classes=dataset.num_class),
                      dataset.index_affordance_map)

    args = parser.parse_args()

    load_dotenv()
    main(args, dataset, model)