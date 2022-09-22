import comet_ml
import os

from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from src.pl.pl_wrapper import PLWrapper
from src.models.baseline import BaselineModel
from src.utils import get_dataloader, set_seeds


def init_experiment(resume_id, disable_logging=False):
    experiment = comet_ml.ExistingExperiment(
        api_key=os.environ['COMET_API_KEY'],
        previous_experiment=resume_id,
        disabled=disable_logging)
    logger = CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                         rest_api_key=os.environ.get('COMET_API_KEY'),
                         save_dir='./comet_logs',
                         experiment_key=experiment.get_key(),
                         project_name="object-affordances")
    return experiment, logger


def main(args):
    set_seeds(1)
    torch.set_num_threads(1)

    experiment_id = os.path.dirname(args.checkpoint).split('/')[-1]

    experiment, comet_logger = init_experiment(disable_logging=args.dev,
                                               resume_id=experiment_id)

    model = PLWrapper(BaselineModel())
    test_dataloader = get_dataloader('test',
                                     small=args.dev,
                                     batch_size=8,
                                     pc_size=1024)
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
    parser.add_argument('--checkpoint')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([
            '--dev', '--checkpoint',
            './checkpoints/e49a610f1059418abb73f0837613b696/last.ckpt'
        ])

    load_dotenv()
    main(args)