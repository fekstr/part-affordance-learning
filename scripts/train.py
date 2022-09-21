import comet_ml
import os

from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.pl.pl_wrapper import PLWrapper
from src.models.baseline import BaselineModel
from src.utils import get_dataloader, set_seeds


def init_experiment(resume_id, disable_logging=False, dev=False):
    if resume_id:
        experiment = comet_ml.ExistingExperiment(
            api_key=os.environ['COMET_API_KEY'],
            previous_experiment=resume_id,
            disabled=disable_logging)
    else:
        experiment = comet_ml.Experiment(api_key=os.environ['COMET_API_KEY'],
                                         project_name='object-affordances',
                                         disabled=disable_logging)
        if dev:
            experiment.add_tag('dev')
    logger = CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                         rest_api_key=os.environ.get('COMET_API_KEY'),
                         save_dir='./comet_logs',
                         experiment_key=experiment.get_key(),
                         project_name="object-affordances")
    return experiment, logger


def main(args):
    set_seeds(1)
    torch.set_num_threads(1)

    if args.resume_id:
        ckpt_path = os.path.join('checkpoints', args.resume_id, 'last.ckpt')
    else:
        ckpt_path = None

    experiment, comet_logger = init_experiment(disable_logging=args.no_logging,
                                               resume_id=args.resume_id,
                                               dev=args.dev)

    model = PLWrapper(BaselineModel(), learning_rate=1e-3)

    train_dataloader = get_dataloader('train',
                                      small=args.dev,
                                      batch_size=16,
                                      pc_size=1024)
    valid_dataloader = get_dataloader('valid',
                                      small=args.dev,
                                      batch_size=8,
                                      pc_size=1024)

    checkpoint_cb = ModelCheckpoint(dirpath=os.path.join(
        'checkpoints', experiment.get_key()),
                                    save_top_k=3,
                                    monitor='valid_loss',
                                    save_last=True)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.device_count() == 1 else None,
        logger=[comet_logger],
        callbacks=None if args.no_checkpoints else [checkpoint_cb],
        log_every_n_steps=1 if args.dev else 50,
        max_epochs=10)

    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-id')
    parser.add_argument('--no-logging', action='store_true', default=False)
    parser.add_argument('--no-checkpoints', action='store_true', default=False)
    parser.add_argument('--dev', action='store_true', default=False)

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args(['--dev', '--no-logging', '--no-checkpoints'])

    load_dotenv()

    torch.cuda.empty_cache()
    main(args)
