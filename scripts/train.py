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


def init_experiment(resume_id, disable_logging=False):
    if resume_id:
        experiment = comet_ml.ExistingExperiment(
            api_key=os.environ['COMET_API_KEY'],
            previous_experiment=resume_id,
            disabled=disable_logging)
    else:
        experiment = comet_ml.Experiment(api_key=os.environ['COMET_API_KEY'],
                                         project_name='object-affordances',
                                         disabled=disable_logging)
    logger = CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                         rest_api_key=os.environ.get('COMET_API_KEY'),
                         save_dir='./comet_logs',
                         experiment_key=experiment.get_key(),
                         project_name="object-affordances")
    return experiment, logger


def main(args):
    set_seeds(1)

    if args.resume_id:
        ckpt_path = os.path.join('checkpoints', args.resume_id, 'last.ckpt')
    else:
        ckpt_path = None

    experiment, comet_logger = init_experiment(disable_logging=args.no_logging,
                                               resume_id=args.resume_id)
    torch.set_num_threads(1)

    model = PLWrapper(BaselineModel(), learning_rate=1e-3)

    train_dataloader = get_dataloader('train',
                                      small=True,
                                      batch_size=4,
                                      pc_size=1024)

    checkpoint_cb = ModelCheckpoint(dirpath=os.path.join(
        'checkpoints', experiment.get_key()),
                                    save_top_k=3,
                                    monitor='train_loss',
                                    save_last=True)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.device_count() == 1 else None,
        logger=[comet_logger],
        callbacks=[checkpoint_cb],
        max_epochs=10)

    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-id')
    parser.add_argument('--no-logging', action='store_true', default=False)

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args(['--resume-id', None])

    load_dotenv()
    main(args)

# TODO:
# [ ] Check that resuming experiments works on Comet (online)
# [ ] Make sure everything logs to Comet