import os
import os.path
from argparse import ArgumentParser
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import arguments
import data
from model import CMBClassifier


def log_config(args, trainer):
    config = {**vars(args), **vars(trainer)}

    log_dir = trainer.logger.log_dir
    os.makedirs(log_dir, exist_ok=True)
    config_log_path = os.path.join(log_dir, "config.yaml")

    with open(config_log_path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)


def append_to_log(key: str, value: Any, trainer):
    log_dir = trainer.logger.log_dir
    os.makedirs(log_dir, exist_ok=True)
    config_log_path = os.path.join(log_dir, "config.yaml")

    with open(config_log_path, "a") as config_file:
        yaml.dump({key: value}, config_file, default_flow_style=False)
        # config_file.write(f'{key}: {value}\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())

    dataset_train = arguments.add_arguments(parser, "dataset", data.LazyBox)(
        split_partition="train"
    )
    dataset_val = arguments.add_arguments(parser, "dataset", data.LazyBoxVal)(
        split_partition="val"
    )

    activation_function = arguments.add_options_from_module(
        parser, "activation", torch.nn.modules.activation, torch.nn.Module
    )

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logger = pl.loggers.CSVLogger("logs")

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args=args,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=10,
        callbacks=[
            ModelCheckpoint(
                monitor="train_loss",
                save_top_k=3,
                mode="min",
            )
        ],
    )

    log_config(args, trainer)

    model = CMBClassifier(activation_function=activation_function)

    train_data_loader = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_data_loader = DataLoader(
        dataset_val, shuffle=False, batch_size=1, num_workers=args.num_workers
    )

    if args.learning_rate is None:
        lr_finder = trainer.tuner.lr_find(model, train_data_loader)
        lr_fig = lr_finder.plot(suggest=True)
        lr_fig.savefig(os.path.join(trainer.logger.log_dir, "lr.png"))

        lr = lr_finder.suggestion()

        model.learning_rate = lr
        print("Auto LR:", model.learning_rate)
        append_to_log("found_lr", lr, trainer)
    else:
        model.learning_rate = args.learning_rate

    trainer.fit(model, train_data_loader, val_data_loader)
