import logging
from typing import Optional, List, Union

import numpy as np
import torchvision.models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import multiprocessing

from torch.nn import functional as F

import pandas as pd

import os

from pathlib import Path

from trainer import _set_random, print_args, init_logger
from utils.data_manager import DataManager, DummyDataset

import sys
sys.path.append('../')
from config import SEED


experiment_args = {
    "run_name": "BASELINE-FAIR_resnet152-from_scratch-100_classes",
    "prefix": "reproduce",
    "dataset": "LogoDet-3K_cropped",
    "shuffle": True,
    "init_cls": 100,
    "increment": 10,
    "model_name": "der",
    "data_augmentation": True,
    "seed": SEED,

    # Grid search parameters
    "dropout": 0.3,
    "convnet_type": None,
    "pretrained": None,

    # Baseline method?
    "baseline": True,

    # Training
    "batch_size": 128,
    "validation_fraction": 0.2,
    "max_epoch": 150,
    "patience": 40,
    "checkpoint_path": Path('model_checkpoint'),

}


class Model(LightningModule):

    def __init__(self, args, model=None, lr=1e-3, batch_size=experiment_args['batch_size']):
        super().__init__()
        # Save args
        self.args = args
        self.save_hyperparameters(ignore="model")
        # Datamanager
        self.data_manager = None
        # Best validation
        self.best_val_acc = 0
        self.test_acc = None
        # Define network backbone
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.dropout = torch.nn.Dropout(self.args['dropout']) if self.args['dropout'] else None
        self.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features,
            out_features=self.args['init_cls']
        )
        assert self.fc.out_features == args['init_cls']

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        if self.dropout:
            x = self.dropout(x)

        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        loss, train_acc = self._step_helper(batch)
        return dict(loss=loss, train_acc=train_acc)

    def validation_step(self, batch, batch_idx):
        loss, val_acc = self._step_helper(batch)
        return dict(loss=loss, val_acc=val_acc)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss, test_acc = self._step_helper(batch)
        return dict(loss=loss, test_acc=test_acc)

    def _step_helper(self, batch):
        _, x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        # Accuracy
        _, preds = torch.max(logits, dim=1)
        n_correct = preds.eq(y.expand_as(preds)).cpu().sum()
        acc = n_correct / y.size(dim=0)

        return loss, acc

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        last = outputs[-1]
        self.log("train_loss", last['loss'])
        self.log("train_acc", last['train_acc'])

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        last = outputs[-1]
        self.log("val_loss", last['loss'])
        self.log("val_acc", last['val_acc'])
        self.best_val_acc = max(self.best_val_acc, last['val_acc'])

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        last = outputs[-1]
        self.log("test_loss", last['loss'])
        self.log("test_acc", last['test_acc'])
        self.test_acc = last['test_acc']

    def on_test_end(self) -> None:
        self.logger.log_metrics({'CIL/top1_acc': self.test_acc * 100, 'task': 0})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 100], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def train(args):
    # Init logger
    init_logger(args, 'logs')
    # Set up seed
    _set_random()
    # Print args
    print_args(args)

    # Model
    model = Model(args)
    logging.info('Network architecture')
    logging.info(model.resnet)
    logging.info(model.dropout)
    logging.info(model.fc)
    wandb_logger = WandbLogger(project='pycil', name=args['run_name'], tags=['baseline', 'onlytop'])

    # Datamanger
    data_manager = init_datamanager(args)
    train_loader, val_loader, test_loader = init_data(data_manager, args)

    # Training
    trainer = Trainer(
        log_every_n_steps=1, accelerator='auto', devices="auto",
        max_epochs=args['max_epoch'],
        logger=wandb_logger,
        callbacks=[
            # Early stopping
            EarlyStopping(monitor="val_acc", min_delta=0.00, patience=args['patience'],
                          verbose=True, mode="max"),
            # Model Checkpoint
            ModelCheckpoint(dirpath=args['checkpoint_path'], filename='best',
                            monitor='val_acc', save_top_k=1, mode='max', verbose=True)
        ]
    )
    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Test
    trainer.test(ckpt_path='best', dataloaders=test_loader)


def init_datamanager(args):
    data_manager = DataManager(
        args['dataset'], args['shuffle'], args['seed'], args['init_cls'],
        args['increment'], data_augmentation=args['data_augmentation']
    )
    return data_manager


def init_data(data_manager, args):
    training_dummy = data_manager.get_dataset(indices=np.arange(0, args['init_cls']), source='train', mode='train')

    # Split train/val
    all_train_df = pd.DataFrame(
        data={
            'images': training_dummy.images,
            'labels': training_dummy.labels
        }
    )
    train_ids = all_train_df.groupby('labels').sample(frac=(1-args['validation_fraction']), random_state=SEED).index
    test_ids = np.array(list(set(all_train_df.index) - set(train_ids)))
    train_df = all_train_df.loc[train_ids]
    val_df = all_train_df.loc[test_ids]

    x_train, x_val, y_train, y_val = train_df["images"], val_df["images"], train_df["labels"], val_df["labels"]

    assert x_train.size == train_ids.size
    assert x_val.size == test_ids.size
    assert y_train.size == train_ids.size
    assert y_val.size == test_ids.size

    # Create dataset
    test = data_manager.get_dataset(indices=np.arange(0, args['init_cls']), source='test', mode='test')
    train = DummyDataset(x_train.values, y_train.values, training_dummy.trsf, True)
    val = DummyDataset(x_val.values, y_val.values, test.trsf, True)

    # Sanity check
    assert np.unique(train.labels).size == args['init_cls']
    assert np.unique(val.labels).size == args['init_cls']
    assert np.unique(test.labels).size == args['init_cls']

    # Return dataloader
    return (
        init_dataloader(train, args['batch_size']),
        init_dataloader(val, args['batch_size']),
        init_dataloader(test, args['batch_size']),
    )


def init_dataloader(split, batch_size):
    return DataLoader(
        split,
        batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()
    )


def main(args):
    # Create checkpoint directory
    os.makedirs(args['checkpoint_path'], exist_ok=True)
    Path(args['checkpoint_path'] / 'best.ckpt').unlink(missing_ok=True)
    # Train the model
    train(args)


if __name__ == '__main__':
    main(experiment_args)
