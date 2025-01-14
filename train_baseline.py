import argparse
import logging
from abc import abstractmethod
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

import os

from pathlib import Path

from utils.inc_net import DERNet
from trainer import _set_random, print_args, init_logger
from utils.data_manager import DataManager

from config import SEED

parser = argparse.ArgumentParser(description='Download LogoDet-3k.')

parser.add_argument('--dropout', type=float, required=True,
                    help='Dropout rate for fully connected layer.')

parser.add_argument('--baseline-type', type=str, required=True, choices=['resnet152', 'der'],
                    help='Dropout rate for fully connected layer.')

parser.add_argument('--init-cls', type=int, required=True, default=None,
                    help='Dropout rate for fully connected layer.')

parser.add_argument('--increment-cls', type=int, required=True, default=None,
                    help='Dropout rate for fully connected layer.')

parser.add_argument('--n-tasks', type=int, required=False, default=None,
                    help='Number of iterations for incremental learning.')

parser.add_argument('--batch', type=int, required=False, default=512,
                    help='Batch size.')

parser.add_argument('--lr', type=float, required=False, default=1e-3,
                    help='Learning rate.')

parser.add_argument('--epochs', type=int, required=False, default=150,
                    help='Number of maximum training epochs.')

parser.add_argument('--patience', type=int, required=False, default=15,
                    help='Patience for early stopping.')

parser.add_argument('--min-delta', type=float, required=False, default=0.005,
                    help='Min-delta early stopping.')

parsed_args = parser.parse_args()

if parsed_args.baseline_type == 'der':
    assert parsed_args.n_tasks is not None, 'Error: Set the number CIL tasks'

assert 0 <= parsed_args.dropout < 1


experiment_args = {
    "run_name": "BASELINE-{}-from_scratch-{}_classes-drop{}",
    "prefix": "reproduce",
    "dataset": "LogoDet-3K_cropped",
    "shuffle": True,
    "model_name": "der",
    "data_augmentation": True,
    "seed": SEED,

    # Grid search parameters
    "dropout": parsed_args.dropout,
    "convnet_type": 'resnet34',
    "pretrained": True,

    # Baseline method?
    "baseline": True,
    "init_cls": parsed_args.init_cls,
    "increment": parsed_args.increment_cls,
    "n_tasks": parsed_args.n_tasks,

    # Training
    "batch_size": parsed_args.batch,
    "max_epoch": parsed_args.epochs,
    "patience": parsed_args.patience,
    "early_stopping_delta": parsed_args.min_delta,
    "checkpoint_path": Path('model_checkpoint'),
}


def batch_mean(outputs, metric):
    return torch.stack([torch.tensor(x[metric]) for x in outputs]).mean()


class BaselineModel(LightningModule):

    def __init__(self, args, model=None, lr=parsed_args.lr, batch_size=experiment_args['batch_size']):
        super().__init__()
        # Save args
        self.args = args
        self.save_hyperparameters(ignore="model")
        # Datamanager
        self.data_manager = None
        self.test_acc = None
        self.test_acc5 = None
        # Define network backbone
        self.init_network()

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def info(self):
        pass

    def training_step(self, batch, batch_idx):
        loss, train_acc, train_acc5 = self._step_helper(batch)
        return dict(loss=loss, train_acc=train_acc, train_acc5=train_acc5)

    def validation_step(self, batch, batch_idx):
        loss, val_acc, val_acc5 = self._step_helper(batch)
        return dict(loss=loss, val_acc=val_acc, val_acc5=val_acc5)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss, test_acc, test_acc5 = self._step_helper(batch)
        return dict(loss=loss, test_acc=test_acc, test_acc5=test_acc5)

    def _step_helper(self, batch):
        _, x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        y = y.unsqueeze(-1)

        _, preds = torch.sort(logits, dim=1, descending=True)

        acc1 = ((preds[:, 0:1] == y).sum(dim=0) / y.shape[0]).item()
        acc5 = ((preds[:, 0:5] == y).sum(dim=1).sum(dim=0) / y.shape[0]).item()

        return loss, acc1, acc5

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log("train_loss", batch_mean(outputs, 'loss'))
        self.log("train_acc", batch_mean(outputs, 'train_acc'))
        self.log("train_acc5", batch_mean(outputs, 'train_acc5'))

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log("val_loss", batch_mean(outputs, 'loss'))
        self.log("val_acc", batch_mean(outputs, 'val_acc'))
        self.log("val_acc5", batch_mean(outputs, 'val_acc5'))

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.test_acc = batch_mean(outputs, 'test_acc')
        self.test_acc5 = batch_mean(outputs, 'test_acc5')

        self.log("test_loss", batch_mean(outputs, 'loss'))
        self.log("test_acc", self.test_acc)
        self.log("test_acc5", self.test_acc5)

    def on_test_end(self) -> None:
        self.logger.log_metrics(
            {
                'CIL/top1_acc': self.test_acc * 100,
                'CIL/top5_acc': self.test_acc5 * 100,
                'task': 0
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 100], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class BaselineResnet152(BaselineModel):

    def init_network(self):
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.dropout = torch.nn.Dropout(self.args['dropout']) if self.args['dropout'] else None
        self.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features,
            out_features=self.args['init_cls']
        )

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

    def info(self):
        logging.info(self.resnet)
        logging.info(self.dropout)
        logging.info(self.fc)


class DerBaseline(BaselineModel):

    def init_network(self):
        init_cls = self.args["init_cls"]
        increment_cls = self.args["increment"]
        tasks = self.args["n_tasks"]

        model_cil = DERNet(experiment_args['convnet_type'],
                           experiment_args['pretrained'],
                           dropout=experiment_args.get('dropout'))

        for task, n_update in enumerate(np.cumsum([init_cls] + [increment_cls] * tasks)):
            print(n_update)
            model_cil.update_fc(n_update)
            if task > 0:
                for i in range(task):
                    for p in model_cil.convnets[i].parameters():
                        p.requires_grad = True

        model_cil.train()

        self.backbone = model_cil

    def forward(self, x):
        out = self.backbone(x)
        return out['logits']

    def info(self):
        logging.info(self.backbone)
        n_parameters_convnets = sum([x.numel() for x in self.backbone.convnets.parameters()])
        n_parameters_fc = sum([x.numel() for x in self.backbone.fc.parameters()])

        logging.info(f"Total number of parameters: {n_parameters_convnets + n_parameters_fc}")


def train(args):
    # Init logger
    init_logger(args, 'logs')
    # Set up seed
    _set_random()
    # Print args
    print_args(args)

    # Model
    if parsed_args.baseline_type == 'resnet152':
        model = BaselineResnet152(args)
    elif parsed_args.baseline_type == 'der':
        model = DerBaseline(args)

    logging.info('Network architecture')
    model.info()

    # Datamanger
    data_manager = init_datamanager(args)

    # Create run name
    run_name = args['run_name'].format(parsed_args.baseline_type, len(data_manager._class_order), parsed_args.dropout)

    # Init checkpoint
    os.makedirs(args['checkpoint_path'], exist_ok=True)
    Path(args['checkpoint_path'] / Path(run_name).with_suffix('.ckpt')).unlink(missing_ok=True)

    # Init the logger
    tags = ['baseline', 'onlytop']
    if parsed_args.baseline_type == 'resnet152':
        tags.append('baseline-resnet')
    elif parsed_args.baseline_type == 'der':
        tags.append('baseline-der')
    wandb_logger = WandbLogger(project='pycil', name=run_name, tags=tags, config=args)

    # Load dataset
    train_loader, val_loader, test_loader = init_data(data_manager, args)

    # Training
    trainer = Trainer(
        log_every_n_steps=1, accelerator='auto', devices="auto",
        max_epochs=args['max_epoch'],
        logger=wandb_logger,
        callbacks=[
            # Early stopping
            EarlyStopping(monitor="val_acc", min_delta=args['early_stopping_delta'], patience=args['patience'],
                          verbose=True, mode="max"),
            # Model Checkpoint
            ModelCheckpoint(dirpath=args['checkpoint_path'], filename=run_name,
                            monitor='val_acc', save_top_k=1, mode='max', verbose=True)
        ]
    )
    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Test
    trainer.test(
        ckpt_path=str(Path(args['checkpoint_path']) / f"{run_name}.ckpt"),
        dataloaders=test_loader
    )


def init_datamanager(args):
    data_manager = DataManager(
        args['dataset'], args['shuffle'], args['seed'], args['init_cls'],
        args['increment'], data_augmentation=args['data_augmentation']
    )
    return data_manager


def init_data(data_manager, args):
    n_total_classes = len(data_manager._class_order)
    # Create dataset
    train = data_manager.get_dataset(indices=np.arange(0, n_total_classes), source='train', mode='train')
    val = data_manager.get_dataset(indices=np.arange(0, n_total_classes), source='val', mode='test')
    test = data_manager.get_dataset(indices=np.arange(0, n_total_classes), source='test', mode='test')

    # Sanity check
    assert np.unique(train.labels).size == n_total_classes
    assert np.unique(val.labels).size == n_total_classes
    assert np.unique(test.labels).size == n_total_classes

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
    # Train the model
    train(args)


if __name__ == '__main__':
    main(experiment_args)
