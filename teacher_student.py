import os

from torch import nn

import wandb

from pathlib import Path

#from config import SEED
SEED = 830694

from utils.inc_net import DERNet
from utils.data_manager import DataManager
from trainer import _set_random, print_args, init_logger

import argparse
import logging
from typing import Optional, List, Union, Any

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


from collections import Counter
import pathlib


def batch_mean(outputs, metric):
    return torch.stack([torch.tensor(x[metric]) for x in outputs]).mean()


if __name__ == '__main__':
    '''
    from pycil.utils.inc_net import DERNet
    from pycil.utils.data_manager import DataManager
    from trainer import _set_random, print_args, init_logger
    '''

    FILE_PATH = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = (FILE_PATH / '..').resolve()

    parser = argparse.ArgumentParser(description='Download LogoDet-3k.')

    parser.add_argument('--dropout', type=float, required=True,
                        help='Dropout rate for fully connected layer.')

    parser.add_argument('--init-cls', type=int, required=True, default=None,
                        help='Dropout rate for fully connected layer.')

    parser.add_argument('--increment-cls', type=int, required=True, default=None,
                        help='Dropout rate for fully connected layer.')

    parser.add_argument('--n-tasks', type=int, required=False, default=None,
                        help='Dropout rate for fully connected layer.')

    parser.add_argument('--batch', type=int, required=False, default=512,
                        help='Batch size.')

    parser.add_argument('--epochs', type=int, required=False, default=85,
                        help='Number of maximum training epochs.')

    parser.add_argument('--patience', type=int, required=False, default=30,
                        help='Number of maximum training epochs.')

    parser.add_argument('--min-delta', type=float, required=False, default=0.0025,
                        help='Number of maximum training epochs.')

    parser.add_argument('--architecture', type=str, required=True, default='resnet50',
                        help='Student architecture.')

    parser.add_argument('--use-memory', type=bool, required=False, default=True,
                        help='Use data memory.')

    parser.add_argument('--teacher-path', type=str, required=True, default=None,
                        help='Teacher model path.')


    parsed_args = parser.parse_args()

    assert 0 <= parsed_args.dropout < 1


    experiment_args = {
        "run_name": "{}-drop{}-mem{}",
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

        # Dataset
        "init_cls": parsed_args.init_cls,
        "increment": parsed_args.increment_cls,

        # Training
        "batch_size": parsed_args.batch,
        "max_epoch": parsed_args.epochs,
        "patience": parsed_args.patience,
        "early_stopping_delta": parsed_args.min_delta,
        "checkpoint_path": Path('model_checkpoint'),
    }


def load_cil_model(cil_model_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model_dict = torch.load(cil_model_path, map_location=device)

    cil_model = DERNet(model_dict['convnet_type'], model_dict['pretrained'], model_dict['dropout_rate'])

    for n_classes in np.cumsum(model_dict['task_sizes']):
        cil_model.update_fc(n_classes)

    cil_model.load_state_dict(model_dict['state_dict'])

    # Eval mode
    cil_model.eval()

    n_classes = len(model_dict['class_remap'])
    assert len(model_dict['cil_class2idx']) == n_classes
    assert len(model_dict['cil_idx2class']) == n_classes

    return (
        cil_model,
        model_dict['cil_idx2class'],
        model_dict['cil_class2idx'],
        model_dict['cil_prediction2folder'],
        dict(data_memory=model_dict['data_memory'], target_memory=model_dict['target_memory'])
    )


class TeacherStudent(LightningModule):

    def __init__(self, teacher_model, student_arch, args, model=None, lr=1e-3, batch_size=256):
        super().__init__()
        # Save args
        self.args = args
        self.save_hyperparameters(ignore="model")
        # Datamanager
        self.data_manager = None
        # Teacher network
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        # Student network
        self.student_model = None
        # Define network backbone
        self.init_network(student_arch)

        # Losses
        self.kl_div_loss = nn.KLDivLoss(log_target=True)
        self.loss_func = nn.CrossEntropyLoss()

        # Temperature
        self.temperature: float = 5.

        self.soft_targets_weight: float = 100.
        self.label_loss_weight: float = 0.5

    def init_network(self, arch):
        resnet_backbone = getattr(torchvision.models, arch)(pretrained=True)
        fc_in_features = resnet_backbone.fc.in_features
        modules = list(resnet_backbone.children())[:-1]
        resnet_backbone = nn.Sequential(*modules)
        self.student_model = resnet_backbone

        self.dropout = torch.nn.Dropout(self.args['dropout']) if self.args['dropout'] else None
        self.fc = torch.nn.Linear(
            in_features=fc_in_features,
            out_features=self.teacher_model.fc.out_features
        )

    def info(self):
        logging.info(self.student_model)
        logging.info(self.dropout)
        logging.info(self.fc)

    def forward(self, x, *args, **kwargs) -> Any:
        x = self.student_model(x)
        x = torch.flatten(x, 1)

        if self.dropout:
            x = self.dropout(x)

        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        soft_targets_loss, label_loss, loss, acc, acc5 = self._step_helper(batch)
        return dict(
            loss_kl=soft_targets_loss,
            loss_label=label_loss,
            loss=loss,
            train_acc=acc,
            train_acc5=acc5
        )

    def validation_step(self, batch, batch_idx):
        soft_targets_loss, label_loss, loss, acc, acc5 = self._step_helper(batch)
        return dict(
            loss_kl=soft_targets_loss,
            loss_label=label_loss,
            loss=loss,
            val_acc=acc,
            val_acc5=acc5
        )

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        soft_targets_loss, label_loss, loss, acc, acc5 = self._step_helper(batch)
        return dict(
            loss_kl=soft_targets_loss,
            loss_label=label_loss,
            loss=loss,
            test_acc=acc,
            test_acc5=acc5
        )

    def _step_helper(self, batch):
        _, x, y = batch

        student_logits = self.forward(x)
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)['logits']

        soft_targets = nn.functional.log_softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)

        soft_targets_loss = self.kl_div_loss(soft_prob, soft_targets)
        label_loss = self.loss_func(student_logits, y)

        loss = self.soft_targets_weight * soft_targets_loss + self.label_loss_weight * label_loss

        # Accuracy
        y = y.unsqueeze(-1)
        _, preds = torch.sort(student_logits, dim=1, descending=True)

        acc1 = ((preds[:, 0:1] == y).sum(dim=0) / y.shape[0]).item()
        acc5 = ((preds[:, 0:5] == y).sum(dim=1).sum(dim=0) / y.shape[0]).item()

        return soft_targets_loss, label_loss, loss, acc1, acc5

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log("train_loss", batch_mean(outputs, 'loss'))
        self.log("train_acc", batch_mean(outputs, 'train_acc'))
        self.log("train_acc5", batch_mean(outputs, 'train_acc5'))

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log("val_loss", batch_mean(outputs, 'loss'))
        self.log("val_acc", batch_mean(outputs, 'val_acc'))
        self.log("val_acc5", batch_mean(outputs, 'val_acc5'))

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.log("test_loss", batch_mean(outputs, 'loss'))
        self.log("test_acc", batch_mean(outputs, 'test_acc'))
        self.log("test_acc5", batch_mean(outputs, 'test_acc5'))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 100], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def train(args):
    # Model
    cil_model, cil_idx2class, cil_class2idx, cil_class_remap, memory = load_cil_model(parsed_args.teacher_path)
    model = TeacherStudent(cil_model, parsed_args.architecture, args)

    # Init logger
    args['run_name'] = args['run_name'].format(
        parsed_args.architecture,
        parsed_args.dropout,
        f'{max(Counter(memory["target_memory"]).values()) if parsed_args.use_memory else "_full"}'
    )
    init_logger(args, 'logs')
    # Set up seed
    _set_random()
    # Print args
    print_args(args)

    logging.info('Network architecture')
    model.info()

    # Datamanger
    data_manager = init_datamanager(args)

    # Create run name
    arch = "resnet50"
    # Create run name
    run_name = args['run_name'].format(arch, parsed_args.dropout)

    # Init checkpoint
    os.makedirs(args['checkpoint_path'], exist_ok=True)
    Path(args['checkpoint_path'] / Path(run_name).with_suffix('.ckpt')).unlink(missing_ok=True)

    # Init the logger
    wandb_logger = WandbLogger(project='knowledge-distillation', name=run_name, config=args)

    # Load dataset
    train_loader, val_loader, test_loader = init_data(data_manager, args, memory)

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

    # Create artifact
    logging.info('Dumping ...')
    final_dict = {'args': args, 'arch': parsed_args.architecture, 'state_dict': model.state_dict()}
    dict_path = str(Path(args['checkpoint_path'] / Path(run_name))) + '.pt'
    torch.save(final_dict, dict_path)
    # Attach dumped file to wandb run
    res = wandb.save(str(dict_path))
    assert res is not None
    logging.info('Dump completed!')
    # Create artifact
    artifact = wandb.Artifact(wandb_logger.experiment.name, type='model')
    artifact.add_file(str(dict_path))
    wandb.log_artifact(artifact)
    # Artifact crated
    logging.info('Artifact created!')


def init_datamanager(args):
    data_manager = DataManager(
        args['dataset'], args['shuffle'], args['seed'], args['init_cls'],
        args['increment'], data_augmentation=args['data_augmentation']
    )
    return data_manager


def init_data(data_manager, args, memory):
    n_total_classes = len(data_manager._class_order)

    # Create dataset
    if not parsed_args.use_memory:
        train = data_manager.get_dataset(indices=np.arange(0, n_total_classes), source='train', mode='train')
    else:
        resolve_path = lambda p: str(PROJECT_ROOT / os.path.join(*pathlib.Path(p).parts[-5:]))
        data_memory = [resolve_path(x) for x in memory['data_memory']]
        appendent = [data_memory, memory['target_memory']]
        train = data_manager.get_dataset([], 'train', 'train', appendent=appendent)

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
