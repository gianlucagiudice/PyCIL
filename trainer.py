import sys
import logging
import copy
import torch
from pathlib import Path
import os

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import wandb
import numpy as np
from config import SEED


def init_logger(args, dir_path):
    logfilename = '{}/{}-{}_{}_{}_{}_{}_{}_{}'.format(
        dir_path, args['run_name'], args['prefix'], args['seed'], args['model_name'],
        args['convnet_type'], args['dataset'], args['init_cls'], args['increment'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def train(args):
    seed_list = copy.deepcopy(args['seed'])

    for seed in seed_list:
        args['seed'] = seed
        _train(args)


def _train(args):
    # Init data manager
    data_manager = DataManager(
        args['dataset'], args['shuffle'], args['seed'], args['init_cls'],
        args['increment'], args.get('data_augmentation', False)
    )

    # Init logger
    init_logger(args, 'logs')

    # Init tags
    tags = []
    tags += ['adam'] if args.get('adam') else []
    tags += ['baseline'] if args.get('baseline') else ['cil']
    tags += ['onlytop'] if args.get('onlytop') else []

    # Init weights and biases
    wandb.init(project='pycil', config=args, tags=tags)

    wandb.run.name = f"{'BASELINE-' if args.get('baseline', None) else ''}" \
                     f"CIL_{args['init_cls']}_{args['increment']}_{len(data_manager._class_order)}" \
                     f"-{args['convnet_type']}" \
                     f"-{'pretrained' if args['pretrained'] else 'nopretrained'}" \
                     f"-drop{args.get('dropout', 0)}" \
                     f"{'-augmented' if args.get('data_augmentation') else ''}" \
                     f"{'-onlytop' if args.get('onlytop') else ''}" \
                     f"{'-adam' if args.get('adam') else ''}"
    wandb.run.save()

    _set_random()
    _set_device(args)

    print_args(args)

    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))

        wandb.log({'CIL/top1_acc': cnn_accy['top1'], 'CIL/top5_acc': cnn_accy['top5'], 'task': task})

    # Save the model
    logging.info('Saving the model . . .')
    PYCIL_PATH = Path(__file__).parent.resolve()
    os.makedirs(PYCIL_PATH / 'model_checkpoint', exist_ok=True)
    model_path = PYCIL_PATH / 'model_checkpoint' / f'{wandb.run.name}.pt'
    model_path.unlink(missing_ok=True)

    # Create dict
    model_dict = {
        'dropout_rate': model._network.dropout.p,
        'pretrained':  model._network.pretrained,
        'convnet_type': model._network.convnet_type,
        'task_sizes': model._network.task_sizes,
        'feature_dim': model._network.feature_dim,
        'out_dim': model._network.out_dim,
        'state_dict': model._network.state_dict(),
        'class_remap': data_manager._class_order,

        'cil_class2idx': data_manager._class_to_idx,
        'cil_idx2class': data_manager._idx_to_class,

        'cil_prediction2folder': map_prediction2folder(data_manager)
    }

    # Dump dict
    torch.save(model_dict, model_path)
    # Attach dumped file to wandb run
    res = wandb.save(str(model_path))
    assert res is not None
    # Create artifact
    artifact = wandb.Artifact(wandb.run.name, type='model')
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)


def map_prediction2folder(data_manager):
    loader = data_manager.get_dataset(np.arange(0, len(data_manager._class_order)), source='test', mode='test')
    remap = {y: int(Path(path).parts[-2]) for y, path in zip(loader.labels, loader.images)}
    return remap


def _set_device(args):
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(SEED)


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
