import pickle
import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import multiprocessing
import wandb


def setup_train_device(args):
    try:
        device = copy.deepcopy(args['device'])
    except KeyError:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
    return device


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = setup_train_device(args)
    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)


def _train(args):
    logfilename = 'logs/{}-{}_{}_{}_{}_{}_{}_{}'.format(
        args['run_name'], args['prefix'], args['seed'], args['model_name'],
        args['convnet_type'], args['dataset'], args['init_cls'], args['increment'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    wandb.init(project='pycil')

    _set_random()
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
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

            wandb.log({'top1-acc': cnn_curve['top1']})
            wandb.log({'top5-acc': cnn_curve['top5']})

        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))

            wandb.log({'top1-acc': cnn_curve['top1']})
            wandb.log({'top5-acc': cnn_curve['top5']})

    # Dump training history
    logging.info('Dumping training hitsory . . .')
    filename = 'logs/{}_{}_{}_{}_{}_{}_{}_training-history.pickle'.format(
        args['prefix'], args['seed'], args['model_name'], args['convnet_type'],
        args['dataset'], args['init_cls'], args['increment'])

    with open(filename, 'wb') as handle:
        pickle.dump(model._training_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filename, 'rb') as handle:
        _ = pickle.load(handle)


def _set_device(args):
    device_type = args['device']
    if device_type != -1:
        args['device'] = [torch.device(f'cuda:{device}') for device in device_type]


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
