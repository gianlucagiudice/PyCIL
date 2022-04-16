import pickle
import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import wandb


def train(args):
    seed_list = copy.deepcopy(args['seed'])

    for seed in seed_list:
        args['seed'] = seed
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

    wandb.init(project='pycil', config=args)
    wandb.run.name = f"{'BASELINE_' if args['baseline'] else ''}" \
                     f"{args['convnet_type']}-" \
                     f"{'pretrained' if args['pretrained'] else 'nopretrained'}-" \
                     f"drop{args.get('dropout', 0)}" \
                     f"{'-augmented' if args.get('data_augmentation') else ''}"
    wandb.run.save()

    _set_random()
    _set_device(args)

    print_args(args)
    data_manager = DataManager(
        args['dataset'], args['shuffle'], args['seed'], args['init_cls'],
        args['increment'], args.get('data_augmentation', False)
    )
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


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
