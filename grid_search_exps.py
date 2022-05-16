from copy import copy
import json
import subprocess
import itertools
import argparse

parser = argparse.ArgumentParser(description='Grid search for experiments.')

parser.add_argument('--baseline', action=argparse.BooleanOptionalAction,
                    help='Gridsearch for baseline.', required=True)

parser.add_argument('--grid-ids', nargs="+", type=int, default=None,
                    help='Gridsearch ids.', required=False)

parser.add_argument('--interactive', action=argparse.BooleanOptionalAction,
                    help='Gridsearch for baseline.', required=False, default=False)

parser.add_argument('--init-cls', type=int, required=True, default=30,
                    help='Number of init classes.')

parser.add_argument('--increment-cls', type=int, required=True, default=30,
                    help='Number of classes for each incremental task.')

parser.add_argument('--memory-per-class', type=int, required=False, default=1000,
                    help='Number of memory per classes for each incremental task.')

parser.add_argument('--min-delta', type=float, required=False, default=0.5,
                    help='Min delta for early stopping.')

parser.add_argument('--weight-align', action=argparse.BooleanOptionalAction,
                    help='Weigh align for incremental networks.', required=False, default=True)

args = parser.parse_args()

config_dict = {
    "run_name": None,
    "prefix": "reproduce",
    "model_name": "der",
    "dataset": "LogoDet-3K_cropped",
    "shuffle": True,
    "seed": [830694],

    "data_augmentation": True,
    "convnet_type": "resnet34",
    "pretrained": True,

    # Weight align
    "weight_align": args.weight_align,

    # Min delta for early stopping
    'min_delta': args.min_delta,

    # CIL task
    "init_cls": args.init_cls,
    "increment": args.increment_cls,

    # Memory for CIL tasks
    "memory_size": 2000,
    "memory_per_class": args.memory_per_class,
    "fixed_memory": True,

    # Grid search parameters
    "dropout": 0,

    # Adam optimizer
    "adam": True,

    # Only top classes
    "onlytop": True,

    # Baseline method?
    "baseline": args.baseline,

    "grid_search_ids": args.grid_ids
}


grid_search_path = "exps/CIL_LogoDet-3k_grid_search.json"

grid_search = [
    # Dropout rate
    [0.5, 0.3, 0.1],
    # Memory per class,
    [50, 100],
    # Pretrained
    [True]
]
grid_search = list(itertools.product(*grid_search))

grid_search_ids = config_dict['grid_search_ids']
if not grid_search_ids:
    grid_search_ids = list(range(1, len(grid_search) + 1))

if args.interactive:
    for i, x in enumerate(grid_search, 1):
        print(f'{i}) {x}')
    ids = input('Insert grid search ids: ')
    grid_search_ids = [int(i) for i in ids.split()]


subprocess.run('ulimit -n 2048', shell=True)
for (i, element) in enumerate(grid_search, 1):
    # Process only ids
    if i not in grid_search_ids:
        continue

    # Print grid search info
    print(f'{"=" * 20} Grid search {i}/{len(grid_search)} {"=" * 20}')

    # Unpack gridsearch
    dropout, memory_per_class, _ = element
    architecture = config_dict['convnet_type']
    pretrained = config_dict['pretrained']

    # Construct dict parameters
    config_dict_temp = copy(config_dict)
    config_dict_temp['run_name'] = f'exp_grid-search{"_BASELINE_" if args.baseline else ""}{i}_' \
                                   f'arch={architecture}_' \
                                   f'pretrained={pretrained}_' \
                                   f'dropout={dropout}'
    config_dict_temp['convnet_type'] = architecture
    config_dict_temp['pretrained'] = pretrained
    config_dict_temp['dropout'] = dropout
    config_dict_temp['memory_per_class'] = memory_per_class

    # Generate temp config file
    with open(grid_search_path, "w") as outfile:
        json.dump(config_dict_temp, outfile, indent=4)

    # Run
    command = f'python main.py --config {grid_search_path}'
    subprocess.run(command, shell=True)
