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

args = parser.parse_args()

config_dict = {
    "run_name": "exp02_docker-top-frequency-nopretrained-50",
    "prefix": "reproduce",
    "dataset": "LogoDet-3K_cropped",
    "memory_size": 2000,
    "memory_per_class": 1000,
    "fixed_memory": True,
    "shuffle": True,
    "init_cls": 100 if args.baseline else 30,
    "increment": 10,
    "model_name": "der",
    "data_augmentation": True,
    "seed": [830694],

    # Grid search parameters
    "dropout": 0,
    "convnet_type": None,
    "pretrained": None,

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
    # Architecture
    ["resnet34"],
    # Pretrained
    [True, False],
    # Dropout rate
    [0.1, 0.3, 0.5]
]
grid_search = list(itertools.product(*grid_search))

grid_search_ids = config_dict['grid_search_ids']
if not grid_search_ids:
    grid_search_ids = list(range(1, len(grid_search) + 1))

subprocess.run('ulimit -n 2048', shell=True)
for (i, element) in enumerate(grid_search, 1):
    # Process only ids
    if i not in grid_search_ids:
        continue

    # Print grid search info
    print(f'{"=" * 20} Grid search {i}/{len(grid_search)} {"=" * 20}')

    # Unpack gridsearch
    architecture, pretrained, dropout = element

    # Construct dict parameters
    config_dict_temp = copy(config_dict)
    config_dict_temp['run_name'] = f'exp_grid-search{"_BASELINE_" if args.baseline else ""}{i}_' \
                                   f'arch={architecture}_' \
                                   f'pretrained={pretrained}_' \
                                   f'dropout={dropout}'
    config_dict_temp['convnet_type'] = architecture
    config_dict_temp['pretrained'] = pretrained
    config_dict_temp['dropout'] = dropout

    # Generate temp config file
    with open(grid_search_path, "w") as outfile:
        json.dump(config_dict_temp, outfile, indent=4)

    # Run
    command = f'python main.py --config {grid_search_path}'
    subprocess.run(command, shell=True)
