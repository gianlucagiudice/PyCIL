from copy import copy
import json
import subprocess
import itertools

config_dict = {
    "run_name": "exp02_docker-top-frequency-nopretrained-50",
    "prefix": "reproduce",
    "dataset": "LogoDet-3K_cropped",
    "memory_size": 2000,
    "memory_per_class": 1000,
    "fixed_memory": True,
    "shuffle": True,
    "init_cls": 30,
    "increment": 10,
    "model_name": "der",
    "data_augmentation": True,
    "seed": [830694],

    # Grid search parameters
    "dropout": None,
    "convnet_type": None,
    "pretrained": None,
}

grid_search_path = "exps/CIL_LogoDet-3k_grid_search.json"

grid_search = [
    ["resnet50"],
    [True, False],
    [0.1, 0.3, 0.5]
]
grid_search = list(itertools.product(*grid_search))

subprocess.run('ulimit -n 2048', shell=True)
for (i, element) in enumerate(grid_search):
    # Print grid search info
    print(f'Grid search {i+1}/{len(grid_search)}')

    # Unpack gridsearch
    architecture, pretrained, dropout = element

    # Construct dict parameters
    config_dict_temp = copy(config_dict)
    config_dict_temp['run_name'] = f'exp_grid-search{i}_' \
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
