ulimit -n 2048
python main.py --config exps/CIL_LogoDet-3k_pretrained.json
python main.py --config exps/CIL_LogoDet-3k_nopretrained.json
python main.py --config exps/CIL_LogoDet-3k_pretrained_50.json
python main.py --config exps/CIL_LogoDet-3k_nppretrained_50.json