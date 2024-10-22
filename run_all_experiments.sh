#!/bin/bash


# run training of all combinations of TML methods with all representations

# random forest with novel and all electronic descriptors
python run_experiments.py --tml_representation novel_feat --e_descriptor v1 --ml_algorithm  rf
python run_experiments.py --tml_representation novel_feat --e_descriptor v2 --ml_algorithm  rf
python run_experiments.py --tml_representation novel_feat --e_descriptor v3 --ml_algorithm  rf
python run_experiments.py --tml_representation novel_feat --e_descriptor v4 --ml_algorithm  rf

# gradient boosting with novel and all electronic descriptors
python run_experiments.py --tml_representation novel_feat --e_descriptor v1 --ml_algorithm  gb
python run_experiments.py --tml_representation novel_feat --e_descriptor v2 --ml_algorithm  gb
python run_experiments.py --tml_representation novel_feat --e_descriptor v3 --ml_algorithm  gb
python run_experiments.py --tml_representation novel_feat --e_descriptor v4 --ml_algorithm  gb

# linear regression with novel and all electronic descriptors
python run_experiments.py --tml_representation novel_feat --e_descriptor v1 --ml_algorithm  lr
python run_experiments.py --tml_representation novel_feat --e_descriptor v2 --ml_algorithm  lr
python run_experiments.py --tml_representation novel_feat --e_descriptor v3 --ml_algorithm  lr
python run_experiments.py --tml_representation novel_feat --e_descriptor v4 --ml_algorithm  lr

# rdkit and all ML algorithms
python run_experiments.py --tml_representation rdkit  --ml_algorithm  rf
python run_experiments.py --tml_representation rdkit  --ml_algorithm  gb
python run_experiments.py --tml_representation rdkit  --ml_algorithm  lr




