import argparse
import os, sys
import pandas as pd

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error
from math import sqrt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.utils_model import choose_model, hyperparam_tune, load_variables, electronic_descriptor, \
    split_data, tml_report, network_outer_report, descriptors_all, select_features


def train_tml_nested_CV(opt: argparse.Namespace, parent_dir:str, representation = 'novel_feat', ml_algorithm = 'rf', e_descriptor = 'v1'):

    print('Initialising chiral ligands selectivity prediction using a traditional ML approach.')
    
    # Get the current working directory 
    current_dir = parent_dir

    # Load the data
    filename = opt.filename[:-4] + '_folds' + opt.filename[-4:]
    data = pd.read_csv(f'{opt.root}/raw/{filename}')

    # Select the descriptors
    if representation == 'novel_feat':
        data = data[[electronic_descriptor(e_descriptor), 'A(stout)', 'B(volume)', 'B(Hammett)',  'C(volume)', 'C(Hammett)', 'D(volume)', 'D(Hammett)', 
                     'UL(volume)', 'LL(volume)', 'UR(volume)', 'LR(volume)', 'dielectric constant', '%topA', 'fold', 'index']]
        descriptors = [electronic_descriptor(e_descriptor), 'A(stout)', 'B(volume)', 'B(Hammett)',  'C(volume)', 'C(Hammett)', 'D(volume)', 
                       'D(Hammett)', 'UL(volume)', 'LL(volume)', 'UR(volume)', 'LR(volume)', 'dielectric constant']
        
        dir = f'{representation}/results_{ml_algorithm}/e_descriptor_{e_descriptor}/'
        
    elif representation == 'rdkit':
        data = data[['substrate_smiles', 'ligand_smiles' ,'solvent_smiles', '%topA', 'fold', 'index']]
        data, descriptors = descriptors_all(data)
        data = data[descriptors + ['%topA', 'fold', 'index']]
        dir = f'{representation}/results_{ml_algorithm}/'
    


    # Initiate the counter of the total runs and the total number of runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)    
    print("Number of splits: {}".format(opt.folds))
    print("Total number of runs: {}".format(TOT_RUNS))

    # Hyperparameter optimisation
    print("Hyperparameter optimisation starting...")
    X, y = load_variables(data)
    best_params = hyperparam_tune(X, y, choose_model(best_params=None, algorithm = ml_algorithm), opt.global_seed)
    print('Hyperparameter optimisation has finalised')

    if representation == 'rdkit':
        descriptors = list(select_features(choose_model(best_params, ml_algorithm), X=data[descriptors], y=data['%topA'], names=descriptors))
        data = data[descriptors + ['%topA', 'fold', 'index']]

    X  = data[descriptors]
    X_scaled = RobustScaler().fit_transform(X=X)
    scaled_df = pd.DataFrame(X_scaled, columns=descriptors)
    data[descriptors] = scaled_df

    # Nested cross validation
    ncv_iterator = split_data(data)

    print("Training starting...")
    print("********************************")
    
    # Loop through the nested cross validation iterators
    # The outer loop is for the outer fold or test fold
    for outer in range(1, opt.folds+1):
        # The inner loop is for the inner fold or validation fold
        for inner in range(1, opt.folds):

            # Inner fold is incremented by 1 to avoid having same inner and outer fold number for logging purposes
            real_inner = inner +1 if outer <= inner else inner
            # Increment the counter
            counter += 1

            # Get the train, validation and test sets
            train_set, val_set, test_set = next(ncv_iterator)
            # Choose the model
            model = choose_model(best_params, ml_algorithm)
            # Fit the model
            model.fit(train_set[descriptors], train_set['%topA'])
            # Predict the train set
            preds = model.predict(train_set[descriptors])
            train_rmse = sqrt(mean_squared_error(train_set['%topA'], preds))
            # Predict the validation set
            preds = model.predict(val_set[descriptors])
            val_rmse = sqrt(mean_squared_error(val_set['%topA'], preds))
            # Predict the test set
            preds = model.predict(test_set[descriptors])
            test_rmse = sqrt(mean_squared_error(test_set['%topA'], preds))

            print('Outer: {} | Inner: {} | Run {}/{} | Train RMSE {:.3f} % | Val RMSE {:.3f} % | Test RMSE {:.3f} %'.\
                  format(outer, real_inner, counter, TOT_RUNS, train_rmse, val_rmse, test_rmse) )
            
            # Generate a report of the model performance
            tml_report(log_dir=f"{current_dir}/{opt.log_dir_results}/{dir}/",
                       data = (train_set, val_set, test_set),
                       outer = outer,
                       inner = real_inner,
                       model = model, 
                       descriptors=descriptors)

            
            # Reset the variables of the training
            del model, train_set, val_set, test_set
        
        print('All runs for outer test fold {} completed'.format(outer))
        print('Generating outer report')

        # Generate a report of the model performance for the outer/test fold
        network_outer_report(
            log_dir=f"{current_dir}/{opt.log_dir_results}/{dir}/Fold_{outer}_test_set/",
            outer=outer,
        )

        print('---------------------------------')
        
    print('All runs completed')