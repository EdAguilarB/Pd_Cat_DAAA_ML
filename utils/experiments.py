import os
import sys
import joblib
from options.base_options import BaseOptions
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, CaptumExplainer
import pandas as pd
import numpy as np
from utils.utils_model import choose_model, hyperparam_tune, load_variables, electronic_descriptor, \
    split_data, tml_report, network_outer_report, extract_metrics, descriptors_all, select_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score
from utils.plot_utils import create_bar_plot, create_violin_plot, create_strip_plot, \
    create_parity_plot, plot_importances, plot_shap
from model.gcn import GCN_explain
from math import sqrt
import argparse
import shap
from utils.other_utils import plot_molecule_importance, get_graph_by_idx

from icecream import ic


sys.path.append(os.path.dirname(os.path.abspath(__file__)))





def compare_results(exp_dir, opt, path1, path2, method1, method2):


    r2_gnn, mae_gnn, rmse_gnn = [], [], []
    accuracy_gnn, precision_gnn, recall_gnn = [], [], []
    r2_tml, mae_tml, rmse_tml = [], [], []
    accuracy_tml, precision_tml, recall_tml = [], [], []

    results_all = pd.DataFrame(columns = ['index', 'Test_Fold', 'Val_Fold', 'Method', 'real_%top', 'predicted_%top'])
    
    for outer in range(1, opt.folds+1):

        outer_m1 = os.path.join(path1, f'Fold_{outer}_test_set')
        outer_m2 = os.path.join(path2, f'Fold_{outer}_test_set')

        metrics_gnn = extract_metrics(file=outer_m1+f'/performance_outer_test_fold{outer}.txt')
        metrics_tml = extract_metrics(file=outer_m2+f'/performance_outer_test_fold{outer}.txt')

        r2_gnn.append(metrics_gnn['R2'])
        mae_gnn.append(metrics_gnn['MAE'])
        rmse_gnn.append(metrics_gnn['RMSE'])
        accuracy_gnn.append(metrics_gnn['Accuracy'])
        precision_gnn.append(metrics_gnn['Precision'])
        recall_gnn.append(metrics_gnn['Recall'])

        r2_tml.append(metrics_tml['R2'])
        mae_tml.append(metrics_tml['MAE'])
        rmse_tml.append(metrics_tml['RMSE'])
        accuracy_tml.append(metrics_tml['Accuracy'])
        precision_tml.append(metrics_tml['Precision'])
        recall_tml.append(metrics_tml['Recall'])

        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            m1_dir = os.path.join(path1, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_gnn = pd.read_csv(m1_dir+'/predictions_test_set.csv')
            df_gnn['Test_Fold'] = outer
            df_gnn['Val_Fold'] = real_inner
            df_gnn['Method'] = method1

            results_all = pd.concat([results_all, df_gnn], axis=0)

            m2_dir = os.path.join(path2, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_tml = pd.read_csv(m2_dir+'/predictions_test_set.csv')
            df_tml['Test_Fold'] = outer
            df_tml['Val_Fold'] = real_inner
            df_tml['Method'] = method2

            results_all = pd.concat([results_all, df_tml], axis=0)

    save_dir = f'{exp_dir}/{method1}_vs_{method2}'
    os.makedirs(save_dir, exist_ok=True)
    
    mae_mean_gnn = np.array([entry['mean'] for entry in mae_gnn])
    mae_gnn_std = np.array([entry['std'] for entry in mae_gnn])

    mae_mean_tml = np.array([entry['mean'] for entry in mae_tml])
    mae_tml_std = np.array([entry['std'] for entry in mae_tml])

    rmse_mean_gnn = np.array([entry['mean'] for entry in rmse_gnn])
    rmse_gnn_std = np.array([entry['std'] for entry in rmse_gnn])

    rmse_mean_tml = np.array([entry['mean'] for entry in rmse_tml])
    rmse_tml_std = np.array([entry['std'] for entry in rmse_tml])

    minimun = np.min(np.array([
        (mae_mean_gnn - mae_gnn_std).min(), 
        (mae_mean_tml - mae_tml_std).min(),
        (rmse_mean_gnn - rmse_gnn_std).min(), 
        (rmse_mean_tml - rmse_tml_std).min()]))
    
    maximun = np.max(np.array([
        (mae_mean_gnn + mae_gnn_std).max(),
        (mae_mean_tml + mae_tml_std).max(),
        (rmse_mean_gnn + rmse_gnn_std).max(), 
        (rmse_mean_tml + rmse_tml_std).max()]))
    
    create_bar_plot(means=(mae_mean_gnn, mae_mean_tml), stds=(mae_gnn_std, mae_tml_std), min = minimun, max = maximun, metric = 'MAE', save_path= save_dir, method1=method1, method2=method2)
    create_bar_plot(means=(rmse_mean_gnn, rmse_mean_tml), stds=(rmse_gnn_std, rmse_tml_std), min = minimun, max = maximun, metric = 'RMSE', save_path= save_dir, method1=method1, method2=method2)

    r2_mean_gnn = np.array([entry['mean'] for entry in r2_gnn])
    r2_gnn_std = np.array([entry['std'] for entry in r2_gnn])

    r2_mean_tml = np.array([entry['mean'] for entry in r2_tml])
    r2_tml_std = np.array([entry['std'] for entry in r2_tml])

    accuracy_mean_gnn = np.array([entry['mean'] for entry in accuracy_gnn])
    accuracy_gnn_std = np.array([entry['std'] for entry in accuracy_gnn])

    accuracy_mean_tml = np.array([entry['mean'] for entry in accuracy_tml])
    accuracy_tml_std = np.array([entry['std'] for entry in accuracy_tml])

    precision_mean_gnn = np.array([entry['mean'] for entry in precision_gnn])
    precision_gnn_std = np.array([entry['std'] for entry in precision_gnn])

    precision_mean_tml = np.array([entry['mean'] for entry in precision_tml])
    precision_tml_std = np.array([entry['std'] for entry in precision_tml])

    recall_mean_gnn = np.array([entry['mean'] for entry in recall_gnn])
    recall_gnn_std = np.array([entry['std'] for entry in recall_gnn])

    recall_mean_tml = np.array([entry['mean'] for entry in recall_tml])
    recall_tml_std = np.array([entry['std'] for entry in recall_tml])
    
    minimun = np.min(np.array([
    (r2_mean_gnn - r2_gnn_std).min(),
    (r2_mean_tml - r2_tml_std).min(),
    (accuracy_mean_gnn - accuracy_gnn_std).min(),
    (accuracy_mean_tml - accuracy_tml_std).min(),
    (precision_mean_gnn - precision_gnn_std).min(),
    (precision_mean_tml - precision_tml_std).min(),
    (recall_mean_gnn - recall_gnn_std).min(),
    (recall_mean_tml - recall_tml_std).min()
    ]))

    maximun = np.max(np.array([
    (r2_mean_gnn + r2_gnn_std).max(),
    (r2_mean_tml + r2_tml_std).max(),
    (accuracy_mean_gnn + accuracy_gnn_std).max(),
    (accuracy_mean_tml + accuracy_tml_std).max(),
    (precision_mean_gnn + precision_gnn_std).max(),
    (precision_mean_tml + precision_tml_std).max(),
    (recall_mean_gnn + recall_gnn_std).max(),
    (recall_mean_tml + recall_tml_std).max()
    ]))
    
    
    create_bar_plot(means=(r2_mean_gnn, r2_mean_tml), stds=(r2_gnn_std, r2_tml_std), min = minimun, max = maximun, metric = 'R2', save_path= save_dir, method1=method1, method2=method2)
    create_bar_plot(means=(accuracy_mean_gnn, accuracy_mean_tml), stds=(accuracy_gnn_std, accuracy_tml_std), min = minimun, max = maximun, metric = 'Accuracy', save_path= save_dir, method1=method1, method2=method2)
    create_bar_plot(means=(precision_mean_gnn, precision_mean_tml), stds=(precision_gnn_std, precision_tml_std), min = minimun, max = maximun, metric = 'Precision', save_path= save_dir, method1=method1, method2=method2)
    create_bar_plot(means=(recall_mean_gnn, recall_mean_tml), stds=(recall_gnn_std, recall_tml_std), min = minimun, max = maximun, metric = 'Recall', save_path= save_dir, method1=method1, method2=method2)

    results_all['Error'] = results_all['real_%top'] - results_all['predicted_%top']
    results_all['real_face'] = np.where(results_all['real_%top'] > 50, 1, 0)
    results_all['predicted_face'] = np.where(results_all['predicted_%top'] > 50, 1, 0)

    create_violin_plot(data=results_all, save_path= save_dir)
    create_strip_plot(data=results_all, save_path= save_dir)

    create_parity_plot(data=results_all, save_path= save_dir, method1=method1, method2=method2)

    results_all = results_all.reset_index(drop=True)
    results_all.to_csv(f'{save_dir}/predictions_all.csv', index=False)


    print('All plots have been saved in the directory {}'.format(save_dir))

    gnn_predictions = results_all[results_all['Method'] == method1]

    print('\n')

    print(f'Final metrics for {method1}:')

    print('Accuracy: {:.3f}'.format(accuracy_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('Precision: {:.3f}'.format(precision_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('Recall: {:.3f}'.format(recall_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('R2: {:.3f}'.format(r2_score(gnn_predictions['real_%top'], gnn_predictions['predicted_%top'])))
    print('MAE: {:.3f}'.format(mean_absolute_error(gnn_predictions['real_%top'], gnn_predictions['predicted_%top'])))
    print('RMSE: {:.3f} \n'.format(sqrt(mean_squared_error(gnn_predictions['real_%top'], gnn_predictions['predicted_%top']))))
    
    tml_predictions = results_all[results_all['Method'] == method2]
    print(f'Final metrics for {method2}:')
    print('Accuracy: {:.3f}'.format(accuracy_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('Precision: {:.3f}'.format(precision_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('Recall: {:.3f}'.format(recall_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('R2: {:.3f}'.format(r2_score(tml_predictions['real_%top'], tml_predictions['predicted_%top'])))
    print('MAE: {:.3f}'.format(mean_absolute_error(tml_predictions['real_%top'], tml_predictions['predicted_%top'])))
    print('RMSE: {:.3f} \n'.format(sqrt(mean_squared_error(tml_predictions['real_%top'], tml_predictions['predicted_%top']))))


def explain_tml_model(exp_path:str, opt: argparse.Namespace, feat_names) -> None:   

    outer, inner = opt.explain_model[0], opt.explain_model[1]

    print('Analysing outer {}, inner {}'.format(outer, inner))

    model = joblib.load(os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set', 'model.sav'))
    explainer = shap.Explainer(model, output_names=feat_names)

    train = pd.read_csv(os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set', 'train.csv'))
    val = pd.read_csv(os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set', 'val.csv'))
    test = pd.read_csv(os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set', 'test.csv'))

    all_data = pd.concat([train, val, test], axis=0)

    shap_values = explainer.shap_values(all_data[feat_names])

    print(f'Ploting shap analysis for {str(model)}:\n')


    plot_shap(shap_values, all_data[feat_names], save_path=os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set'), feat_names=feat_names)


    

def explain_GNN_model(exp_path:str, opt: argparse.Namespace) -> None:

    outer, inner = opt.explain_model[0], opt.explain_model[1]

    print('Analysing outer {}, inner {}'.format(outer, inner))

    model_path = os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set')

    train_loader = torch.load(os.path.join(model_path, 'train_loader.pth'))
    val_loader = torch.load(os.path.join(model_path, 'val_loader.pth'))
    test_loader = torch.load(os.path.join(model_path, 'test_loader.pth'))

    all_data = train_loader.dataset + val_loader.dataset + test_loader.dataset

    loader_all = DataLoader(all_data)

    model = GCN_explain(opt, n_node_features=all_data[0].num_node_features)
    model_params = torch.load(os.path.join(model_path, 'model_params.pth'))
    model.load_state_dict(model_params)


    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('ShapleyValueSampling'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    loader = DataLoader(loader_all.dataset)

    for molecule in tqdm(opt.explain_reactions):
        mol = get_graph_by_idx(loader, molecule)
        print('Analysing reaction {}'.format(molecule))
        print('Ligand id: {}'.format(mol.ligand_id[0]))
        print('Reaction %top: {:.2f}'.format(mol.y.item()))
        print('Reaction predicted %top: {:.2f}'.format(explainer.get_prediction(x = mol.x, edge_index=mol.edge_index, batch_index=mol.batch).item()))
        explanation = explainer(x = mol.x, edge_index=mol.edge_index,  batch_index=mol.batch)
        plot_molecule_importance(mol_graph=mol, mol='l', explanation=explanation, palette='normal')


