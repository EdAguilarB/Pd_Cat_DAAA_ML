import os

from options.base_options import BaseOptions
from scripts_experiments.compare_methods import compare_results
from scripts_experiments.explain_GNN import explain_GNN_model
from scripts_experiments.explain_tml import explain_tml_model
from scripts_experiments.train_GNN import train_model
from scripts_experiments.train_TML import train_tml_nested_CV

terms_dict = {
    'rf': 'Random Forest',
    'gb': 'Gradient Boosting',
    'lr': 'Linear Regression',
}


def run_experiments(opt) -> None:
    # Run the experiments

    print('Running experiments with the following configurations:')
    print(opt)

    if not os.path.exists(os.path.join(opt.log_dir_results, 'results_GNN')):
        train_model(opt)
    else:
        print('GNN model already trained in dir: {}'.format(os.path.join(opt.log_dir_results, 'results_GNN')))

    if opt.tml_representation == 'novel_feat':
        path = f'{opt.tml_representation}/results_{opt.ml_algorithm}/e_descriptor_{opt.e_descriptor}/'
        method = f'Bespoke Descriptors {terms_dict[opt.ml_algorithm]} {opt.e_descriptor}'
    elif opt.tml_representation == 'rdkit':
        path = f'{opt.tml_representation}/results_{opt.ml_algorithm}/'
        method = f'{opt.tml_representation}_{opt.ml_algorithm}'

    if not os.path.exists(os.path.join(opt.log_dir_results, path)):
        print(os.path.join(opt.log_dir_results, path))
        train_tml_nested_CV(opt, parent_dir=os.getcwd())
    else:
        print(f'TML model already trained with these configurations in dir: {os.path.join(opt.log_dir_results, path)}')

    if not os.path.exists(os.path.join(opt.log_dir_results, 'comparison', f'HCat-GNet_vs_{method}')):
        compare_results(opt=opt, exp_dir = f'comparison',
                        path1=os.path.join(opt.log_dir_results, 'results_GNN'),
                        path2=os.path.join(opt.log_dir_results, path),
                        method1='HCat-GNet', method2=method)
    else:
        print(f'Comparison already done in dir: {os.path.join(opt.log_dir_results, "comparison", f"HCat-GNet_vs_{method}")}')

    explain_tml_model(opt)
    explain_GNN_model(opt)





if __name__ == '__main__':
    opt = BaseOptions().parse()
    run_experiments(opt)