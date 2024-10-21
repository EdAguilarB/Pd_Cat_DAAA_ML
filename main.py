from options.base_options import BaseOptions
from data import daaa_reaction_graph
import sys, os
import torch
import time
from copy import deepcopy
from call_methods import make_network, create_loaders
from utils.utils_model import train_network, eval_network, network_report, network_outer_report
from torch_geometric import seed_everything
from icecream import ic
from utils.experiments import train_tml_model_nested_cv, compare_results, explain_tml_model, explain_GNN_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_experiment() -> None:
    opt = BaseOptions().parse()

    seed_everything(opt.global_seed)    

    current_dir = os.getcwd()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = daaa_reaction_graph.daaa_reaction(opt, opt.filename, opt.mol_cols, opt.root)

    # Create the loaders and nested cross validation iterators
    ncv_iterators = create_loaders(data, opt)

    # Initiate the counter of the total runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)

    # Loop through the nested cross validation iterators
    # The outer loop is for the outer fold or test fold
    for outer in range(1, opt.folds+1):
        # The inner loop is for the inner fold or validation fold
        for inner in range(1, opt.folds):

            # Inner fold is incremented by 1 to avoid having same inner and outer fold number for logging purposes
            real_inner = inner +1 if outer <= inner else inner

            # Initiate the early stopping parameters
            val_best_loss = float('inf')
            early_stopping_counter = 0
            best_epoch = 0
            # Increment the counter
            counter += 1
            # Get the data loaders
            train_loader, val_loader, test_loader = next(ncv_iterators)
            # Initiate the lists to store the losses
            train_list, val_list, test_list = [], [], []
            # Create the GNN model
            model = make_network(network_name = "GCN",
                                 opt = opt, 
                                 n_node_features= data.num_node_features).to(device)
            
            # Start the timer for the training
            start_time = time.time()

            for epoch in range(opt.epochs):
                # Checks if the early stopping counter is less than the early stopping parameter
                if early_stopping_counter <= opt.early_stopping:
                    # Train the model
                    train_loss = train_network(model, train_loader, device)
                    # Evaluate the model
                    val_loss = eval_network(model, val_loader, device)
                    test_loss = eval_network(model, test_loader, device)  

                    print('{}/{}-Epoch {:03d} | Train loss: {:.3f} % | Validation loss: {:.3f} % | '             
                        'Test loss: {:.3f} %'.format(counter, TOT_RUNS, epoch, train_loss, val_loss, test_loss))
                    
                    # Model performance is evaluated every 5 epochs
                    if epoch % 5 == 0:
                        # Scheduler step
                        model.scheduler.step(val_loss)
                        # Append the losses to the lists
                        train_list.append(train_loss)
                        val_list.append(val_loss)
                        test_list.append(test_loss)
                        
                        # Save the model if the validation loss is the best
                        if val_loss < val_best_loss:
                            # Best validation loss and early stopping counter updated
                            val_best_loss, best_epoch = val_loss, epoch
                            early_stopping_counter = 0
                            print('New best validation loss: {:.4f} found at epoch {}'.format(val_best_loss, best_epoch))
                            # Save the  best model parameters
                            best_model_params = deepcopy(model.state_dict())
                        else:
                            # Early stopping counter is incremented
                            early_stopping_counter += 1

                    if epoch == opt.epochs:
                        print('Maximum number of epochs reached')

                else:
                    print('Early stopping limit reached')
                    break
            
            print('---------------------------------')
            # End the timer for the training
            training_time = (time.time() - start_time)/60
            print('Training time: {:.2f} minutes'.format(training_time))

            print(f"Training for test outer fold: {outer}, and validation inner fold: {real_inner} completed.")
            print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")

            print('---------------------------------')

            # Report the model performance
            network_report(
                log_dir=f"{current_dir}/{opt.log_dir_results}/results_GNN/",
                loaders=(train_loader, val_loader, test_loader),
                outer=outer,
                inner=real_inner,
                loss_lists=(train_list, val_list, test_list),
                save_all=True,
                model=model,
                model_params=best_model_params,
                best_epoch=best_epoch,
            )

            # Reset the variables of the training
            del model, train_loader, val_loader, test_loader, train_list, val_list, test_list, best_model_params, best_epoch
        
        print(f'All runs for outer test fold {outer} completed')
        print('Generating outer report')

        network_outer_report(
            log_dir=f"{current_dir}/{opt.log_dir_results}/results_GNN/Fold_{outer}_test_set/",
            outer=outer,
        )

        print('---------------------------------')
    
    print('All runs completed')

    explain_GNN_model(exp_path=os.path.join(os.getcwd(), opt.log_dir_results, 'results_GNN'), opt=opt)


    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='rf', e_descriptor='v1')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='rf', e_descriptor='v2')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='rf', e_descriptor='v3')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='rf', e_descriptor='v4')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='gb', e_descriptor='v1')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='gb', e_descriptor='v2')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='gb', e_descriptor='v3')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='gb', e_descriptor='v4')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='lr', e_descriptor='v1')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='lr', e_descriptor='v2')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='lr', e_descriptor='v3')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='lr', e_descriptor='v4')

    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), representation='rdkit')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='gb', representation='rdkit')
    train_tml_model_nested_cv(opt=opt, parent_dir=os.getcwd(), ml_algorithm='lr', representation='rdkit')

    compare_results(opt=opt, exp_dir=os.getcwd()+'/results/comparison/',
                    path1='results/novel_feat/results_rf/e_descriptor_v2', path2='results/results_GNN', 
                    method1='novel_rf_V2', method2='HCat-GNet')

    compare_results(opt=opt, exp_dir=os.getcwd()+'/results/comparison/',
                    path1='results/novel_feat/results_rf/e_descriptor_v2', path2='results/rdkit/results_gb', 
                    method1='novel_rf_V2', method2='rdkit_gb')
    
    novel_desc = ['A(stout)', 'B(volume)', 'B(Hammett)',  'C(volume)', 'C(Hammett)', 'D(volume)', 
                       'D(Hammett)', 'UL(volume)', 'LL(volume)', 'UR(volume)', 'LR(volume)', 'dielectric constant']
    explain_tml_model(exp_path=os.path.join(os.getcwd(), opt.log_dir_results, 'novel_feat', 'results_rf', 'e_descriptor_v2'), \
                      opt=opt, feat_names=novel_desc)

    explain_tml_model(exp_path=os.path.join(os.getcwd(), opt.log_dir_results, 'novel_feat', 'results_gb', 'e_descriptor_v2'), \
                      opt=opt, feat_names=novel_desc)
    
    rdkit_desc = ['fr_ether_substrate','fr_methoxy_substrate','BalabanJ_ligand','BertzCT_ligand','Chi0_ligand',\
                  'Chi0n_ligand','Chi0v_ligand','Chi1_ligand','Chi1n_ligand','Chi1v_ligand','HallKierAlpha_ligand',\
                    'MaxAbsEStateIndex_ligand','MaxAbsPartialCharge_ligand','MaxEStateIndex_ligand','MinAbsEStateIndex_ligand',\
                        'MinEStateIndex_ligand','MinPartialCharge_ligand','MolMR_ligand']

    explain_tml_model(exp_path=os.path.join(os.getcwd(), opt.log_dir_results, 'rdkit', 'results_rf'), \
                      opt=opt, feat_names=rdkit_desc)
    
    rdkit_desc = ['fr_C_O_substrate','fr_C_O_noCOO_substrate','fr_NH0_substrate','fr_Ndealkylation1_substrate','fr_alkyl_halide_substrate','fr_amide_substrate',
                  'fr_aniline_substrate','fr_aryl_methyl_substrate','fr_benzene_substrate','fr_bicyclic_substrate','fr_ester_substrate','fr_ether_substrate','fr_imide_substrate',
                  'fr_ketone_substrate','fr_ketone_Topliss_substrate','fr_lactone_substrate','fr_methoxy_substrate','fr_para_hydroxylation_substrate','Chi0_ligand',
                  'Chi0n_ligand','Chi1n_ligand','Chi1v_ligand','HallKierAlpha_ligand','MaxAbsPartialCharge_ligand','MinAbsEStateIndex_ligand','MinEStateIndex_ligand','MinPartialCharge_ligand',
                  'HA_solv']

    explain_tml_model(exp_path=os.path.join(os.getcwd(), opt.log_dir_results, 'rdkit', 'results_gb'), \
                      opt=opt, feat_names=rdkit_desc)

                        



opt = BaseOptions().parse()

if __name__ == "__main__": 
    run_experiment() 