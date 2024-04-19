import argparse

class BaseOptions:

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        self.parser.add_argument(
            '--experiment_name',
            type=str,
            default='daaa_ML',
            help='Name of the experiment',
        ),

        self.parser.add_argument(
            '--root',
            type=str,
            default='data/dataset/',
            help='Root directory containing the database',
        )

        self.parser.add_argument(
            '--filename',
            type=str,
            default='DAAA.csv',
            help='Name of the dataset',
        )
        
        self.parser.add_argument(
            '--log_dir_results',
            type=str,
            default='results/',
            help='path to the folder where the results will be saved',
            )
        
        self.parser.add_argument(
            '--mol_cols',
            type=list,
            default=['ligand_smiles', 'substrate_smiles', 'solvent_smiles'],
            help='Column name of the molecule',
        )

        self.parser.add_argument(
            '--folds',
            type=int,
            default=10,
            help='Number of folds for the dataset',
        )

        self.parser.add_argument(
            '--n_classes',
            type=int,
            default=1,
            help='Number of classes',
            )

        self.parser.add_argument(
            '--n_convolutions',
            type=int,
            default=2,
            help='Number of convolutions',
            )
        
        self.parser.add_argument(
            '--readout_layers',
            type=int,
            default=2,
            help='Number of readout layers',
            )
        
        self.parser.add_argument(
            '--embedding_dim',
            type=int,
            default=64,
            help='Embedding dimension',
            )
        
        self.parser.add_argument(
            '--improved',
            type=bool,
            default=False,
            help='Whether to use the improved version of the GCN',
            )
        
        self.parser.add_argument(
            '--problem_type',
            type=str,
            default='regression',
            help='Type of problem',
            )
        
        self.parser.add_argument(
            '--optimizer',
            type=str,
            default='Adam',
            help='Type of optimizer',
            )
        
        self.parser.add_argument(
            '--lr',
            type=float,
            default=0.01,
            help='Learning rate',
            )
        
        self.parser.add_argument(
            '--early_stopping',
            type=int,
            default=6,
            help='Early stopping',
            )
        
        self.parser.add_argument(
            '--scheduler',
            type=str,
            default='ReduceLROnPlateau',
            help='Type of scheduler',
            )
        
        self.parser.add_argument(
            '--step_size',
            type=int,
            default=7,
            help='Step size for the scheduler',
            )
        
        self.parser.add_argument(
            '--gamma',
            type=float,
            default=0.7,
            help='Factor for the scheduler',
            )
        
        self.parser.add_argument(
            '--min_lr',
            type=float,
            default=1e-08,
            help='Minimum learning rate for the scheduler',
            )
        
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=40,
            help='Batch size',
            )
        
        self.parser.add_argument(
            '--epochs',
            type=int,
            default=250,
            help='Number of epochs',
            )  
        
        self.parser.add_argument(
            '--explain_model',
            type=tuple,
            default=(9,5),
            help='Model to explain',
            )
        
        self.parser.add_argument(
            '--explain_reactions',
            type=tuple,
            default=(22,23,24),
            help='Model to explain',
            )
    

        self.parser.add_argument(
            '--global_seed',
            type=int,
            default=2023,
            help='Global random seed for reproducibility',
            )
        


        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt