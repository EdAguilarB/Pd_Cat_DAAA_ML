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
            '--mol_cols',
            type=list,
            default=['substrate_smiles', 'ligand_smiles', 'solvent_smiles'],
            help='Column name of the molecule',
        )

        self.parser.add_argument(
            '--folds',
            type=int,
            default=10,
            help='Number of folds for the dataset',
        )


        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt