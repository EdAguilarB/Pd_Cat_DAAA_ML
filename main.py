from options.base_options import BaseOptions
from data import daaa_reaction_graph
import sys, os
import torch
import time
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_experiment() -> None:
    opt = BaseOptions().parse()

    current_dir = os.getcwd()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ligands = daaa_reaction_graph.daaa_reaction(opt, opt.filename, opt.mol_cols, opt.root)

    print("Dataset type: ", type(ligands))
    print("Dataset node features: ", ligands.num_features)
    print("Dataset length: ", ligands.len)
    print("Dataset sample: ", ligands[0])
    print('Sample features: ',  ligands[0].x)
    print('Sample outcome: ',  ligands[0].y)


    

if __name__ == "__main__": 
    run_experiment() 