from scripts_experiments.train_GNN import train_model
from scripts_experiments.train_TML import train_tml_nested_CV
from options.base_options import BaseOptions

def run_experiments(opt) -> None:
    # Run the experiments
    train_model(opt)

    train_tml_nested_CV(opt)




if __name__ == '__main__':
    opt = BaseOptions().parse()
    run_experiments(opt)