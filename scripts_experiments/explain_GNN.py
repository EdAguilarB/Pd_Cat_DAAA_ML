import argparse
import os
import sys

import torch
from icecream import ic
from torch_geometric.explain import CaptumExplainer, Explainer
from torch_geometric.loader import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from model.gcn import GCN_explain
from options.base_options import BaseOptions
from utils.other_utils import get_graph_by_idx, plot_molecule_importance


def explain_GNN_model(opt: argparse.Namespace) -> None:

    exp_path = os.path.join(os.getcwd(), opt.log_dir_results, "results_GNN")

    outer, inner = opt.explain_model[0], opt.explain_model[1]

    print("Analysing outer {}, inner {}".format(outer, inner))

    model_path = os.path.join(
        exp_path, f"Fold_{outer}_test_set", f"Fold_{inner}_val_set"
    )

    train_loader = torch.load(os.path.join(model_path, "train_loader.pth"))
    val_loader = torch.load(os.path.join(model_path, "val_loader.pth"))
    test_loader = torch.load(os.path.join(model_path, "test_loader.pth"))

    all_data = train_loader.dataset + val_loader.dataset + test_loader.dataset

    loader_all = DataLoader(all_data)

    model = GCN_explain(opt, n_node_features=all_data[0].num_node_features)
    model_params = torch.load(os.path.join(model_path, "model_params.pth"))
    model.load_state_dict(model_params)

    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer("ShapleyValueSampling"),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    loader = DataLoader(loader_all.dataset)

    for molecule in tqdm(opt.explain_reactions):
        mol = get_graph_by_idx(loader, molecule)
        print("Analysing reaction {}".format(molecule))
        print("Ligand id: {}".format(mol.ligand_id[0]))
        print("Reaction %top: {:.2f}".format(mol.y.item()))
        print(
            "Reaction predicted %top: {:.2f}".format(
                explainer.get_prediction(
                    x=mol.x, edge_index=mol.edge_index, batch_index=mol.batch
                ).item()
            )
        )
        explanation = explainer(
            x=mol.x, edge_index=mol.edge_index, batch_index=mol.batch
        )
        plot_molecule_importance(
            mol_graph=mol, mol="l", explanation=explanation, palette="normal"
        )


if __name__ == "__main__":
    opt = BaseOptions().parse()
    explain_GNN_model(opt)
