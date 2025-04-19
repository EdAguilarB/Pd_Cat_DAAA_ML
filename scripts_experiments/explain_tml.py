import argparse
import os
import sys

import joblib
import pandas as pd
import shap

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from options.base_options import BaseOptions
from utils.plot_utils import plot_shap


def explain_tml_model(opt: argparse.Namespace) -> None:

    outer, inner = opt.explain_model[0], opt.explain_model[1]

    if opt.tml_representation == "novel_feat":
        results_path = os.path.join(
            opt.tml_representation,
            f"results_{opt.ml_algorithm}",
            f"e_descriptor_{opt.e_descriptor}",
        )

    elif opt.tml_representation == "rdkit":
        results_path = os.path.join(
            opt.tml_representation, f"results_{opt.ml_algorithm}"
        )

    exp_path = os.path.join(opt.log_dir_results, results_path)

    print("Analysing outer {}, inner {}".format(outer, inner))

    model = joblib.load(
        os.path.join(
            exp_path, f"Fold_{outer}_test_set", f"Fold_{inner}_val_set", "model.sav"
        )
    )

    train = pd.read_csv(
        os.path.join(
            exp_path, f"Fold_{outer}_test_set", f"Fold_{inner}_val_set", "train.csv"
        ),
        index_col=0,
    )
    val = pd.read_csv(
        os.path.join(
            exp_path, f"Fold_{outer}_test_set", f"Fold_{inner}_val_set", "val.csv"
        ),
        index_col=0,
    )
    test = pd.read_csv(
        os.path.join(
            exp_path, f"Fold_{outer}_test_set", f"Fold_{inner}_val_set", "test.csv"
        ),
        index_col=0,
    )

    train_data = pd.concat([train, val], axis=0)
    test_data = test

    feat_names = train_data.columns.tolist()[:-3]

    explainer = shap.Explainer(model, output_names=feat_names)
    shap_values = explainer.shap_values(train_data[feat_names])
    print(f"Ploting shap analysis for {str(model)}:\n")
    plot_shap(
        shap_values,
        train_data[feat_names],
        save_path=os.path.join(
            exp_path,
            f"Fold_{outer}_test_set",
            f"Fold_{inner}_val_set",
            "train_shap.png",
        ),
        feat_names=feat_names,
    )

    shap_values = explainer.shap_values(test_data[feat_names])
    print(f"Ploting shap analysis for {str(model)}:\n")
    plot_shap(
        shap_values,
        test_data[feat_names],
        save_path=os.path.join(
            exp_path, f"Fold_{outer}_test_set", f"Fold_{inner}_val_set", "test_shap.png"
        ),
        feat_names=feat_names,
    )


if __name__ == "__main__":
    opt = BaseOptions().parse()
    explain_tml_model(opt)
