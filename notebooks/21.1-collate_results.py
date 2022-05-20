# %%
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# %%
# CHANGE: 'runs' is a dict mapping from experiment to a list of all multiruns for that experiment.
# See the example below. You need to manually add them to these lists.
# You can leave lists empty.
runs = {
    # example for baseline experiments
    "Public Split": ["PATH_TO_CLONED_REPO/multirun/2022-05-17/13-00-00",],
    "Proportional Splits": [  # training size
        # add paths
    ],
    "Width": [
        # add
    ],
    "Dropout": [
        # add
    ],
    "Layers": [
        # add
    ],
    "L2": [
        # add
    ],
    "Optimizer": [
        # add
    ],
    # Additional experiments regarding stability with fixed seeds or CPU compute platform with otherwise baseline models
    "PublicAllStable": [
        # add
    ],
    "PublicGATCPU": [
        # add
    ],
    "PublicFixedSeed": [
        # add
    ],
    "PublicFixedSeedCPU": [
        # add
    ],
}
run_to_fname = {
    "Public Split": "publicsplit.parquet",
    "Proportional Splits": "proportionalsplit.parquet",
    "Width": "width.parquet",
    "Dropout": "dropout.parquet",
    "Layers": "layers.parquet",
    "L2": "l2.parquet",
    "Optimizer": "optimizer.parquet",
    "PublicAllStable": "publicallstable.parquet",
    "PublicGATCPU": "public_gatcpu.parquet",
    "PublicFixedSeed": "public_fixedseed.parquet",
    "PublicFixedSeedCPU": "public_fixedseed_cpu.parquet",
}
runs = {exp: [Path(p) for p in ps] for exp, ps in runs.items()}
# %%
def add_to_frame(df, path, accs, cfg, val_name):
    vals = np.load(path)
    for val, (i, j) in zip(vals, itertools.combinations(np.arange(len(accs)), 2)):
        df["Dataset"].append(cfg.dataset.name)
        df["Model"].append(cfg.model.name)
        df["Value"].append(val)
        df["Metric"].append(val_name)
        df["Acc1"].append(accs[i])
        df["Acc2"].append(accs[j])
        df["Acc-mean"].append(np.mean(accs))
        df["Acc-std"].append(np.std(accs))
        df["Val Size"].append(cfg.part_val)
        df["Test Size"].append(cfg.part_test)
        df["Init Seed 1"].append(cfg.seed + i)
        df["Init Seed 2"].append(cfg.seed + j)
        df["Split Seed"].append(cfg.datasplit_seed)
        if hasattr(cfg.model, "n_heads"):
            width = cfg.model.n_heads * cfg.model.hidden_dim
        else:
            width = cfg.model.hidden_dim
        df["Width"].append(width)
        df["Dropout"].append(cfg.model.dropout_p)
        if hasattr(cfg.model, "n_layers"):
            layers = cfg.model.n_layers
        else:
            layers = 2
        df["Layers"].append(layers)
        df["L2"].append(cfg.optim.weight_decay)
        if cfg.optim.name == "SGD":
            optim = f"SGD-{cfg.optim.momentum:.1f}M"
        else:
            optim = cfg.optim.name
        df["Optimizer"].append(optim)


columns = [
    "Dataset",
    "Model",
    "Value",
    "Metric",
    "Acc1",
    "Acc2",
    "Acc-mean",
    "Acc-std",
    "Val Size",
    "Test Size",
    "Init Seed 1",
    "Init Seed 2",
    "Split Seed",
    "Width",
    "Dropout",
    "Layers",
    "L2",
    "Optimizer",
]

for runtype, roots in runs.items():
    df = {c: [] for c in columns}
    for root in roots:
        print(f"{datetime.now()}: Processing {root}")
        for experiment in root.iterdir():
            if not experiment.is_dir():
                continue

            cfg_path = experiment / ".hydra" / "config.yaml"
            cfg = OmegaConf.load(cfg_path)

            predictions_dir = experiment / "predictions"
            with (predictions_dir / "evals.json").open() as f:
                evals = json.load(f)
            accs = [e["test_acc"] for e in evals]

            add_to_frame(df, predictions_dir / "pi_distr.npy", accs, cfg, "PI")
            add_to_frame(df, predictions_dir / "normpi_distr.npy", accs, cfg, "NormPI")
            add_to_frame(df, predictions_dir / "l1_distr.npy", accs, cfg, "MAE")
            add_to_frame(df, predictions_dir / "symkl_distr.npy", accs, cfg, "SymKL")

            for path, val_name in zip(
                [
                    predictions_dir / "true_pi_distr.npy",
                    predictions_dir / "false_pi_distr.npy",
                ],
                ["True PI", "False PI"],
            ):
                vals = np.load(path)
                for val_idx, (i, j) in zip(
                    range(0, len(vals), 2),
                    itertools.combinations(np.arange(len(accs)), 2),
                ):
                    df["Dataset"].append(cfg.dataset.name)
                    df["Model"].append(cfg.model.name)
                    df["Value"].append(vals[val_idx])
                    df["Metric"].append(val_name)
                    df["Acc1"].append(accs[i])
                    df["Acc2"].append(accs[j])
                    df["Acc-mean"].append(np.mean(accs))
                    df["Acc-std"].append(np.std(accs))
                    df["Val Size"].append(cfg.part_val)  ##
                    df["Test Size"].append(cfg.part_test)  ##
                    df["Init Seed 1"].append(cfg.seed + i)  ##
                    df["Init Seed 2"].append(cfg.seed + j)  ##
                    df["Split Seed"].append(cfg.datasplit_seed)  ##
                    if hasattr(cfg.model, "n_heads"):
                        width = cfg.model.n_heads * cfg.model.hidden_dim
                    else:
                        width = cfg.model.hidden_dim
                    df["Width"].append(width)
                    df["Dropout"].append(cfg.model.dropout_p)
                    if hasattr(cfg.model, "n_layers"):
                        layers = cfg.model.n_layers
                    else:
                        layers = 2
                    df["Layers"].append(layers)
                    df["L2"].append(cfg.optim.weight_decay)
                    if cfg.optim.name == "SGD":
                        optim = f"SGD-{cfg.optim.momentum:.1f}M"
                    else:
                        optim = cfg.optim.name
                    df["Optimizer"].append(optim)

                    df["Dataset"].append(cfg.dataset.name)
                    df["Model"].append(cfg.model.name)
                    df["Value"].append(vals[val_idx + 1])
                    df["Metric"].append(val_name)
                    df["Acc1"].append(accs[j])
                    df["Acc2"].append(accs[i])
                    df["Acc-mean"].append(np.mean(accs))
                    df["Acc-std"].append(np.std(accs))
                    df["Val Size"].append(cfg.part_val)  ##
                    df["Test Size"].append(cfg.part_test)  ##
                    df["Init Seed 1"].append(cfg.seed + j)  ##
                    df["Init Seed 2"].append(cfg.seed + i)  ##
                    df["Split Seed"].append(cfg.datasplit_seed)  ##
                    df["Width"].append(width)
                    df["Dropout"].append(cfg.model.dropout_p)
                    if hasattr(cfg.model, "n_layers"):
                        layers = cfg.model.n_layers
                    else:
                        layers = 2
                    df["Layers"].append(layers)
                    df["L2"].append(cfg.optim.weight_decay)
                    if cfg.optim.name == "SGD":
                        optim = f"SGD-{cfg.optim.momentum:.1f}M"
                    else:
                        optim = cfg.optim.name
                    df["Optimizer"].append(optim)
    print(f"{datetime.now()}: Processing finished!")

    df = pd.DataFrame.from_dict(df)
    df.to_parquet(
        Path(__file__).parent / Path("..") / Path("reports") / run_to_fname[runtype]
    )

# %%

