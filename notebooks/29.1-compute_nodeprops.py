# %% [markdown]
# # Correlating Node Properties with Subgroup Prediction Disagreement

# %%
import itertools
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch_geometric.transforms as T
import torch_geometric.utils
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.utils import get_model
from src.similarity.predictions import (
    normalized_pairwise_instability,
    pairwise_conditioned_instability,
    pairwise_instability,
    pairwise_l1loss,
)
from src.training.node import get_dataset

RUNDIR = (
    Path(__file__).parent / "../multirun/2022-03-25/12-13-33"
)  # CHANGE: path to baseline run
CACHEDIR = Path(__file__).parent / "cache"
device = (
    torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
)  # CHANGE: cuda:?

# %%
def model_outputs(dataset, cfg, experiment):
    model = get_model(dataset, cfg.model)
    test_idx = torch.load(experiment / "test_mask.pt", map_location="cpu")
    weights_paths = [
        (experiment / str(cfg.seed + i) / "checkpoint.pt") for i in range(cfg.n_repeat)
    ]

    outputs = []
    with torch.no_grad():
        for path in weights_paths:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()

            for p in model.parameters():
                p.requires_grad = False

            model.to(device)
            data = dataset[0].to(str(device))
            data.adj_t = data.adj_t.to_symmetric()
            data.to(device)  # type:ignore
            test_idx.to(device)

            output = model(data)[test_idx]
            outputs.append(output)
    return outputs


def node_properties(dataset):
    g = torch_geometric.utils.to_networkx(dataset[0].cpu())
    g = g.to_undirected()

    prs = np.asarray([deg for node, deg in nx.pagerank(g).items()])
    classes = dataset[0].y.numpy()
    degrees = np.asarray([deg for node, deg in g.degree()])  # type:ignore
    ccs = np.asarray([cc for node, cc in nx.clustering(g).items()])  # type:ignore
    g.remove_edges_from(nx.selfloop_edges(g))  # type:ignore
    kcores = np.asarray([cc for node, cc in nx.core_number(g).items()])  # type:ignore

    return {
        "class": classes,
        "pagerank": prs,
        "degree": degrees,
        "clustering": ccs,
        "kcore": kcores,
    }


def add_records(idx, test_idx, preds, outputs, dataset, cfg, binidx, binval):
    accs = [
        torch_geometric.utils.metric.accuracy(
            torch.from_numpy(pred).view(-1)[idx], dataset[0].y[test_idx][idx]
        )
        for pred in preds
    ]
    pis = pairwise_instability(preds[:, idx])
    norm_pis = normalized_pairwise_instability(preds[:, idx], np.asarray(accs))
    true_pis, false_pis = pairwise_conditioned_instability(
        preds[:, idx], dataset[0].y[test_idx][idx]
    )
    maes = pairwise_l1loss(
        torch.softmax(torch.from_numpy(outputs), dim=-1).numpy()[:, idx]
    )

    for vals, metric_name in zip(
        [pis, norm_pis, true_pis, false_pis, maes],
        ["PI", "NormPI", "True PI", "False PI", "MAE"],
    ):
        for val, (i, j) in zip(
            vals, itertools.combinations(np.arange(cfg.n_repeat), 2)
        ):
            records.append(
                (
                    cfg.dataset.name,
                    cfg.model.name,
                    val,
                    prop,
                    binidx,
                    str(binval),
                    idx.sum().item(),
                    metric_name,
                    cfg.seed + i,
                    cfg.seed + j,
                    accs[i],
                    accs[j],
                )
            )


def recursive_qcut(vals, n_groups):
    if n_groups <= 0:
        return None
    try:
        qc = pd.qcut(vals, n_groups)
    except ValueError as e:
        print(f"{e}. Attempting {n_groups - 1} groups.")
        qc = recursive_qcut(vals, n_groups - 1)
    return qc


# %% [markdown]
# ## Precomputing all required information
#
# Results are cached, so running this a second time should be much faster.

# %%
records = []
columns = [
    "Dataset",
    "Model",
    "Value",
    "Property",
    "Bin",
    "Bin_val",
    "Bin_size",
    "Metric",
    "InitSeed1",
    "InitSeed2",
    "SubgroupAcc1",
    "SubgroupAcc2",
]
n_groups = 7
overwrite = True

for experiment in tqdm(list(filter(lambda p: p.is_dir(), RUNDIR.iterdir()))):
    cfg = OmegaConf.load(experiment / ".hydra" / "config.yaml")
    output_name = f"test_outputs_{cfg.dataset.name}_{cfg.model.name}.npy"
    props_name = f"props_{cfg.dataset.name}.pkl"

    dataset = get_dataset(
        cfg.dataset.name,
        cfg.data_root,
        transforms=[T.ToSparseTensor(remove_edge_index=False)],
    )

    # Reproduce predictions of model
    if CACHEDIR / output_name in CACHEDIR.iterdir() and not overwrite:
        print(f"{output_name} already exists. Loading...")
        outputs = np.load(CACHEDIR / output_name)
    else:
        outputs = model_outputs(dataset, cfg, experiment)
        outputs = torch.stack(outputs, dim=0).cpu().numpy()
        np.save(CACHEDIR / output_name, outputs)
    assert isinstance(outputs, np.ndarray)

    # Convert node properties on the networkx version of the graph
    if CACHEDIR / props_name in CACHEDIR.iterdir() and not overwrite:
        print(f"{props_name} already exists. Loading...")
        with (CACHEDIR / props_name).open("rb") as f:
            props = pickle.load(f)
    else:
        props = node_properties(dataset)
        with (CACHEDIR / props_name).open("wb") as f:
            pickle.dump(props, f)
    assert isinstance(props, dict)

    # Calculate the prediction disagreements for different subgroups of a specific node
    # property
    test_idx = torch.load(experiment / "test_mask.pt", map_location="cpu").numpy()
    preds = outputs.argmax(axis=-1)
    for prop, vals in props.items():
        if prop == "class":
            bins = np.arange(dataset.num_classes)
            for binidx, binval in enumerate(bins):
                idx = dataset[0].y[test_idx] == binval  # type:ignore
                add_records(idx, test_idx, preds, outputs, dataset, cfg, binidx, binval)
        else:
            qc = recursive_qcut(vals[test_idx], n_groups)
            if qc is None:
                continue
            for binidx, binval in enumerate(sorted(qc.unique())):
                idx = qc == binval
                add_records(idx, test_idx, preds, outputs, dataset, cfg, binidx, binval)

df = pd.DataFrame.from_records(records, columns=columns)
if overwrite:
    df.to_parquet(Path(__file__).parent / "../reports/nodeprops.parquet")
df.head()
