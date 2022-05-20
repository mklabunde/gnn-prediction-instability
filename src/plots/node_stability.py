import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import torch

log = logging.getLogger(__name__)


def save_class_prevalence_plots(
    y_true: torch.Tensor,
    test_idx: torch.Tensor,
    prevalences_path: Union[str, Path],
    savepath: Optional[Union[str, Path]] = None,
    dataset_name: str = "?",
) -> None:
    prevs = pd.read_csv(
        prevalences_path,
        index_col=0,
        names=["subset", "class", "mean", "std"],
        header=0,
    )
    assert isinstance(prevs, pd.DataFrame)
    prevs = prevs.loc[prevs.subset == "test"]

    # Ground truth class prevalence
    # sorted returns uniqs in ascending order -> 0, 1, 2,...
    _, counts = y_true[test_idx].unique(sorted=True, return_counts=True)
    prevs["gt"] = counts.numpy()

    prevs["std_scaled"] = prevs["std"] / prevs["gt"]

    facet_size = 5  # inch (?)
    fig, axes = plt.subplots(1, 3, figsize=(3 * facet_size, facet_size))

    ax = axes[0]
    sns.barplot(x=prevs["class"], y=prevs["mean"], yerr=prevs["std"], ax=ax)
    ax.set_title("Mean class prevalence on test set ($\\pm$ stddev)")
    ax.set_ylabel("Number of nodes predicted")
    ax.set_xlabel("Class")

    ax = axes[1]
    sns.barplot(x=prevs["class"], y=prevs["std_scaled"], ax=ax)
    ax.set_title(
        "Stddev of class prevalence on test set\nas fraction of ground truth prevalence"
    )
    ax.set_ylabel("Stddev / Ground Truth")
    ax.set_xlabel("Class")

    ax = axes[2]
    sorted_idx = np.argsort(prevs["gt"].values)  # type:ignore
    sns.scatterplot(
        x=prevs["gt"].iloc[sorted_idx], y=prevs["std_scaled"].iloc[sorted_idx], ax=ax
    )
    ax.set_title(
        "Stddev of class prevalence on test set\nin relation to ground truth prevalence"
    )
    ax.set_ylabel("Stddev / Ground Truth")
    ax.set_xlabel("Ground Truth Prevalence")
    ax.text(
        0.9,
        0.9,
        f"Dataset: {dataset_name}\nSubset: Test",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    if savepath is not None:
        savefig(fig, savepath, fname="class_prevalence.pdf")
    plt.close()


def save_node_instability_distribution(
    test_idx: torch.Tensor,
    prediction_distr_path: Union[str, Path],
    savepath: Optional[Union[str, Path]] = None,
    dataset_name: str = "?",
) -> None:
    # Load array of shape (n_nodes, n_classes) which contains in cell (i, j) the number
    # of times node i was predicted as j.
    distr = np.load(prediction_distr_path)

    data = pd.DataFrame(
        data=distr.astype(bool).sum(axis=1), columns=["Predicted as N Classes"]
    )
    # we are only interested in the distribution on test data
    data = data[test_idx.numpy()]
    g = sns.catplot(data=data, x="Predicted as N Classes", kind="count")
    ax = g.axes.flat[0]
    ax.text(
        0.9,
        0.9,
        f"Dataset: {dataset_name}\nSubset: Test",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    if savepath is not None:
        savefig(g, savepath, fname="n_diff_class_predicted.pdf")
    plt.close()


def save_pairwise_instability_distribution(
    diffs: np.ndarray, savepath: Optional[Union[str, Path]] = None
):
    if savepath is not None:
        g = sns.displot(x=diffs)
        g.set_xlabels("Pairwise instable predictions")
        savefig(g, savepath, fname="pairwise_instable.jpg")
        plt.close()


def save_scatter_correlation(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    log_msg: str,
    savepath: Optional[Union[str, Path]] = None,
):
    corr = scipy.stats.spearmanr(x, y)
    log.info(log_msg, str(corr))
    g = sns.relplot(y=y, x=x, kind="scatter")
    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)
    if savepath is not None:
        savefig(g, savepath, fname=None)
    plt.close()


def savefig(
    fig: Union[matplotlib.figure.Figure, sns.FacetGrid],
    path: Union[str, Path],
    fname: Optional[str] = None,
):
    if os.path.isdir(path):
        if fname is None:
            raise ValueError(
                "Path to save at is a directory, but no filename is specified"
            )
        fig.savefig(Path(path, fname))
    else:
        fig.savefig(path)

