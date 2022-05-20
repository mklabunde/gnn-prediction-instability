import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import src.plots.node_stability


def check_consecutive(lst: List[Any]) -> bool:
    return all(np.diff(lst) == 1)


def fraction_stable_predictions(preds: List[torch.Tensor]) -> float:
    """Compute the fraction of predictions that are identical for a set of prediction vectors

    Args:
        preds (List[torch.Tensor]): a list of predictions

    Returns:
        float: fraction of stable predictions
    """
    all_equal = torch.ones_like(preds[0], dtype=torch.bool)
    for x, y in itertools.combinations(preds, 2):
        pair_equal = torch.eq(x, y)
        all_equal = all_equal & pair_equal
    return all_equal.sum().item() / len(all_equal)


def classification_prevalence(
    preds: List[torch.Tensor], num_classes_gt: int
) -> Dict[Any, Tuple[float, float]]:
    """Calculate the mean and std of the prevalence of all predicted classes

    Args:
        preds (List[torch.Tensor]): list of prediction vectors
        num_classes_gt (int): number of ground truth classes, have start with 0 index

    Returns:
        Dict[Any, Tuple[float, float]]: maps class to (mean, std)
    """
    d = defaultdict(list)

    all_uniques = np.arange(num_classes_gt)

    for pred in preds:
        uniq, counts = torch.unique(pred, sorted=True, return_counts=True)
        uniq, counts = uniq.numpy(), counts.numpy()
        # Append the counts and keep track if all possible values are seen
        # Some models might be really bad/ignore minorities
        # This is necessary so mean and std are computed correctly
        visited = {u: False for u in all_uniques}
        for u, count in zip(uniq, counts):
            d[u].append(count)
            visited[u] = True

        for u, was_seen in visited.items():
            if not was_seen:
                d[u].append(0)

    return {u: (np.mean(d[u]), np.std(d[u])) for u in all_uniques}


def classification_node_distr(
    preds: List[torch.Tensor], num_classes_gt: int
) -> np.ndarray:
    """Aggregate predictions into a matrix that counts how often a node is predicted as 
    specific class

    Args:
        preds (List[torch.Tensor]): list of prediction vectors for classification (ints)
        num_classes_gt (int): number of ground truth classes

    Returns:
        np.ndarray: shape num_nodes x num_classes, contains counts
    """
    hist = np.zeros((len(preds[0]), num_classes_gt))  # N x n_classes
    for pred in preds:
        for i, node_pred in enumerate(pred.numpy()):
            hist[i, node_pred] += 1
    return hist


def pairwise_instability(
    preds: np.ndarray, figurepath: Optional[Path] = None
) -> np.ndarray:
    diffs = []
    for i, j in itertools.combinations(np.arange(preds.shape[0]), 2):
        x, y = preds[i], preds[j]
        diff = np.not_equal(x, y).sum() / len(x)
        diffs.append(diff)
    diffs = np.asarray(diffs)
    src.plots.node_stability.save_pairwise_instability_distribution(
        diffs, savepath=figurepath
    )
    return diffs


def max_pi(acc1: float, acc2: float) -> float:
    return min(1.0, 2.0 - acc1 - acc2)


def min_pi(acc1: float, acc2: float) -> float:
    return abs(acc1 - acc2)


def normalized_pairwise_instability(
    preds: np.ndarray, accs: np.ndarray, figurepath: Optional[Path] = None
) -> np.ndarray:
    """Calculate the min-max normalized pairwise instability.

    Args:
        preds (np.ndarray): Predictions of shape (n_models, n_nodes)
        accs (np.ndarray): Model accuracies of shape (n_models)
        figurepath (Optional[Path]): Path to save a plot of the distribution to. Defaults to None.
    """
    assert (
        preds.shape[0] == accs.shape[0]
    ), f"preds and accs must have same size of first dimension: {preds.shape[0]=}, {accs.shape[0]=}"

    diffs = []
    for i, j in itertools.combinations(np.arange(preds.shape[0]), 2):
        x, y = preds[i], preds[j]
        diff = np.not_equal(x, y).sum() / len(x)

        acc1, acc2 = accs[i], accs[j]
        lower = min_pi(acc1, acc2)
        upper = max_pi(acc1, acc2)
        diff = (diff - lower) / (upper - lower)

        diffs.append(diff)
    diffs = np.asarray(diffs)

    if figurepath is not None and figurepath.is_dir():
        figurepath = Path(figurepath, "rel_pi.jpg")

    src.plots.node_stability.save_pairwise_instability_distribution(
        diffs, savepath=figurepath
    )
    return diffs


def pairwise_sym_kldiv(
    outputs: np.ndarray, figurepath: Optional[Path] = None
) -> np.ndarray:
    """Calculate the symmetric Kullback-Leibler Divergence between pairs of models.

    Args:
        outputs (np.ndarray): raw logit outputs
        figurepath (Optional[Path], optional): Path to distribution plot. Defaults to None.

    Returns:
        np.ndarray: symmetric KLDivs for all pairs of outputs rows
    """
    kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    preds = outputs

    diffs = []
    for i, j in itertools.combinations(np.arange(preds.shape[0]), 2):
        x, y = torch.from_numpy(preds[i]), torch.from_numpy(preds[j])
        x, y = F.log_softmax(x).unsqueeze(0), F.log_softmax(y).unsqueeze(0)
        diff = kl(x, y) + kl(y, x)
        diffs.append(diff.item())
    diffs = np.asarray(diffs)

    if figurepath is not None and figurepath.is_dir():
        figurepath = Path(figurepath, "p_symKLD.jpg")

    src.plots.node_stability.save_pairwise_instability_distribution(
        diffs, savepath=figurepath
    )
    return diffs


def pairwise_l1loss(
    probas: np.ndarray, figurepath: Optional[Path] = None
) -> np.ndarray:
    loss = torch.nn.L1Loss(reduction="mean")  # MAE
    preds = probas

    diffs = []
    for i, j in itertools.combinations(np.arange(preds.shape[0]), 2):
        x, y = torch.from_numpy(preds[i]), torch.from_numpy(preds[j])
        diff = loss(x, y).item()
        diffs.append(diff)
    diffs = np.asarray(diffs)

    if figurepath is not None and figurepath.is_dir():
        figurepath = Path(figurepath, "p_L1.jpg")

    src.plots.node_stability.save_pairwise_instability_distribution(
        diffs, savepath=figurepath
    )
    return diffs


def pairwise_conditioned_instability(
    preds: np.ndarray, gt_class: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    true_diffs = []
    false_diffs = []
    preds = torch.from_numpy(preds)  # type:ignore

    for i, j in itertools.combinations(torch.arange(preds.size(0)), 2):
        x, y = preds[i], preds[j]
        true_diff = torch.not_equal(
            x[x == gt_class], y[x == gt_class]
        ).sum().item() / max(
            1, (x == gt_class).sum().item()
        )  # avoid division by zero
        false_diff = torch.not_equal(
            x[x != gt_class], y[x != gt_class]
        ).sum().item() / max(1, (x != gt_class).sum().item())
        true_diffs.append(true_diff)
        false_diffs.append(false_diff)

        # Do the operations again but with switched x and y.
        # This way we have the conditioned churn from the perspective of both i and j
        x, y = preds[j], preds[i]
        true_diff = torch.not_equal(
            x[x == gt_class], y[x == gt_class]
        ).sum().item() / max(1, (x == gt_class).sum().item())
        false_diff = torch.not_equal(
            x[x != gt_class], y[x != gt_class]
        ).sum().item() / max(1, (x != gt_class).sum().item())
        true_diffs.append(true_diff)
        false_diffs.append(false_diff)

    true_diffs = np.asarray(true_diffs)
    false_diffs = np.asarray(false_diffs)
    return true_diffs, false_diffs
