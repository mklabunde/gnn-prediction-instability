import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def load_representation(path: str, device=torch.device("cpu"), numpy: bool = True):
    with open(path, "rb") as f:
        x = torch.load(f, map_location=device)
    if numpy:
        return x.numpy()
    else:
        return x


def accuracy_layer_identification(sim: np.ndarray) -> float:
    """
    Calculate how often the diagonal similarity scores are the largest ones of a row.
    This corresponds to whether a layer can be reidentified between differently seeded runs
    just by the similarity score.

    Args:
        sim (np.ndarray): square matrix of similarity scores

    Raises:
        ValueError: if sim is not square

    Returns:
        float: fraction of correctly identified layers
    """
    if sim.shape[0] != sim.shape[1]:
        raise ValueError(f"Similarity matrix must be square but has shape {sim.shape}.")
    return (sim.argmax(axis=0) == np.arange(len(sim))).sum() / len(sim)


def find_activation_fnames(
    seed_pair: Tuple[str, str], activations_root: Path
) -> Tuple[List[float], List[float]]:
    """Find and convert filenames of saved activations

    Args:
        seed_pair (Tuple[str, str]): two paths to directories relative to current 
        working directory

    Returns:
        Tuple[List[float], List[float]]: sorted lists of filenames converted to float
    """
    fnames: List[List[float]] = []
    for seed_dir in seed_pair:
        assert isinstance(seed_dir, str)
        fnames.append(
            sorted(
                (
                    float(fname.removesuffix(".pt"))
                    for fname in os.listdir(Path(activations_root, seed_dir))
                    if fname != "checkpoint.pt"
                )
            )
        )
    return tuple(fnames)
