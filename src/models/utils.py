from typing import Any, Dict

import torch.nn
import torch_geometric.data
from omegaconf import DictConfig

from src.models import GAT2017, GCN2017


def count_parameters(m: torch.nn.Module, trainable: bool = True) -> int:
    """Count the number of (trainable) parameters of a model

    Args:
        m (torch.nn.Module): model to count parameters of
        trainable (bool, optional): Whether to only count trainable parameters. Defaults to True.

    Returns:
        int: number of parameters
    """
    if trainable:
        return sum(w.numel() for w in m.parameters() if w.requires_grad)
    else:
        return sum(w.numel() for w in m.parameters())


def get_model(
    dataset: torch_geometric.data.Dataset, cfg: DictConfig
) -> torch.nn.Module:
    if cfg.name == "GAT2017":
        return get_GAT2017(dataset, cfg)
    elif cfg.name == "GCN2017":
        return get_GCN2017(dataset, cfg)
    else:
        raise ValueError(f"Unkown model name: {cfg.name}")


def get_GAT2017(
    dataset: torch_geometric.data.Dataset, cfg: DictConfig
) -> torch.nn.Module:
    assert isinstance(dataset.num_classes, int)
    return GAT2017(
        in_dim=dataset.num_features,
        out_dim=dataset.num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout_p=cfg.dropout_p,
        n_heads=cfg.n_heads,
        n_output_heads=cfg.n_output_heads,
        n_layers=cfg.n_layers if hasattr(cfg, "n_layers") else 2,
    )


def get_GCN2017(
    dataset: torch_geometric.data.Dataset, cfg: DictConfig
) -> torch.nn.Module:
    assert isinstance(dataset.num_classes, int)
    return GCN2017(
        in_dim=dataset.num_features,
        out_dim=dataset.num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout_p=cfg.dropout_p,
        n_layers=cfg.n_layers if hasattr(cfg, "n_layers") else 2,
    )
