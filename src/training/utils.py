import logging
import os
from typing import Dict, Iterator, Optional

import torch
from omegaconf.dictconfig import DictConfig

log = logging.getLogger(__name__)


def get_optimizer(
    params: Iterator[torch.nn.Parameter], cfg: DictConfig
) -> torch.optim.Optimizer:
    """Get an optimizer as configured

    Args:
        params (Iterator[Parameter]): model.parameters()
        cfg (DictConfig): config of optimizer

    Raises:
        NotImplementedError: if trying to use optimizer that is not Adam

    Returns:
        torch.optim.Optimizer: configured optimizer
    """
    if cfg.name == "Adam":
        return torch.optim.Adam(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    elif cfg.name == "SGD":
        return torch.optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise NotImplementedError()


def save_split_idx(
    split_idx: Dict[str, torch.Tensor], path: Optional[str] = None
) -> None:
    """Saves a dictionary of tensors to disk.

    Args:
        split_idx (Dict[str, torch.Tensor]): maps split name (train, val, test, etc.)
         to boolean tensor
        path (Optional[str]): path to save tensors to. If not specified, uses os.getcwd.
    """
    if path is None:
        path = os.getcwd()
    log.info("Saving data split in %s", path)
    for key, mask in split_idx.items():
        torch.save(mask, os.path.join(path, f"{key}_mask.pt"))
