import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
import torch_geometric.data
import torch_geometric.data.storage
import torch_geometric.transforms as T
import torch_geometric.utils
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from omegaconf import DictConfig
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS

import src.models.utils as model_utils
import src.training.utils as train_utils

log = logging.getLogger(__name__)


def train(
    model: torch.nn.Module,
    data: torch_geometric.data.Data,
    idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
):
    model.train()
    out = model(data)
    loss = criterion(out[idx], data.y[idx])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval(
    model: torch.nn.Module,
    data: torch_geometric.data.Data,
    train_idx: Optional[torch.Tensor] = None,
    val_idx: Optional[torch.Tensor] = None,
    test_idx: Optional[torch.Tensor] = None,
    criterion: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    model.eval()
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = {}
    for key, idx in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
        if idx is not None:
            results[f"{key}_acc"] = torch_geometric.utils.metric.accuracy(
                y_pred.view(-1)[idx], data.y[idx]
            )
            if criterion is not None:
                loss = criterion(out[idx], data.y[idx]).item()
                results[f"{key}_loss"] = loss
    return results


def get_dataset(
    name: str,
    root: str,
    transforms: List[Callable] = [],
    pre_transforms: List[Callable] = [],
    public_split: bool = True,
    split_type: str = "num",
    num_train_per_class: int = 20,
    part_val: float = 0.15,
    part_test: float = 0.8,
) -> torch_geometric.data.Dataset:
    """Return a benchmarking dataset. If the dataset does not have data splits, add them. 
    
    To be reproducible, this requires seeding of torch random number generation outside 
    this function! Splits are recreated every access (as transform), so they need to be 
    saved before the seed is changed.

    Args:
        name (str): Name of dataset. Casing is irrelevant.
        root (str): Root directory, where data is stored.
        transforms (List[Callable], optional): torch_geometric transforms. Defaults to empty list.
        pre_transforms (List[Callable], optional): torch_geometric pre_transforms. Defaults to empty list.
        public_split (bool, optional): Whether to use the public split, if available. 
        Otherwise, creates split according to rest of arguments. Defaults to True.
        split_type (str, optional): One of "num", "proportional", "degree". If "proportional", created
        splits will have nodes for every class proportional to their prevalence in the whole 
        dataset. If "num", a fixed number of nodes per class is used (num_train_per_class). 
        If "degree", nodes with highest degrees are used for training, otherwise like "num". 
        Defaults to "num".
        num_train_per_class (int, optional): Number of nodes used for training. Only 
        applies to datasets without pre-specified data splits. Defaults to 20.
        part_val (float, optional): Fraction of dataset used for validation. Only used 
        with proportional split. Defaults to 0.15.
        part_test (float, optional): Fraction of dataset used for testing. Only used with 
        proportional split. Defaults to 0.8.

    Raises:
        ValueError: if unknown dataset name is given.

    Returns:
        torch_geometric.data.Dataset: a dataset with train-val-test split.
    """
    if split_type == "proportional":
        split = T.RandomNodeSplit(
            split="train_rest", num_splits=1, num_test=part_test, num_val=part_val,
        )
    elif split_type == "degree":
        split = DegreeNodeSplit(
            split="test_rest",
            num_splits=1,
            num_train_per_class=num_train_per_class,
            num_val=500,
        )
    elif split_type == "num":
        split = T.RandomNodeSplit(
            split="test_rest",
            num_splits=1,
            num_train_per_class=num_train_per_class,
            num_val=500,
        )
    else:
        raise ValueError(f"Unknown split type: {split_type}")

    no_split_transforms = transforms.copy()
    if not public_split:
        transforms.append(split)

    # Now differentiate between different datasets
    if name.lower() == "wikics":
        if transforms:
            dataset = WikiCS(
                os.path.join(root, "wikics"), transform=T.Compose(transforms)
            )
        else:
            dataset = WikiCS(os.path.join(root, "wikics"))
    elif name.lower() == "arxiv":
        if transforms:
            dataset = PygNodePropPredDataset(
                "ogbn-arxiv", root, transform=T.Compose(transforms)
            )
        else:
            dataset = PygNodePropPredDataset("ogbn-arxiv", root)
    elif name.lower() == "citeseer":
        if transforms:
            dataset = Planetoid(
                root=root,
                name="CiteSeer",
                split="public",
                transform=T.Compose(transforms),
            )
        else:
            dataset = Planetoid(root=root, name="CiteSeer", split="public")
    elif name.lower() == "pubmed":
        if transforms:
            dataset = Planetoid(
                root=root,
                name="Pubmed",
                split="public",
                transform=T.Compose(transforms),
            )
        else:
            dataset = Planetoid(root=root, name="Pubmed", split="public")
    else:
        # dataset is from Coauthor or Amazon -> no predefined train-val-test split
        if transforms:
            transform = T.Compose([*no_split_transforms, split,])
        else:
            transform = T.Compose([split])

        if pre_transforms:
            pre_transform = T.Compose(pre_transforms)
        else:
            pre_transform = None

        if name.lower() == "cs":
            dataset = Coauthor(
                root, name="CS", transform=transform, pre_transform=pre_transform
            )
        elif name.lower() == "physics":
            dataset = Coauthor(
                root, name="Physics", transform=transform, pre_transform=pre_transform
            )
        elif name.lower() == "photo":
            dataset = Amazon(
                root, name="Photo", transform=transform, pre_transform=pre_transform
            )
        elif name.lower() == "computers":
            dataset = Amazon(
                root, name="Computers", transform=transform, pre_transform=pre_transform
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")
    return dataset


def get_idx_split(dataset: torch_geometric.data.Dataset) -> Dict[str, torch.Tensor]:
    data = dataset[0]
    if isinstance(dataset, PygNodePropPredDataset):
        idx: Dict[str, torch.Tensor] = dataset.get_idx_split()  # type:ignore
        # convert each index of variable length with node ids into a boolean vector with fixed length
        for key, tensor in idx.items():
            new_tensor = torch.zeros((data.num_nodes,), dtype=torch.bool)  # type:ignore
            new_tensor[tensor] = True
            idx[key] = new_tensor
    else:
        idx = {
            "train": data.train_mask,  # type: ignore
            "valid": data.val_mask,  # type: ignore
            "test": data.test_mask,  # type: ignore
        }
    # If there are multiple datasplits (mainly WikiCS), then .{train,val,test}_mask has
    # shape [num_nodes, num_splits]. For ease of use, remove all but the first one.
    for mask_name, mask in idx.items():
        assert isinstance(mask, torch.Tensor)
        if mask.ndim > 1:
            log.debug("Discarding surplus %s splits", mask_name)
            idx[mask_name] = mask[:, 0]
    return idx  # type:ignore


def train_node_classifier(
    cfg: DictConfig,
    dataset: torch_geometric.data.Dataset,
    split_idx: Dict[str, torch.Tensor],
    init_seed: int,
    train_seed: int,
) -> Tuple[torch.nn.Module, torch_geometric.data.Data, Dict[str, float]]:
    only_implemented_ogbdataset = "ogbn-arxiv"
    # Reproducibility
    # Seeds are set later for training and initialization individually
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    if isinstance(cfg.cuda, str):
        device = torch.device("cpu")
    else:
        device = torch.device(
            f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu"
        )
    log.info(f"Using device: {device}")

    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    data = dataset[0].to(str(device))
    data.adj_t = data.adj_t.to_symmetric()

    # Build model
    log.info(f"Using model: {cfg.model.name}")
    log.info(f"Initializing model with seed={init_seed}")
    pl.seed_everything(init_seed)
    model = model_utils.get_model(dataset, cfg.model).to(device)
    log.info(
        f"Model has {model_utils.count_parameters(model)} parameters "
        f"({model_utils.count_parameters(model, trainable=True)} trainable)."
    )

    # Set up training
    pl.seed_everything(train_seed)
    optimizer = train_utils.get_optimizer(model.parameters(), cfg.optim)
    early_stopper = EarlyStopping(
        cfg.patience,
        verbose=True,
        path=Path(os.getcwd(), "checkpoint.pt"),
        trace_func=log.debug,
    )
    criterion = torch.nn.CrossEntropyLoss()
    n_epochs = cfg.n_epochs
    train_idx = train_idx.to(device)

    start = time.perf_counter()
    for e in range(n_epochs):
        train_loss = train(model, data, train_idx, optimizer, criterion)
        eval_results = eval(
            model, data, train_idx=train_idx, val_idx=valid_idx, criterion=criterion,
        )
        log.info(
            f"time={time.perf_counter() - start:.2f} epoch={e}: "
            f"{train_loss=:.3f}, train_acc={eval_results['train_acc']:.2f}, "
            f"val_loss={eval_results['val_loss']:.3f}, val_acc={eval_results['val_acc']:.2f}"
        )
        early_stopper(eval_results["val_loss"], model)
        if early_stopper.early_stop and cfg.early_stopping:
            log.info(
                "Stopping training early because validation loss has not decreased"
                " after %i epochs",
                early_stopper.patience,
            )
            break

    log.info("Reverting to model with best val loss")
    if Path(early_stopper.path).exists():
        model.load_state_dict(torch.load(early_stopper.path))
    eval_results = eval(
        model,
        data,
        train_idx=train_idx,
        val_idx=valid_idx,
        test_idx=test_idx,
        criterion=criterion,
    )
    log.info(
        f"train_loss={eval_results['train_loss']:.3f}, train_acc={eval_results['train_acc']:.2f}, "
        f"val_loss={eval_results['val_loss']:.3f}, val_acc={eval_results['val_acc']:.2f}, "
        f"test_loss={eval_results['test_loss']:.3f}, test_acc={eval_results['test_acc']:.2f}"
    )

    return model, data, eval_results


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Credit: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class DegreeNodeSplit(T.RandomNodeSplit):
    def __init__(
        self,
        split: str = "train_rest",
        num_splits: int = 1,
        num_train_per_class: int = 20,
        num_val: Union[int, float] = 500,
        num_test: Union[int, float] = 1000,
        key: Optional[str] = "y",
    ):
        super().__init__(split, num_splits, num_train_per_class, num_val, num_test, key)

    def _split(
        self, store: torch_geometric.data.storage.NodeStorage
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Modification of torch_geometric.transforms.RandomNodeSplit to split based on 
        highest degree nodes.
        """
        num_nodes = store.num_nodes
        assert isinstance(num_nodes, int)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = round(num_nodes * self.num_val)
        else:
            num_val = self.num_val

        if isinstance(self.num_test, float):
            num_test = round(num_nodes * self.num_test)
        else:
            num_test = self.num_test

        # Calculate the degrees here so we can split based on them.
        # Graph is treated as undirected.
        if not hasattr(store, "adj_t"):
            raise ValueError(
                "NodeStorage does not have adj_t attribute. ToSparseTensor has to be called first."
            )
        degrees = store.adj_t.to_symmetric().sum(dim=0)

        if self.split == "train_rest":
            raise NotImplementedError(f"train_rest split not implemented")
            # perm = torch.randperm(num_nodes)
            # val_mask[perm[:num_val]] = True
            # test_mask[perm[num_val : num_val + num_test]] = True
            # train_mask[perm[num_val + num_test :]] = True
        else:
            assert isinstance(self.key, str)
            y = getattr(store, self.key)
            num_classes = int(y.max().item()) + 1
            for c in range(num_classes):
                # [1, 4, 5, ...]  die indizes
                idx = (y == c).nonzero(as_tuple=False).view(-1)
                degree_ranks = torch.argsort(degrees[idx], descending=True)
                degrees_sorted = degrees[idx][degree_ranks]
                idx = idx[degree_ranks]
                idx = idx[: self.num_train_per_class]
                degrees_sorted = degrees[idx]
                train_mask[idx] = True

            remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            val_mask[remaining[:num_val]] = True

            if self.split == "test_rest":
                test_mask[remaining[num_val:]] = True
            elif self.split == "random":
                raise NotImplementedError(f"random split not implemented")
                # test_mask[remaining[num_val : num_val + num_test]] = True

        return train_mask, val_mask, test_mask
