from tempfile import TemporaryDirectory

import pytest
import torch
import torch_geometric.datasets
import torch_geometric.transforms as T

from src.training.node import DegreeNodeSplit


def test_DegreeNodeSplit():
    with TemporaryDirectory() as tmpdir:
        dataset = torch_geometric.datasets.WikiCS(
            tmpdir,
            transform=T.Compose(
                [
                    T.ToSparseTensor(),
                    DegreeNodeSplit(
                        split="test_rest", num_train_per_class=20, num_val=500
                    ),
                ]
            ),
        )

        data = dataset[0]
        degrees = data.adj_t.to_symmetric().sum(dim=0)  # type:ignore

        for c in torch.unique(data.y):
            c_idx = data.y == c
            c_train_idx = c_idx & data["train_mask"]  # type:ignore
            c_test_idx = c_idx & data["test_mask"]  # type:ignore
            c_val_idx = c_idx & data["val_mask"]  # type:ignore

            assert degrees[c_train_idx].min() >= degrees[c_test_idx].max()
            assert degrees[c_train_idx].min() >= degrees[c_val_idx].max()
