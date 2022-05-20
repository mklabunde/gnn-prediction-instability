import numpy as np
import pytest
import torch

from src.similarity.predictions import (
    check_consecutive,
    classification_node_distr,
    classification_prevalence,
    fraction_stable_predictions,
    pairwise_conditioned_instability,
    pairwise_l1loss,
    pairwise_sym_kldiv,
)


@pytest.mark.parametrize(
    ("preds", "expected"),
    [
        ([torch.arange(0, 3), torch.arange(1, 4)], 0),
        ([torch.arange(0, 3), torch.arange(0, 3)], 1),
        ([torch.arange(0, 3), torch.tensor([0, 1, -1])], 2 / 3),
        (
            [
                torch.tensor([0, 1, 2]),
                torch.tensor([0, 1, 3]),
                torch.tensor([-1, 1, 3]),
            ],
            1 / 3,
        ),
    ],
)
def test_fraction_stable_prediction(preds, expected):
    assert fraction_stable_predictions(preds) == expected


@pytest.mark.parametrize(
    ("preds", "num_classes", "expected"),
    [
        ([torch.zeros((3,))], 1, {0: 3.0}),
        ([torch.zeros((3,)), torch.ones((3,))], 2, {1: 1.5, 0: 1.5}),
    ],
)
def test_classification_prevalence_mean(preds, num_classes, expected):
    output = classification_prevalence(preds, num_classes)
    output = {k: v[0] for k, v in output.items()}
    output == expected


@pytest.mark.parametrize(
    ("preds", "num_classes_gt", "expected"),
    [
        ([torch.arange(3), torch.arange(3)], 3, np.eye(3, 3) * 2),
        (
            [torch.arange(2), torch.arange(2), torch.ones(2, dtype=torch.long)],
            2,
            np.eye(2, 2) * 2 + np.array([[0, 1], [0, 1]]),
        ),
    ],
)
def test_classification_node_distr(preds, num_classes_gt, expected):
    np.testing.assert_array_equal(
        classification_node_distr(preds, num_classes_gt), expected
    )


@pytest.mark.parametrize(
    ("lst", "expected"),
    [([0, 1, 2, 3], True), ([5, 6, 7, 8], True), ([1, 3, 4], False)],
)
def test_check_consecutive(lst, expected):
    assert check_consecutive(lst) == expected


def test_pairwise_sym_kldiv():
    model_outputs = np.array([[2, 8], [3, 4]], dtype=np.float32)
    diffs = pairwise_sym_kldiv(model_outputs)
    assert isinstance(diffs, np.ndarray)
    assert diffs.shape == (1,)


def test_pairwise_l1loss():
    probas = np.array([[0.3, 0.7], [0.4, 0.6]], dtype=np.float64)
    diffs = pairwise_l1loss(probas)
    assert isinstance(diffs, np.ndarray)
    assert diffs.shape == (1,)
    np.testing.assert_almost_equal(diffs[0], 0.1, decimal=7)


def test_pairwise_conditioned_instability():
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 1, 1])
    gt = torch.tensor([0, 1, 1])
    true_diffs, false_diffs = pairwise_conditioned_instability(
        np.stack([p1, p2], axis=0), gt
    )

    np.testing.assert_array_equal(true_diffs, [0.0, 1 / 3])
    np.testing.assert_array_equal(false_diffs, [1.0, 0.0])
