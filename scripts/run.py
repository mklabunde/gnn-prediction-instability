import itertools
import json
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.stats
import seaborn as sns
import torch
import torch.backends.cudnn
import torch.nn.functional as F
import torch_geometric.data
import torch_geometric.transforms as T
import torch_geometric.utils
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import src.plots
import src.similarity
import src.similarity.predictions
from src.training.node import get_dataset, get_idx_split, train_node_classifier

log = logging.getLogger(__name__)


def save_heatmap(
    ids: Tuple[str, str],
    ticklabels: Tuple[List[float], List[float]],
    vals: np.ndarray,
    split: str = "",
) -> None:
    """Save a heatmap of CKA values

    Args:
        ids (Tuple[str, str]): Names for the two models. First one is yaxis, second one xaxis
        ticklabels (Tuple[List[float], List[float]]): (yticks, xticks)
        vals (np.ndarray): CKA scores
        split (str): which data split is plotted. Is added to filename and plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(len(ticklabels[0]) + 2, len(ticklabels[1])))
    hm = sns.heatmap(
        vals[::-1],
        yticklabels=list(reversed(list((map(str, ticklabels[0]))))),
        xticklabels=list(map(str, ticklabels[1])),
        ax=ax,
        annot=True,
        vmin=0,
        vmax=1,
    )
    plt.suptitle(f"Linear CKA between {ids[0]} and {ids[1]} ({split})")
    savepath = os.path.join(
        os.getcwd(), "figures", f"cka_{'_'.join(list(ids) + [split])}.pdf"
    )
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
        log.debug("Created %s directory", os.path.dirname(savepath))
    plt.savefig(savepath)
    plt.close()


def clean_list(lst: List[Any], items_to_remove: List[Any]) -> None:
    for item in items_to_remove:
        if item in lst:
            lst.remove(item)


def main(cfg: DictConfig, activations_root: Optional[Union[str, Path]] = None):
    print(OmegaConf.to_yaml(cfg))
    # Setup additional needed subdirectories
    figures_dir = Path(os.getcwd(), "figures")
    os.makedirs(figures_dir)
    predictions_dir = Path(os.getcwd(), "predictions")
    os.makedirs(predictions_dir)
    cka_dir = Path(os.getcwd(), "cka")
    os.makedirs(cka_dir)
    if activations_root is None:
        activations_root = os.getcwd()
    log.info("Activations root is %s", activations_root)

    # ----------------------------------------------------------------------------------
    # Prepare data
    pl.seed_everything(cfg.datasplit_seed)
    if cfg.proportional_split and cfg.degree_split:
        raise ValueError("Only one of proportional_split and degree_split can be true.")
    if cfg.proportional_split:
        split_type = "proportional"
    elif cfg.degree_split:
        split_type = "degree"
    else:
        split_type = "num"
    dataset = get_dataset(
        name=cfg.dataset.name,
        root=cfg.data_root,
        transforms=[T.ToSparseTensor(remove_edge_index=False)],
        public_split=cfg.public_split,
        split_type=split_type,
        num_train_per_class=cfg.num_train_per_class,
        part_val=cfg.part_val,
        part_test=cfg.part_test,
    )
    split_idx = get_idx_split(dataset)
    pl.seed_everything(cfg.seed)
    for key, mask in split_idx.items():
        torch.save(mask, os.path.join(os.getcwd(), f"{key}_mask.pt"))
    split_idx["full"] = split_idx["train"] | split_idx["valid"] | split_idx["test"]

    predictions: List[torch.Tensor] = []
    outputs_test: List[torch.Tensor] = []
    logits_test: List[torch.Tensor] = []
    evals: List[Dict[str, float]] = []
    seed: int = cfg.seed
    for i in range(cfg.n_repeat):
        current_seed = seed + i
        init_seed = current_seed if not cfg.keep_init_seed_constant else seed
        if cfg.keep_train_seed_constant:
            log.info(
                f"Training model {i + 1} out of {cfg.n_repeat} with seed {seed} "
                f"(init_seed={init_seed})."
            )
            model, data, eval_results = train_node_classifier(
                cfg, dataset, split_idx, init_seed=init_seed, train_seed=seed
            )
        else:
            log.info(
                f"Training model {i + 1} out of {cfg.n_repeat} with seed {current_seed}."
            )
            model, data, eval_results = train_node_classifier(
                cfg, dataset, split_idx, init_seed=init_seed, train_seed=current_seed
            )
        evals.append(eval_results)

        # After training, save the activations of a model
        savedir = os.path.join(activations_root, str(current_seed))
        os.makedirs(savedir, exist_ok=False)
        if cfg.cka.use_masks:  # no need to save activations if they are not used later
            log.info("Saving model activations to %s", savedir)
            with torch.no_grad():
                model.eval()
                assert callable(model.activations)
                act = model.activations(data)
                for key, acts in act.items():
                    savepath = os.path.join(savedir, key + ".pt")
                    torch.save(acts, savepath)
        log.info("Done!")

        log.info("Saving predictions")
        with torch.no_grad():
            output = model(data)
            preds = output.argmax(dim=-1)
            outputs_test.append(
                F.softmax(output, dim=-1).cpu().detach()[split_idx["test"]]
            )
            predictions.append(preds.cpu().detach())
            logits_test.append(output.cpu().detach())

        # Backup the trained weights currently in the working directory as checkpoint.pt
        checkpoint_dir = os.path.join(os.getcwd(), str(current_seed))
        os.makedirs(checkpoint_dir, exist_ok=True)
        if Path(os.getcwd(), "checkpoint.pt").exists():
            shutil.move(
                Path(os.getcwd(), "checkpoint.pt"),
                Path(checkpoint_dir, "checkpoint.pt"),
            )

    log.info("Finished training.")

    # Some logging and simple heuristic to catch models that are far from optimally
    # trained
    suboptimal_models = find_suboptimal_models(evals)
    with open(Path(predictions_dir, "suboptimal_models.pkl"), "wb") as f:
        pickle.dump(suboptimal_models, f)
    with open(Path(predictions_dir, "evals.json"), "w") as f:
        json.dump(evals, f)

    # ----------------------------------------------------------------------------------
    classification_stability_experiments(
        split_idx=split_idx,
        cfg=cfg,
        predictions_dir=predictions_dir,
        figures_dir=figures_dir,
        predictions=predictions,
        outputs_test=outputs_test,
        logits_test=logits_test,
        dataset=dataset,
        evals=evals,
    )

    # ----------------------------------------------------------------------------------
    cka_experiments(
        split_idx=split_idx,
        cfg=cfg,
        figures_dir=figures_dir,
        predictions_dir=predictions_dir,
        cka_dir=cka_dir,
        activations_root=activations_root,
    )


def find_suboptimal_models(
    evals: List[Dict[str, float]], allowed_deviation: int = 2
) -> Dict[str, List[Tuple[int, float]]]:
    results = {}
    for split in ["train", "val", "test"]:
        split_results = [r[f"{split}_acc"] for r in evals]
        log.info(
            "Mean %s accuracy=%.3f, Std=%.3f",
            split,
            np.mean(split_results),
            np.std(split_results),
        )
        suspicious_models: List[Tuple[int, float]] = []
        for i, acc in enumerate(split_results):
            if np.abs(acc - np.mean(split_results)) > allowed_deviation * np.std(
                split_results
            ):
                suspicious_models.append((i, acc))
        log.info(
            "Suspicious models (large deviation from mean acc on %s): %s",
            split,
            str(suspicious_models),
        )
        results[split] = suspicious_models
    return results


def classification_stability_experiments(
    split_idx: Dict[str, torch.Tensor],
    cfg: DictConfig,
    predictions_dir: Path,
    figures_dir: Path,
    predictions: List[torch.Tensor],
    outputs_test: List[torch.Tensor],
    logits_test: List[torch.Tensor],  # type: ignore
    dataset: torch_geometric.data.Dataset,
    evals: List[Dict[str, float]],
):
    log.info("Calculating stability of predictions...")
    preval_df = []
    nodewise_distr_path = Path(predictions_dir, f"nodewise_distr.npy")
    distr = src.similarity.predictions.classification_node_distr(
        predictions, dataset.num_classes  # type:ignore
    )
    np.save(nodewise_distr_path, distr)
    for split_name, idx in split_idx.items():
        filtered_preds = [p[idx] for p in predictions]
        prevalences = src.similarity.predictions.classification_prevalence(
            filtered_preds, dataset.num_classes  # type:ignore
        )
        for key, val in prevalences.items():
            preval_df.append((split_name, key, val[0], val[1]))

        frac_stable = src.similarity.predictions.fraction_stable_predictions(
            filtered_preds
        )
        log.info(
            "Predictions (%s) stable over all models: %.2f", split_name, frac_stable
        )
    prevalences_path = Path(predictions_dir, "prevalences.csv")
    pd.DataFrame.from_records(preval_df).to_csv(prevalences_path)
    src.plots.save_class_prevalence_plots(
        dataset[0].y,  # type:ignore
        split_idx["test"],
        prevalences_path=prevalences_path,
        savepath=figures_dir,
        dataset_name=cfg.dataset.name,
    )
    src.plots.save_node_instability_distribution(
        split_idx["test"],
        prediction_distr_path=nodewise_distr_path,
        savepath=figures_dir,
        dataset_name=cfg.dataset.name,
    )

    # Compare the model output distributions to the prediction distribution
    logits_test: np.ndarray = torch.stack(logits_test, dim=0).numpy()
    probas_test: np.ndarray = torch.stack(outputs_test, dim=0).numpy()
    avg_output_entropy = np.mean(scipy.stats.entropy(probas_test, axis=2), axis=0)
    predictions_entropy = scipy.stats.entropy(distr[split_idx["test"].numpy()], axis=1)
    src.plots.node_stability.save_scatter_correlation(
        predictions_entropy,
        avg_output_entropy,
        "Prediction Entropy",
        "Average Model Output Entropy",
        "Prediction Entropy - Avg Model Output Entropy: %s",
        Path(figures_dir, "entropy_scatter.jpg"),
    )

    # Compare the output and prediction entropy to node properties
    dataset[0].edge_index = torch_geometric.utils.to_undirected(  # type:ignore
        dataset[0].edge_index  # type:ignore
    )
    g = torch_geometric.utils.to_networkx(dataset[0])
    nodes = np.array([i for i, is_test in enumerate(split_idx["test"]) if is_test])
    degrees = np.asarray([d for _, d in nx.degree(g, nbunch=nodes)])
    src.plots.node_stability.save_scatter_correlation(
        degrees,
        predictions_entropy,
        "Degrees",
        "Prediction Entropy",
        "Degree - Prediction Entropy: %s",
        Path(figures_dir, "degree_predentropy.jpg"),
    )
    src.plots.node_stability.save_scatter_correlation(
        degrees,
        avg_output_entropy,
        "Degrees",
        "Avg Model Output Entropy",
        "Degree - Avg Output Entropy: %s",
        Path(figures_dir, "degree_outputentropy.jpg"),
    )

    # Compare models pairwise w.r.t. identical predictions
    pi_distr = src.similarity.predictions.pairwise_instability(
        preds=probas_test.argmax(axis=2), figurepath=figures_dir
    )
    np.save(Path(predictions_dir, "pi_distr.npy"), pi_distr)

    norm_pi_distr = src.similarity.predictions.normalized_pairwise_instability(
        preds=probas_test.argmax(axis=2),
        accs=np.asarray([e["test_acc"] for e in evals]),
        figurepath=figures_dir,
    )
    np.save(Path(predictions_dir, "normpi_distr.npy"), norm_pi_distr)

    symkl_distr = src.similarity.predictions.pairwise_sym_kldiv(
        outputs=logits_test, figurepath=figures_dir,
    )
    np.save(Path(predictions_dir, "symkl_distr.npy"), symkl_distr)

    l1_distr = src.similarity.predictions.pairwise_l1loss(
        probas_test, figurepath=figures_dir
    )
    np.save(Path(predictions_dir, "l1_distr.npy"), l1_distr)

    (
        true_diffs,
        false_diffs,
    ) = src.similarity.predictions.pairwise_conditioned_instability(
        probas_test.argmax(axis=2),
        dataset[0].y[split_idx["test"]].cpu(),  # type:ignore
    )
    np.save(Path(predictions_dir, "true_pi_distr.npy"), true_diffs)
    np.save(Path(predictions_dir, "false_pi_distr.npy"), false_diffs)


def cka_experiments(
    split_idx: Dict[str, torch.Tensor],
    cfg: DictConfig,
    figures_dir: Path,
    predictions_dir: Path,
    cka_dir: Path,
    activations_root: Path,
):
    log.info("Starting pairwise CKA computation.")
    # Jetzt startet die Analyse auf allen paaren der trainierten Modelle
    accuracy_records: List[Tuple[str, str, str, float]] = []
    for split_name, idx in split_idx.items():
        if not split_name in cfg.cka.use_masks:
            log.info("Skipping CKA analysis for %s", split_name)
            continue

        log.info("Starting CKA analysis for %s", split_name)
        idx = idx.numpy()
        pair_length = 2
        # Every model has its own subdirectory, but there are also other output
        # directories, which we have to remove to only iterate over pairs of model dirs
        _, dirnames, _ = next(os.walk(os.getcwd(),))
        log.debug(f"Found dirnames: {dirnames}. Removing output directories.")
        clean_list(
            dirnames,
            items_to_remove=[
                ".hydra",
                figures_dir.parts[-1],
                cka_dir.parts[-1],
                predictions_dir.parts[-1],
            ],
        )

        cka_matrices = src.similarity.cka_matrix(
            dirnames=dirnames,
            idx=idx,
            cka_dir=cka_dir,
            split_name=split_name,
            mode=cfg.cka.mode,
            save_to_disk=cfg.cka.save_to_disk,
            activations_root=activations_root,
        )

        # ------------------------------------------------------------------------------
        log.info("Finished CKA computation. Preparing output for %s.", split_name)

        # Jetzt müssen die Ergbenisse der Paare noch aggregiert werden
        cka_matrices = np.array(cka_matrices)
        np.save(cka_dir / f"ckas_{split_name}.npy", cka_matrices)
        cka_mean = np.mean(cka_matrices, axis=0)
        cka_std = np.std(cka_matrices, axis=0)
        log.debug("CKA matrices shape: %s", (cka_matrices.shape,))
        log.debug("Mean CKA shape: %s", (cka_mean.shape,))

        src.plots.save_cka_diagonal(
            cka_matrices, Path(figures_dir, f"cka_diag_{split_name}.pdf"),
        )

        if cfg.cka.mode == "full":
            for i, seed_pair in enumerate(
                itertools.combinations(sorted(dirnames), pair_length)
            ):
                # Extract the activation filenames again, so we can use them as
                # ticklabels in plots
                fnames = src.similarity.find_activation_fnames(
                    seed_pair, activations_root
                )
                save_heatmap(
                    seed_pair,
                    (fnames[0], fnames[1]),
                    cka_matrices[i],
                    split=split_name,
                )
                accuracy_records.append(
                    (
                        seed_pair[0],
                        seed_pair[1],
                        split_name,
                        src.similarity.accuracy_layer_identification(cka_matrices[i]),
                    )
                )
                if i == 0:
                    # To create heatmaps for mean and std CKA, we need ticklabels, which
                    # we have inside this loop
                    save_heatmap(
                        ("mean", "mean"),
                        (fnames[0], fnames[1]),
                        cka_mean,
                        split=split_name,
                    )
                    accuracy_records.append(
                        (
                            "mean",
                            "mean",
                            split_name,
                            src.similarity.accuracy_layer_identification(cka_mean),
                        )
                    )
                    save_heatmap(
                        ("std", "std"),
                        (fnames[0], fnames[1]),
                        cka_std,
                        split=split_name,
                    )
                    accuracy_records.append(
                        (
                            "std",
                            "std",
                            split_name,
                            src.similarity.accuracy_layer_identification(cka_std),
                        )
                    )
        pd.DataFrame.from_records(
            accuracy_records, columns=[0, 1, "split", "acc"]
        ).to_csv(os.path.join(os.getcwd(), "layer_identification.csv"))


@hydra.main(config_path="../config", config_name="main")
def run(cfg):
    if not cfg.store_activations:
        with tempfile.TemporaryDirectory() as tmpdir:
            main(cfg, activations_root=tmpdir)
    else:
        main(cfg, activations_root=None)


if __name__ == "__main__":
    run()
