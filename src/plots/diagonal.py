import logging
import os
import pathlib
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)


def save_cka_diagonal(ckas: np.ndarray, savepath: Union[str, pathlib.Path]) -> None:
    if ckas.ndim != 3:
        raise ValueError(
            f"The input should be of shape (n_pairs, n_layers1, n_layers2) but "
            f"has {ckas.ndim} dimensions."
        )

    # we only need the diagonal elements to study per layer (intra-layer?) stability
    diag_idx = np.diag_indices_from(ckas[0])

    # Create a pandas dataframe from which we can easily plot
    log.info("Building plot of diagonal CKA elements.")
    log.info(
        "Assuming architecture of form: Linear -> (ConvBlock -> Residual) x N -> Linear"
    )
    df = []
    for arr in ckas:
        identical_layer_ckas = arr[diag_idx[0], diag_idx[1]]
        for i, layer_cka in enumerate(identical_layer_ckas):
            # TODO: this assumes a specific architecture,
            # should be replaced by a function that maps layer idx to layer name
            if i == 0:
                layer_type = "Linear"
            elif (i % 2) == 1:
                layer_type = "Conv"
            else:
                layer_type = "Residual"
            df.append((i, layer_cka, layer_type))
    df = pd.DataFrame.from_records(df, columns=["Layer", "CKA", "Layer Type"])

    g = sns.catplot(
        data=df,
        x="Layer",
        y="CKA",
        kind="point",
        hue="Layer Type",
        join=True,
        # sharey=False,
    )

    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
        log.debug("Created %s directory", os.path.dirname(savepath))
    g.savefig(savepath)
    plt.close(plt.gcf())
