import contextlib
from typing import Any, Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

textwidth_pt = 347.12354

HUE_ORDER = ["CiteSeer", "Pubmed", "CS", "Physics", "Computers", "Photo", "WikiCS"]


@contextlib.contextmanager
def update_rcParams(override: Optional[Dict[str, Any]] = None):
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
    if override is not None:
        tex_fonts.update(override)
    with mpl.rc_context(rc=tex_fonts):
        yield


def set_size(
    width: float = textwidth_pt,
    fraction: float = 1.0,
    subplots: Tuple[int, int] = (1, 1),
) -> Tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.
    Credit: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches (width, height)
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
