"""One consistent look for every figure in the notebooks."""

import matplotlib.pyplot as plt
import seaborn as sns

CLASS_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"}
ACCENT = "#264653"


def setup():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 110,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",  # has the sub/superscript glyphs used in labels
    })


def save(name, fig=None):
    """Save the current figure into docs/figures (used by the website and the report)."""
    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "docs" / "figures"
    if out.parent.exists():
        out.mkdir(exist_ok=True)
        (fig or plt.gcf()).savefig(out / f"{name}.png", dpi=130)
