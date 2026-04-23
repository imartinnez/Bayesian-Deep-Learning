# @author: Inigo Martinez Jimenez
# Public API for the thesis visualization layer. Importing this package
# installs the matplotlib theme and exposes the helpers every caller needs.

from src.visualization import palette
from src.visualization import theme
from src.visualization import io as fig_io
from src.visualization import primitives
from src.visualization import annotate
from src.visualization import events
from src.visualization import data_loaders
from src.visualization import baselines
from src.visualization import predictions
from src.visualization import figures
from src.visualization import animations
from src.visualization import runner


__all__ = [
    "palette",
    "theme",
    "fig_io",
    "primitives",
    "annotate",
    "events",
    "data_loaders",
    "baselines",
    "predictions",
    "figures",
    "animations",
    "runner",
]
