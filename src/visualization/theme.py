# @author: Inigo Martinez Jimenez
# Centralized theme for every thesis figure. Applying this module once at the
# start of a plotting session guarantees consistent dark backgrounds, warm
# highlights, spacing, typography, and export defaults across the entire
# visualization layer.

from contextlib import contextmanager
import matplotlib as mpl
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from src.visualization import palette as P


def _patch_weight_dict() -> None:
    # matplotlib's SVG backend indexes ``fm.weight_dict`` by the current text
    # weight. That dict ships with string keys only ("normal", "bold", ...),
    # so any Text object whose weight ended up as an integer (400, 500, ...)
    # — which happens when third-party code or animation internals touch a
    # FontProperties — crashes export with ``KeyError: 400``. We inject the
    # numeric aliases idempotently so downstream code stays safe.
    numeric_aliases = {
        100: 100, 200: 200, 300: 300, 400: 400, 500: 500,
        600: 600, 700: 700, 800: 800, 900: 900,
    }
    for key, value in numeric_aliases.items():
        _fm.weight_dict.setdefault(key, value)


# A single font stack so all thesis figures share the same weight and feel.
# We fall back to generic sans-serif on systems where DejaVu Sans is missing.
FONT_STACK = ["Inter", "SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]


# Canonical figure sizes expressed in inches at 300 DPI. "wide" is for
# time-series style plots, "standard" for square-ish report figures,
# "compact" for small panels, and "hero" for the top-level report visuals.
FIGURE_SIZES = {
    "thumbnail": (4.5, 3.0),
    "compact": (6.0, 3.6),
    "standard": (7.5, 4.5),
    "wide": (12.0, 4.8),
    "wide_tall": (12.0, 6.5),
    "panel": (9.0, 5.5),
    "square": (6.0, 6.0),
    "hero": (13.5, 7.0),
    "tall": (7.5, 9.5),
}


def thesis_cmap() -> LinearSegmentedColormap:
    # Dark-to-warm colormap used for density plots, heatmaps, and kde fills
    # so every gradient-based visual shares the same identity as the line plots.
    stops = [
        (0.00, "#0A0C12"),
        (0.15, "#1A1525"),
        (0.35, "#4A1E1C"),
        (0.55, "#8E3413"),
        (0.75, P.ACCENT_PRIMARY_DEEP),
        (0.90, P.ACCENT_PRIMARY),
        (1.00, P.ACCENT_PRIMARY_GLOW),
    ]
    return LinearSegmentedColormap.from_list("thesis_warm", stops, N=512)


def regime_cmap() -> ListedColormap:
    # Three-color categorical colormap for regime segmentation fills and
    # segmentation rasters. The ordering matches LOW / MED / HIGH.
    return ListedColormap([P.REGIME_LOW, P.REGIME_MED, P.REGIME_HIGH], name="thesis_regimes")


def _base_rc() -> dict:
    # We collect every rcParam in one dictionary so the theme is fully
    # declarative. Any new plot automatically inherits this style without
    # needing per-axis tweaks.
    return {
        # Figure
        "figure.facecolor": P.BACKGROUND_FIGURE,
        "figure.edgecolor": P.BACKGROUND_FIGURE,
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.facecolor": P.BACKGROUND_FIGURE,
        "savefig.edgecolor": P.BACKGROUND_FIGURE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "figure.constrained_layout.use": False,
        "figure.autolayout": False,

        # Axes
        "axes.facecolor": P.BACKGROUND_AXES,
        "axes.edgecolor": P.SPINE_COLOR,
        "axes.linewidth": 0.9,
        "axes.labelcolor": P.TEXT_SECONDARY,
        "axes.titlecolor": P.TEXT_PRIMARY,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlesize": 13.0,
        "axes.labelsize": 10.5,
        "axes.labelweight": "medium",
        "axes.titlepad": 14.0,
        "axes.labelpad": 8.0,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.grid.axis": "both",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "axes.unicode_minus": False,
        "axes.prop_cycle": mpl.cycler(color=[
            P.ACCENT_PRIMARY, P.ACCENT_COOL, P.ACCENT_TEAL, P.ACCENT_LAVENDER,
            P.ACCENT_PRIMARY_SOFT, P.REGIME_HIGH, P.TEXT_SECONDARY,
        ]),

        # Grid
        "grid.color": P.GRID_COLOR,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,

        # Ticks
        "xtick.color": P.TEXT_SECONDARY,
        "ytick.color": P.TEXT_SECONDARY,
        "xtick.labelcolor": P.TEXT_SECONDARY,
        "ytick.labelcolor": P.TEXT_SECONDARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.82,
        "legend.facecolor": P.BACKGROUND_PANEL,
        "legend.edgecolor": P.SPINE_COLOR,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 10.0,
        "legend.labelcolor": P.TEXT_PRIMARY,
        "legend.borderpad": 0.7,
        "legend.handlelength": 2.2,
        "legend.handletextpad": 0.7,

        # Text
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 10.0,
        "text.color": P.TEXT_PRIMARY,

        # Lines
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.antialiased": True,

        # Patches (histograms, fills)
        "patch.edgecolor": P.BACKGROUND_AXES,
        "patch.linewidth": 0.4,
        "patch.antialiased": True,

        # Scatter
        "scatter.marker": "o",
        "scatter.edgecolors": "none",

        # Images / heatmaps
        "image.cmap": "magma",
        "image.interpolation": "nearest",

        # Date formatting: concise international date axis looks cleaner
        # than the default matplotlib renderer.
        "date.autoformatter.year": "%Y",
        "date.autoformatter.month": "%Y-%m",
        "date.autoformatter.day": "%Y-%m-%d",

        # Output formats
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def apply_theme() -> None:
    # Install the thesis theme into matplotlib's global rcParams. This is a
    # global side effect but it is exactly what we want: every figure, no
    # matter which module draws it, must look identical.
    _patch_weight_dict()
    rcParams.update(_base_rc())

    # Register the warm colormap under a short name so downstream code can
    # simply pass ``cmap="thesis_warm"`` without re-importing the helper.
    try:
        mpl.colormaps.register(thesis_cmap(), name="thesis_warm")
    except ValueError:
        pass

    try:
        mpl.colormaps.register(regime_cmap(), name="thesis_regimes")
    except ValueError:
        pass


@contextmanager
def thesis_style():
    # Provide a context manager for scripts that want a localized theme.
    # It temporarily pushes the theme and restores the caller's rcParams
    # at exit so we do not pollute long-running interactive sessions.
    original = dict(rcParams)
    try:
        apply_theme()
        yield
    finally:
        rcParams.update(original)


def style_axes(ax, *, grid: bool = True, spines: tuple[str, ...] = ("left", "bottom")) -> None:
    # Helper to apply the thesis look to an axis that may have been created
    # outside of our normal pipeline (e.g. inside third-party wrappers).
    ax.set_facecolor(P.BACKGROUND_AXES)

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in spines)
        ax.spines[side].set_color(P.SPINE_COLOR)
        ax.spines[side].set_linewidth(0.9)

    ax.tick_params(colors=P.TEXT_SECONDARY, labelcolor=P.TEXT_SECONDARY)
    ax.xaxis.label.set_color(P.TEXT_SECONDARY)
    ax.yaxis.label.set_color(P.TEXT_SECONDARY)
    ax.title.set_color(P.TEXT_PRIMARY)
    ax.title.set_fontweight("semibold")

    if grid:
        ax.grid(True, color=P.GRID_COLOR, linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def add_title_block(fig, title: str, subtitle: str | None = None, *, y: float = 0.98) -> None:
    # A shared helper that places a headline/subtitle pair above the plot area.
    # The subtitle is rendered in a muted tone to create clear visual hierarchy.
    fig.suptitle(title, x=0.05, y=y, ha="left", va="top",
                 color=P.TEXT_PRIMARY, fontsize=15, fontweight="bold")
    if subtitle:
        fig.text(0.05, y - 0.045, subtitle, ha="left", va="top",
                 color=P.TEXT_SECONDARY, fontsize=10.5, fontweight="normal")


def add_footer(fig, text: str, *, y: float = 0.01) -> None:
    # Adds a small footer with source / method reference. Useful in hero
    # figures meant for the written thesis or defense slides.
    fig.text(0.5, y, text, ha="center", va="bottom",
             color=P.TEXT_MUTED, fontsize=8.0, alpha=0.8)


def get_figure_size(name: str) -> tuple[float, float]:
    # Small helper to keep figure sizing consistent across modules.
    if name not in FIGURE_SIZES:
        raise KeyError(f"Unknown figure size '{name}'. Known: {list(FIGURE_SIZES)}")
    return FIGURE_SIZES[name]


def make_figure(size: str = "wide", **kwargs):
    # Convenience figure factory that wraps plt.figure with the thesis defaults.
    return plt.figure(figsize=get_figure_size(size), **kwargs)


# Apply theme once on import so downstream scripts do not need to remember to
# call apply_theme() explicitly. The apply call is idempotent.
apply_theme()
