# @author: Inigo Martinez Jimenez
# Central color palette for all thesis figures. One source of truth so that
# any figure or animation shares the same visual identity.

from types import SimpleNamespace


# Background tones: a deep, slightly cool near-black that makes warm highlights
# pop. The figure face is darker than the axes face so inner plotting areas
# feel inset and spacious.
BACKGROUND_FIGURE = "#06070A"
BACKGROUND_AXES = "#0C0E13"
BACKGROUND_PANEL = "#101218"

# Primary warm accent used for the "headline" series of every plot (realized
# volatility, predictive mean, main model of interest...).
ACCENT_PRIMARY = "#FF7A3D"
ACCENT_PRIMARY_SOFT = "#F6A960"
ACCENT_PRIMARY_DEEP = "#CC4E1D"
ACCENT_PRIMARY_GLOW = "#FFB074"

# Cool secondary accents for comparison series and secondary signals. We
# deliberately pick muted, desaturated cool tones so they never fight with the
# primary orange.
ACCENT_COOL = "#5FA8FF"
ACCENT_TEAL = "#4DD0C8"
ACCENT_LAVENDER = "#B09AE0"

# Neutral tones for grids, text, and baselines. Grid is intentionally close to
# the axes face so it supports the eye without creating noise.
TEXT_PRIMARY = "#ECEEF3"
TEXT_SECONDARY = "#A8AEBC"
TEXT_MUTED = "#6B7181"
GRID_COLOR = "#1E2028"
SPINE_COLOR = "#2A2D38"
REFERENCE_COLOR = "#E7E9F0"

# Regime palette: low/medium/high volatility. Low is a calm teal, medium a
# muted violet, high a bright warm red so the three regimes read instantly.
REGIME_LOW = "#4DD0C8"
REGIME_MED = "#B09AE0"
REGIME_HIGH = "#FF6B5E"

# Uncertainty fills: we rely on two hues for epistemic and aleatoric to make
# the decomposition readable at a glance in all stacked plots.
UNCERTAINTY_EPISTEMIC = "#5FA8FF"
UNCERTAINTY_ALEATORIC = "#FF7A3D"
UNCERTAINTY_TOTAL = "#F6A960"

# Model palette used whenever several models share the same axes. The Bayesian
# MLP is the thesis's main model so it inherits the warm headline accent. Other
# models orbit around it with cool / neutral hues.
MODEL_COLORS = {
    "Realized Volatility": "#ECEEF3",
    "Mean Baseline": "#6B7181",
    "Historical Volatility": "#A8AEBC",
    "EWMA (RiskMetrics)": "#4DD0C8",
    "HAR-RV": "#B09AE0",
    "LSTM": "#5FA8FF",
    "Bayesian LSTM": "#F6A960",
    "MLP": "#4DD0C8",
    "Bayesian MLP": "#FF7A3D",
    "LSTM Classifier": "#8FB8E8",
    "Bayesian LSTM Classifier": "#FFB074",
}

# Named bundle so callers can access everything through one object without
# cluttering imports. Example: from .palette import COLORS; COLORS.accent_primary
COLORS = SimpleNamespace(
    background_figure=BACKGROUND_FIGURE,
    background_axes=BACKGROUND_AXES,
    background_panel=BACKGROUND_PANEL,
    accent_primary=ACCENT_PRIMARY,
    accent_primary_soft=ACCENT_PRIMARY_SOFT,
    accent_primary_deep=ACCENT_PRIMARY_DEEP,
    accent_primary_glow=ACCENT_PRIMARY_GLOW,
    accent_cool=ACCENT_COOL,
    accent_teal=ACCENT_TEAL,
    accent_lavender=ACCENT_LAVENDER,
    text_primary=TEXT_PRIMARY,
    text_secondary=TEXT_SECONDARY,
    text_muted=TEXT_MUTED,
    grid=GRID_COLOR,
    spine=SPINE_COLOR,
    reference=REFERENCE_COLOR,
    regime_low=REGIME_LOW,
    regime_med=REGIME_MED,
    regime_high=REGIME_HIGH,
    epistemic=UNCERTAINTY_EPISTEMIC,
    aleatoric=UNCERTAINTY_ALEATORIC,
    total=UNCERTAINTY_TOTAL,
    model=MODEL_COLORS,
)


def regime_color(regime: str) -> str:
    # Maps a regime label to its canonical color. Accepts short codes or full
    # names so both the numeric arrays and the human-readable labels work.
    key = str(regime).strip().upper()
    if key in ("LOW", "LOW VOLATILITY", "L"):
        return REGIME_LOW
    if key in ("MED", "MEDIUM", "MEDIUM VOLATILITY", "M"):
        return REGIME_MED
    if key in ("HIGH", "HIGH VOLATILITY", "H"):
        return REGIME_HIGH
    return TEXT_MUTED


def regime_display_name(regime: str) -> str:
    # Returns the thesis-friendly capitalized regime label.
    key = str(regime).strip().upper()
    if key in ("LOW", "L"):
        return "Low Volatility"
    if key in ("MED", "MEDIUM", "M"):
        return "Medium Volatility"
    if key in ("HIGH", "H"):
        return "High Volatility"
    return str(regime)
