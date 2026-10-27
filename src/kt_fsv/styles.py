"""
Configure the visual properties used by FSV plotting methods.

Notes
-----
PlotStyle is mutable. PRESET_STYLES maps the lowercase names "classic", "dark"
and "publication" to shared PlotStyle instances. Individual views
consume different subsets of the fields; diagnostic grids use fixed panel
styling. A preset does not change Matplotlib's global theme or background.
Numeric and Boolean fields are checked at construction and before plotting;
Matplotlib validates artist-specific settings when a view consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ._internal.value_validation import _validated_display_scalar

# =============================================================================
# Style Constants
# =============================================================================

# Category colors for plotting
CATEGORY_COLORS = {
    "Excellent": "#2ecc71",
    "Very Good": "#27ae60",
    "Good": "#f39c12",
    "Fair": "#e67e22",
    "Poor": "#e74c3c",
    "Very Poor": "#c0392b",
}

# Figure sizes (default 16:9 aspect ratio)
FIG_SIZE = (14, 8)
FIG_SIZE_TALL = (14, 10)

# Grade/Spread chart regions; topic reference: IEEE Std 1597.1-2022 §8.5(b)
# Format: (Spread, Grade) coordinates
GRADE_SPREAD_REGION_EXCELLENT = [
    (0, 0),
    (2.5, 0),
    (2.5, 2.5),
    (0, 2.5),
    (0, 0)
]

GRADE_SPREAD_REGION_GOOD = [
    (2.5, 0),
    (3.5, 0),
    (3.5, 4.5),
    (0, 4.5),
    (0, 2.5),
    (2.5, 2.5),
    (2.5, 0)
]

GRADE_SPREAD_REGION_FAIR = [
    (3.5, 0),
    (4.5, 0),
    (4.5, 5.5),
    (0, 5.5),
    (0, 4.5),
    (3.5, 4.5),
    (3.5, 0)
]

GRADE_SPREAD_REGION_POOR = [
    (4.5, 0),
    (6.5, 0),
    (6.5, 6.5),
    (0, 6.5),
    (0, 5.5),
    (4.5, 5.5),
    (4.5, 0)
]


# =============================================================================
# Plot Style Configuration
# =============================================================================

def _validated_style_number(value: object, name: str) -> float:
    """Check a native plotting scalar without replacing its stored value.

    Parameters
    ----------
    value : object
        Python or NumPy integer/floating scalar, excluding Boolean values.
    name : str
        Style field identified by validation errors.

    Returns
    -------
    float
        Temporary finite nonnegative float64-compatible value.

    Raises
    ------
    TypeError
        If the input has unsupported scalar storage/type or is Boolean.
    ValueError
        If the input is negative, nonfinite or not finitely representable.

    Notes
    -----
    Consumers retain their own range/positivity rules. The original style field
    is not assigned the temporary conversion.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a Python or NumPy integer/floating scalar, not Boolean.")
    return _validated_display_scalar(value, name)


@dataclass
class PlotStyle:
    """
    Configure the colors, dimensions and layout used by supported plot views.

    Parameters
    ----------
    adm_color, fdm_color, gdm_color : str, optional
        Matplotlib colors for applicable measure/response lines. Defaults are
        blue "#1f77b4", green "#2ca02c" and red "#d62728".
    mean_color : str, optional
        Mean/reference-line color, default orange "#ff7f0e".
    grid_color : str, optional
        Applicable grid-line color, default light gray "#d3d3d3".
    line_width : float, optional
        Applicable data-line width in points, default 0.8.
    line_alpha : float, optional
        Applicable line/bar opacity in [0, 1], default 0.7.
    line_style : str, optional
        Measure point-line style, default "-".
    mean_line_style : str, optional
        Measure mean-line style, default "--".
    mean_line_width : float, optional
        Mean/reference-line width in points, default 1.5.
    show_grid : bool, optional
        Enable grids in views that consume this field, default True.
    grid_alpha : float, optional
        Applicable grid opacity in [0, 1], default 0.3.
    grid_style : str, optional
        Applicable grid-line style, default "-".
    grid_width : float, optional
        Measure-view grid width in points, default 0.5.
    marker_size : float, optional
        Measure point-marker size in points, default 6.0.
    marker_style : str, optional
        Measure point-marker identifier, default None for no markers.
    title_size, label_size, legend_size : float, optional
        Applicable font sizes in points, default 14, 12 and 10.
    tight_layout : bool, optional
        Apply tight_layout to newly created figures, default True.
        Does not reposition supplied axes.
    title_weight : str, optional
        Applicable title font weight, default "bold".
    category_colors : dict, optional
        Histogram color mapping for all six canonical category names.
        None copies CATEGORY_COLORS; a supplied mapping is retained directly.

    Returns
    -------
    PlotStyle
        Mutable dataclass with the supplied/default fields.

    Raises
    ------
    ValueError
        If a numeric setting is nonfinite, outside its range or not finitely
        representable for plotting. Widths/sizes must remain positive after
        temporary float64 conversion.
    TypeError
        If a numeric setting is not a Python/NumPy integer or floating scalar,
        is Boolean, or a grid/layout flag is not a Python/NumPy Boolean scalar.

    Notes
    -----
    Numeric fields retain their supplied values and dtypes. Arrays, masked
    wrappers, complex values, numeric strings and other numeric classes such as
    Decimal/Fraction are not accepted; convert them explicitly before use.
    Font sizes may be fractional. Alpha values must be in [0, 1]; all widths,
    marker sizes and font sizes must be positive and finitely representable.
    show_grid and tight_layout require Boolean scalars, without truthiness coercion.

    Mutation does not validate immediately. Each plotting call rechecks all
    numeric/Boolean fields before allocating figures or modifying supplied axes,
    even if that view consumes only a subset of them. Matplotlib validates
    color/marker/line-style/font-weight settings when they are used. Histogram
    mapping keys are checked on use. Those later failures can leave partial
    drawing on caller-owned axes, as described by FSVVisualizer.

    None as a plot's style creates a fresh PlotStyle. Named presets resolve the
    shared mutable objects in PRESET_STYLES; create a new instance for isolated
    customization. The "dark" preset changes line colors, not the background.
    Filters and filtered-component grids keep fixed panel styling. Grade/Spread
    regions/markers are fixed; that view uses this object's fonts/layout.

    Examples
    --------
    >>> from kt_fsv.styles import PlotStyle
    >>> custom = PlotStyle(line_width=1.2, show_grid=False)
    >>> custom.line_width
    1.2
    >>> len(custom.category_colors)
    6
    """
    # Colors
    adm_color: str = "#1f77b4"   # Blue
    fdm_color: str = "#2ca02c"   # Green
    gdm_color: str = "#d62728"   # Red/Magenta
    mean_color: str = "#ff7f0e"  # Orange
    grid_color: str = "#d3d3d3"  # Light gray
    
    # Line styles
    line_width: float = 0.8
    line_alpha: float = 0.7
    line_style: str = "-"
    mean_line_style: str = "--"
    mean_line_width: float = 1.5
    
    # Grid
    show_grid: bool = True
    grid_alpha: float = 0.3
    grid_style: str = "-"
    grid_width: float = 0.5
    
    # Markers
    marker_size: float = 6.0
    marker_style: Optional[str] = None
    
    # Fonts
    title_size: float = 14
    label_size: float = 12
    legend_size: float = 10
    
    # Layout
    tight_layout: bool = True
    title_weight: str = "bold"
    
    # Category-specific colors
    category_colors: Optional[Dict[str, str]] = None
    
    def __post_init__(self) -> None:
        """Supply default histogram colors and validate numeric/control fields.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a numeric field is nonfinite, out of range or not representable.
        TypeError
            If numeric storage/types or Boolean control types are unsupported.

        Notes
        -----
        Validation follows the PlotStyle contract. A user-supplied category
        mapping is borrowed; only the default CATEGORY_COLORS mapping is copied.
        """
        if self.category_colors is None:
            self.category_colors = CATEGORY_COLORS.copy()
        self._validate()

    def _validate(self) -> None:
        """Check numeric and Boolean fields without changing style storage.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If numeric storage/types or Boolean control types are unsupported.
        ValueError
            If a numeric field is nonfinite, out of range or not representable.

        Notes
        -----
        Construction and plot resolution use the same checks. Artist-specific
        settings and category mapping entries retain validation at their use sites.
        """
        # Validate alpha values (must be in [0, 1])
        for name in ("line_alpha", "grid_alpha"):
            value = getattr(self, name)
            _validated_style_number(value, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        
        # Validate positive values
        for name in (
            "line_width", "mean_line_width", "grid_width", "marker_size",
            "title_size", "label_size", "legend_size",
        ):
            value = getattr(self, name)
            if _validated_style_number(value, name) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        for name in ("show_grid", "tight_layout"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be a Python or NumPy Boolean scalar.")


# Preset styles
PRESET_STYLES = {
    "classic": PlotStyle(),  # Default
    
    "dark": PlotStyle(
        adm_color="#3498db",
        fdm_color="#2ecc71",
        gdm_color="#e74c3c",
        mean_color="#f39c12",
        grid_color="#555555",
    ),
    
    "publication": PlotStyle(
        line_width=1.2,
        line_alpha=0.9,
        mean_line_width=2.0,
        grid_alpha=0.2,
        title_size=16,
        label_size=14,
        legend_size=12,
    ),
}
