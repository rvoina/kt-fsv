"""Shared reference settings, configuration defaults and operational thresholds.

Notes
-----
The DC setting, energy split, category boundaries, confidence threshold and
minimum sample counts reflect the reviewed IEEE Std 1597.1-2022 procedure.
They do not establish conformance of a complete analysis. The FFT scaling,
output switches and epsilon floor are implementation defaults, not additional
IEEE requirements. MAX_SIGNAL_LENGTH is a resource-warning threshold, not a
hard input limit or a guarantee that an allocation will succeed.

DEFAULT_CATEGORY_ITEMS is the immutable source for fresh configuration
mappings and fixed histogram boundaries. DEFAULT_CATEGORY_THRESHOLDS retains
the mutable legacy lookup mapping for import compatibility; changing that
mapping does not change the defaults of subsequently constructed configs.
This does not alter the separate policy for caller-supplied mutable mappings.

This module contains no configuration validation, warning emission or imports
from other kt_fsv modules. Formula-specific coefficients and plotting defaults
remain with their respective implementations.
"""

from typing import Dict, Literal, Tuple


# Reference DC width: bin zero plus four positive-frequency bins.
# The four nonzero-frequency bins also have negative-frequency mirrors.
DEFAULT_BREAK_POINT_DC: int = 5

# Cumulative spectral fraction used to locate the Lo/Hi breakpoint (40:60 split).
DEFAULT_THRESHOLD_LO_HI: float = 0.4

# Histogram mass target for Grade/Spread, not a statistical confidence level.
DEFAULT_GRADE_SPREAD_THRESHOLD: float = 0.85

# Canonically ordered category names and upper bounds; immutable default template.
DEFAULT_CATEGORY_ITEMS: Tuple[Tuple[str, float], ...] = (
    ("Excellent", 0.1),
    ("Very Good", 0.2),
    ("Good", 0.4),
    ("Fair", 0.8),
    ("Poor", 1.6),
    ("Very Poor", float("inf")),
)

# Mutable legacy lookup mapping; default configs copy DEFAULT_CATEGORY_ITEMS.
DEFAULT_CATEGORY_THRESHOLDS: Dict[str, float] = dict(DEFAULT_CATEGORY_ITEMS)

# Fixed histogram edges: zero followed by the six reference category upper bounds.
DEFAULT_CONFIDENCE_BINS: Tuple[float, ...] = (0.0,) + tuple(
    upper for _, upper in DEFAULT_CATEGORY_ITEMS
)

# Minimum sample count accepted by the core FSV comparison (not interpolation).
MIN_SIGNAL_LENGTH: int = 32

# Warn for accepted inputs below this count; the recommended range starts at 50.
# IEEE guidance gives 50-100 samples as a workable minimum, not an upper limit.
RECOMMENDED_MIN_LENGTH: int = 50

# Implementation choice: divide the forward FFT by sqrt(original sample count).
# The paired inverse restores this scaling, including when zero padding is used.
DEFAULT_NORMALIZATION: Literal["none", "N", "sqrt_N"] = "sqrt_N"

# Avoid extending the working data to a power of two unless explicitly requested.
DEFAULT_ZERO_PAD: bool = False

# Include the optional histogram-based Grade/Spread summaries by default.
DEFAULT_COMPUTE_GRADE_SPREAD: bool = True

# Disable optional Spread-based confidence weighting in the core FSV by default.
DEFAULT_USE_WEIGHTED_GDM: bool = False

# Absolute denominator floor used by ADM/FDM; implementation policy, not IEEE data.
# It also affects small positive denominators and can introduce scale dependence.
DEFAULT_EPSILON_FLOOR: float = 1e-10

# Warn about resource use above this count; not a hard limit or memory guarantee.
MAX_SIGNAL_LENGTH: int = 100_000_000
