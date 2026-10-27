"""Private category-name, numeric-comparison and threshold rules.

Notes
-----
Configuration and reporting share these existing category rules. This module
depends on canonical constants and scalar validation, not configuration classes.
"""

from typing import Mapping, Sequence, Tuple

import numpy as np

from .. import constants
from .value_validation import _validate_immutable_numeric_scalar


def _validated_category_names(thresholds: Mapping[str, float]) -> Tuple[str, ...]:
    """Require the six category names and return their canonical order.

    Parameters
    ----------
    thresholds : Mapping[str, float]
        Category-to-upper-bound mapping; its insertion order is irrelevant.

    Returns
    -------
    tuple of str
        Names from Excellent through Very Poor, in reference order.

    Raises
    ------
    ValueError
        If a required name is missing or an additional name is present.

    Notes
    -----
    This validates names only and does not mutate the supplied mapping.
    Numeric validation is shared separately by configuration and scalar lookup.
    """
    names = tuple(category for category, _ in constants.DEFAULT_CATEGORY_ITEMS)
    for category in names:
        if category not in thresholds:
            raise ValueError(f"Missing required category: {category}")
    unexpected = [category for category in thresholds if category not in names]
    if unexpected:
        raise ValueError(f"Unexpected category names: {unexpected}")
    return names


def _category_ratio(value: object) -> Tuple[int, int]:
    """Represent a finite category operand without narrowing its value.

    Parameters
    ----------
    value : object
        Validated Python/NumPy real scalar.

    Returns
    -------
    tuple of int
        Exact numerator and positive denominator.
    """
    if isinstance(value, (int, bool, np.integer, np.bool_)):
        return int(value), 1
    return value.as_integer_ratio()


def _category_less(value: object, limit: object) -> bool:
    """Compare actual represented values, including the unbounded final limit.

    Parameters
    ----------
    value : object
        Validated finite score or category limit.
    limit : object
        Validated finite limit or positive infinity.

    Returns
    -------
    bool
        Whether value is strictly below limit, without floating conversion.
    """
    if isinstance(limit, (float, np.floating)) and np.isposinf(limit):
        return True
    numerator, denominator = _category_ratio(value)
    upper_numerator, upper_denominator = _category_ratio(limit)
    return numerator * upper_denominator < upper_numerator * denominator


def _validate_category_limit_values(limits: Sequence[object], terminal: object) -> None:
    """Validate finite increasing upper limits and the unbounded final category.

    Parameters
    ----------
    limits : sequence
        Five upper limits in canonical category order.
    terminal : object
        Very Poor upper marker, required to be positive infinity.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a limit has unsupported storage, is nonpositive/nonfinite, is not
        strictly increasing by represented value, or the terminal is not +inf.

    Examples
    --------
    >>> _validate_category_limit_values([.1, .2, .4, .8, 1.6], float("inf"))
    >>> _validate_category_limit_values([.1, .2, .4, .8, 1.6], 2.)
    Traceback (most recent call last):
        ...
    ValueError: The Very Poor category upper limit must be positive infinity.
    """
    for limit in limits:
        _validate_immutable_numeric_scalar(limit, "category threshold")
        if isinstance(limit, (float, np.floating)) and not np.isfinite(limit):
            raise ValueError("Category upper limits before Very Poor must be positive and finite.")
        if limit <= 0:
            raise ValueError("Category upper limits before Very Poor must be positive and finite.")
    if not all(_category_less(left, right) for left, right in zip(limits, limits[1:])):
        raise ValueError(
            "category_thresholds must be in ascending order "
            "(Excellent < Very Good < Good < Fair < Poor)"
        )
    _validate_immutable_numeric_scalar(terminal, "Very Poor category threshold")
    if not isinstance(terminal, (float, np.floating)) or not np.isposinf(terminal):
        raise ValueError("The Very Poor category upper limit must be positive infinity.")


def _validated_category_limits(thresholds: Mapping[str, object]) -> Tuple[object, ...]:
    """Read a complete valid scale in canonical order without changing its values.

    Parameters
    ----------
    thresholds : mapping
        Six category names and numeric upper limits; insertion order is ignored.

    Returns
    -------
    tuple
        Validated upper limits, retaining their original scalar types.

    Raises
    ------
    ValueError
        If the mapping, names or numeric limits are invalid.
    """
    if not isinstance(thresholds, Mapping):
        raise ValueError("category_thresholds must be a mapping")
    limits = tuple(thresholds[name] for name in _validated_category_names(thresholds))
    _validate_category_limit_values(limits[:-1], limits[-1])
    return limits


def _category_index(value: object, limits: Sequence[object]) -> int:
    """Find the category of a finite score using validated actual-value limits.

    Parameters
    ----------
    value : object
        Validated finite nonnegative score.
    limits : sequence
        Validated ordered upper limits, ending in positive infinity.

    Returns
    -------
    int
        Zero-based category index.

    Raises
    ------
    ValueError
        If the supplied internal scale fails to cover the score.
    """
    for index, limit in enumerate(limits):
        if _category_less(value, limit):
            return index
    raise ValueError("Category limits do not cover the supplied score.")
