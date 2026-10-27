"""Private helpers for count-based and probability-based confidence statistics.

Notes
-----
The existing measure interfaces use these helpers for histogram normalization,
exact represented values and the adopted cumulative Grade/Spread calculation.
"""

from fractions import Fraction
from typing import Callable, Sequence, Tuple

import numpy as np


def _histogram_from_counts(counts: np.ndarray) -> np.ndarray:
    """Normalize a known nonempty sample population using the original divisions.

    Parameters
    ----------
    counts : np.ndarray
        Six integer counts produced by compute_confidence_counts.

    Returns
    -------
    np.ndarray
        Six float64 category proportions.

    Raises
    ------
    ValueError
        If the total count is not positive.
    """
    n = sum(int(value) for value in counts)
    if n <= 0:
        raise ValueError("Confidence counts require a positive population.")
    histogram = np.zeros(6)
    for i, count in enumerate(counts):
        histogram[i] = count / n
    return histogram


def _confidence_count_value(value: object) -> int:
    """Validate one supplied count without coercing Boolean or floating values.

    Parameters
    ----------
    value : object
        Original sequence entry or array scalar.

    Returns
    -------
    int
        Exact nonnegative Python integer.

    Raises
    ------
    ValueError
        If the count is Boolean, noninteger or negative.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("Confidence counts must be nonnegative integers, not Booleans or floats.")
    if value < 0:
        raise ValueError("Confidence counts must be nonnegative integers.")
    return int(value)


def _validated_confidence_counts(counts: object) -> Tuple[int, ...]:
    """Validate count entries before array promotion can erase their types.

    Parameters
    ----------
    counts : object
        Flat six-element integer sequence or one-dimensional ndarray.

    Returns
    -------
    tuple of int
        Exact counts with a positive Python-integer total.

    Raises
    ------
    ValueError
        If shape, count types, population or masked entries are invalid.
    """
    if np.ma.isMaskedArray(counts) and np.any(np.ma.getmaskarray(counts)):
        raise ValueError("Confidence counts contain masked/unknown values.")
    if isinstance(counts, np.ndarray):
        valid_shape = counts.shape == (6,)
    elif isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
        valid_shape = len(counts) == 6
    else:
        valid_shape = False
    if not valid_shape:
        raise ValueError("Confidence counts require a one-dimensional six-element sequence.")
    values = tuple(_confidence_count_value(value) for value in counts)
    if sum(values) <= 0:
        raise ValueError("Confidence counts require a positive population.")
    return values


def _confidence_fraction(value: object) -> Fraction:
    """Preserve the exact numeric value of a validated reporting scalar.

    Parameters
    ----------
    value : object
        Supported real scalar already checked by the reporting validators.

    Returns
    -------
    Fraction
        Exact represented value without narrowing a native floating dtype.
    """
    if isinstance(value, (int, bool, np.integer, np.bool_)):
        return Fraction(int(value))
    return Fraction(*value.as_integer_ratio())


def _confidence_probability_format(histogram: Sequence) -> Tuple[int, int]:
    """Select cumulative precision/range without downcasting wider bins.

    Parameters
    ----------
    histogram : sequence
        Original validated probability entries.

    Returns
    -------
    tuple of int
        Significand bits and smallest subnormal exponent, at least binary64.
    """
    dtype = np.dtype(np.float64)
    for value in histogram:
        if isinstance(value, np.floating):
            dtype = np.result_type(dtype, value.dtype)
    info = np.finfo(dtype)
    return int(info.nmant + 1), int(info.minexp - info.nmant)


def _confidence_binary_scale(exponent: int) -> Fraction:
    """Construct a binary scale without floating overflow or underflow.

    Parameters
    ----------
    exponent : int
        Power of two.

    Returns
    -------
    Fraction
        Exact binary scale.
    """
    if exponent >= 0:
        return Fraction(1 << exponent)
    return Fraction(1, 1 << -exponent)


def _round_confidence_cdf(value: Fraction, precision: int, minimum: int) -> Fraction:
    """Round a cumulative proportion once with nearest/even and gradual underflow.

    Parameters
    ----------
    value : Fraction
        Validated nonnegative cumulative proportion.
    precision : int
        Binary significand bits.
    minimum : int
        Exponent of the format's smallest positive subnormal.

    Returns
    -------
    Fraction
        Exact value of the rounded cumulative representation.

    Notes
    -----
    Counts and supplied probability entries are accumulated exactly before this
    step. Fraction.__round__ supplies integer ties-to-even; keeping its result
    rational avoids a second rounding or a narrowing comparison with a cutoff.
    """
    if value == 0:
        return Fraction()
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _confidence_binary_scale(exponent):
        exponent -= 1
    quantum = _confidence_binary_scale(max(exponent - precision + 1, minimum))
    return round(value / quantum) * quantum


def _confidence_grade(
    category_values: Sequence[Fraction], crossed: Callable[[Fraction], bool],
) -> int:
    """Find the first qualifying prefix from Excellent.

    Parameters
    ----------
    category_values : sequence of Fraction
        Six exact category counts or represented proportions.
    crossed : callable
        Cumulative comparison whose full-population result is known to be True.

    Returns
    -------
    int
        Prefix length, including intervening zero categories.
    """
    # Grade: cumulative from Excellent (index 0) onwards
    cumsum = Fraction()
    grade = 0
    for i in range(6):
        cumsum += category_values[i]
        grade = i + 1
        if crossed(cumsum):
            break
    return grade


def _confidence_spread(
    category_values: Sequence[Fraction], crossed: Callable[[Fraction], bool],
) -> int:
    """Follow the retained first-mode, larger-neighbor, left-tie growth path.

    Parameters
    ----------
    category_values : sequence of Fraction
        Six exact category counts or represented proportions.
    crossed : callable
        Cumulative comparison whose full-population result is known to be True.

    Returns
    -------
    int
        Traversed contiguous width, not a minimum-width estimate.
    """
    # Spread: adjacent categories from largest bar
    max_idx = max(range(6), key=category_values.__getitem__)
    spread = 1
    cumsum = category_values[max_idx]

    # Expand left and right from max until threshold reached
    left = max_idx - 1
    right = max_idx + 1

    while not crossed(cumsum):
        left_val = category_values[left] if left >= 0 else Fraction()
        right_val = category_values[right] if right < 6 else Fraction()

        # Add neighbor with larger value
        if left_val >= right_val and left >= 0:
            cumsum += category_values[left]
            spread += 1
            left -= 1
        elif right < 6:
            cumsum += category_values[right]
            spread += 1
            right += 1
        else:
            break

    return spread
