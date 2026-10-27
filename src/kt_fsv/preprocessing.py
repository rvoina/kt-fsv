"""
Prepare paired real responses on a common coordinate grid.

This module implements step a) of the FSV procedure, "Generation of the working
data sets". Given two raw ``(x, y)`` data sets that may span different ranges and
use different sampling, it produces two *working data sets* sharing a common,
coincident independent-variable grid. The FSV core has its own length and
interpretation requirements.

:func:`build_working_data_sets` implements the adopted one-dimensional profile
of IEEE Std 1597.1-2022 §8.4(a1-a6):

- a1-a3: determine the region of overlap and count each set's points in it.
- a4:    the set with the *fewest* points in the overlap defines the grid.
- a5:    that grid becomes the common independent-variable axis.
- a6:    the other (denser) set is mapped onto the grid by *linear* interpolation.

Notes
-----
Only linear interpolation is provided. Coordinate/response conversion uses
float64; no units conversion, uniform-grid construction or feature-preservation
assessment is performed. Fidelity checks and modification reporting associated
with §8.4(a7) remain caller responsibilities. Native multidimensional
preparation is outside this helper's scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._internal.value_validation import _validate_no_masked_values

__all__ = ["WorkingDataSets", "build_working_data_sets"]

# Minimum points a set must have inside the overlap to define or be interpolated
# onto a grid (linear interpolation requires at least two).
MIN_OVERLAP_POINTS = 2


@dataclass(frozen=True, eq=False)
class WorkingDataSets:
    """
    Store a working grid, paired responses and alignment diagnostics.

    Prefer records returned by :func:`build_working_data_sets`. Direct
    construction stores the supplied objects without validation or copying.

    Parameters
    ----------
    x : np.ndarray
        Common grid. Builder-produced values are strictly increasing float64
        coordinates from the selected input inside the overlap.
    y1 : np.ndarray
        First working response, sampled on x.
    y2 : np.ndarray
        Second working response, sampled on x.
    overlap : tuple of float
        Inclusive coordinate-intersection bounds ``(xmin, xmax)``. The selected
        grid need not contain both bounds; no endpoints are inserted.
    n_overlap : tuple of int
        Counts of converted input coordinates inside the overlap, in input order.
    reference : int
        Input defining the grid: 1 or 2 for builder-produced records. The smaller
        overlap population wins; equal counts select input 1.

    Returns
    -------
    WorkingDataSets
        Record of the six supplied fields.

    Notes
    -----
    Builder-produced x, y1 and y2 have equal shape ``(N,)``, where
    ``N = min(n_overlap)``. Their arrays are writable and do not share numerical
    storage with the original inputs. The frozen dataclass prevents field
    reassignment; it does not freeze array contents. Direct construction retains
    caller references and does not establish these builder guarantees.

    Preserve raw data, coordinates, representation/units and processing choices
    separately when retaining a study record. The record carries no automatic
    operation history or proof that interpolation preserved application features.

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv import build_working_data_sets
    >>> x = np.arange(3, dtype=float)
    >>> working = build_working_data_sets(x, 2 * x, x, 2 * x)
    >>> working.reference, working.n_overlap
    (1, (3, 3))
    >>> working.x.tolist()
    [0.0, 1.0, 2.0]
    >>> working.y1.flags.writeable
    True

    This example demonstrates preparation only; FSV needs at least 32 points.
    """

    x: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    overlap: tuple[float, float]
    n_overlap: tuple[int, int]
    reference: int


def _as_real_float_array(
    values: ArrayLike, name: str, *, axis: bool = False,
) -> np.ndarray:
    """Reject unknown masked entries and complex storage before float64 conversion.

    Parameters
    ----------
    values : array_like
        Incoming signal values or independent coordinates.
    name : str
        Input label for an actionable error.
    axis : bool, optional
        Identify an independent coordinate array rather than dependent values.

    Returns
    -------
    np.ndarray
        Float64 array, with subsequent shape/finite checks left to the caller.

    Raises
    ------
    ValueError
        If an input contains unknown masked entries, has a complex dtype or
        object entries contain complex values, including zero imaginary parts.

    Notes
    -----
    NumPy mask metadata and list/tuple/object-array entries are checked before
    conversion can discard unknown samples. Fully unmasked wrappers retain
    ordinary real-input validation; no missing samples are filled or removed.
    Ordinary numeric arrays use a dtype check. Object arrays additionally need
    entry inspection because NumPy complex scalars can otherwise be cast to
    floats with discarded imaginary parts. No magnitude, phase or real-part
    selection is performed automatically; real-input conversion is unchanged.

    Examples
    --------
    >>> from kt_fsv.preprocessing import _as_real_float_array
    >>> _as_real_float_array([1, 2], 'example')
    array([1., 2.])
    """
    _validate_no_masked_values(values, name)
    array = np.asarray(values)
    has_complex = np.iscomplexobj(array)
    if array.dtype.kind == "O":
        has_complex = any(np.iscomplexobj(value) for value in array.flat)
    if has_complex:
        if axis:
            raise ValueError(
                f"{name} contains complex-typed coordinates. "
                "Provide a real coordinate array with a non-complex dtype."
            )
        raise ValueError(
            f"FSV.fsv() does not accept complex-typed input ({name}). "
            "Provide two real numeric, one-dimensional arrays sampled on the same grid. "
            "For complex data, compare real/imaginary or magnitude/phase components "
            "separately with FSV.fsv() or FSV.fsv_from_xy(), then combine the two "
            "FSVAnalysis results using combine_multidof()."
        )
    return np.asarray(array, dtype=np.float64)


def _prepare_axis(x: ArrayLike, y: ArrayLike, name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and sort a single ``(x, y)`` data set.

    Parameters
    ----------
    x : array_like
        Real independent-variable coordinates, not complex-typed storage.
    y : array_like
        Real dependent values; complex dtype or complex object entries are
        rejected even if their imaginary parts are zero.
    name : str
        Human-readable label used in error messages (e.g. ``"data set 1"``).

    Returns
    -------
    tuple of np.ndarray
        ``(x_sorted, y_sorted)`` as ``float64`` arrays sorted by ascending ``x``.

    Raises
    ------
    ValueError
        If either array contains unknown masked or complex-typed values, is not
        1-D, contains fewer than two points or nonfinite values, if lengths
        differ, or if x coordinates are not unique after conversion.
    """
    x_arr = _as_real_float_array(x, f"{name} x", axis=True)
    y_arr = _as_real_float_array(y, f"{name} y")

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError(f"{name}: x and y must be 1-D arrays.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"{name}: x and y must have equal length; "
            f"got {x_arr.shape[0]} and {y_arr.shape[0]}."
        )
    if x_arr.shape[0] < MIN_OVERLAP_POINTS:
        raise ValueError(
            f"{name}: need at least {MIN_OVERLAP_POINTS} data points; "
            f"got {x_arr.shape[0]}."
        )
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError(f"{name}: x and y must not contain NaN or Inf.")

    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    if np.any(np.diff(x_sorted) <= 0.0):
        raise ValueError(
            f"{name}: independent variable must have unique values "
            f"(duplicate x is ambiguous for interpolation)."
        )
    return x_sorted, y_sorted


def build_working_data_sets(
    x1: ArrayLike,
    y1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    *,
    kind: str = "linear",
) -> WorkingDataSets:
    """
    Align paired real responses by overlap selection and linear interpolation.

    Inputs may have different ranges, lengths and sampling, and may be unsorted.
    The caller supplies compatible coordinate units and response representations.

    Parameters
    ----------
    x1, y1 : array_like
        One-dimensional coordinate and response arrays of equal length for input
        1, with at least two points. Values are converted to float64 before
        finite/uniqueness checks. Coordinates must remain unique after conversion.
        Complex dtype and complex object entries are rejected, even with zero
        imaginary parts. Unknown masked entries are rejected before conversion.
        Fully unmasked wrappers follow the same ordinary input rules.
    x2, y2 : array_like
        Input 2 coordinates and responses, subject to the same rules. The two
        inputs must have a nonzero-width overlap containing at least two points
        from each input.
    kind : str, optional
        Interpolation scheme: ``"linear"`` is the default and only supported value.

    Returns
    -------
    WorkingDataSets
        Writable float64 coordinate/response arrays, overlap bounds, the two
        overlap counts and reference-input number. Returned arrays are detached
        from input storage. The working length is the smaller overlap count.

    Raises
    ------
    ValueError
        For unknown masked entries, complex storage, invalid dimensions/lengths,
        nonfinite converted values, duplicate converted coordinates, unsupported
        kind, absent or zero-width overlap, or fewer than two points per input
        in the overlap.
    TypeError
        If NumPy cannot convert an input representation to a float64 array.
    OverflowError
        If conversion of a supplied value, such as a Python integer, overflows.

    Notes
    -----
    The adopted §8.4(a1-a6) profile is:

    1. Convert to float64, validate, and sort each x/y pair together.
    2. Find the inclusive intersection ``[max(min x), min(max x)]`` and count
       each input's coordinates inside it.
    3. Use the input with fewer overlap points as the reference. Equal counts
       select input 1, a deterministic choice permitted by the source note.
    4. Retain the reference's existing overlap coordinates and response values.
       Interpolate the other response using its full sorted input, including
       bracketing knots outside the overlap where needed. No extrapolation or
       insertion of new grid endpoints is performed.

    Sorting and equality use the converted coordinates; distinct high-precision
    or large integer coordinates can become duplicates in float64. Conversion
    errors are not all normalized to ValueError. NumPy masked entries, including
    those in nested lists, tuples or object arrays, are rejected before conversion.
    Entire inputs are checked, including samples outside the eventual overlap.
    Nomask/all-False wrappers are accepted subject to ordinary validation.
    Resolve missing samples explicitly; no filling, dropping or automatic
    missing-data interpolation is performed.

    A common grid does not guarantee uniform spacing or retained feature fidelity.
    The helper preserves the selected sampling and can omit features present only
    on the denser input. Assess graphical fidelity, coordinate spacing and
    interpolation effects; retain modifications in the study record as required
    by the application and §8.4(a7). No full IEEE conformance claim follows from
    successful alignment.

    Use ``FSV().fsv(ws.y1, ws.y2)`` to analyze prepared responses. Preparation
    permits as few as two working points, while the FSV core requires at least 32.
    The core consumes responses indexed by sample position, not ws.x, so it cannot
    check physical-coordinate spacing or retain the alignment record itself.

    Complex-valued comparisons require an explicit component representation:
    prepare and compare real/imaginary or magnitude/phase component pairs
    separately, then combine their FSVAnalysis results using combine_multidof().
    Coordinates remain real. Strict complex rejection is this API's safety
    contract, not a prohibition of complex comparison workflows in IEEE 8.5(c).

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv import build_working_data_sets, FSV
    >>> x1 = np.linspace(0.0, 1.0, 300)
    >>> x2 = np.linspace(0.05, 1.05, 120)
    >>> ws = build_working_data_sets(x1, np.sin(6 * x1), x2, np.sin(6 * x2))
    >>> ws.reference, ws.n_overlap, ws.x.size
    (2, (285, 114), 114)
    >>> ws.overlap
    (0.05, 1.0)
    >>> analysis = FSV().fsv(ws.y1, ws.y2)
    >>> analysis.len_orig == ws.x.size
    True

    References
    ----------
    IEEE Std 1597.1-2022, Section 8.4(a): working-data-set generation. The
    implemented overlap/interpolation profile covers a1-a6; a7 fidelity and
    modification-reporting responsibilities require caller assessment.
    """
    if kind != "linear":
        raise ValueError(
            f"IEEE Std 1597.1-2022 §8.4.a6 specifies linear interpolation; "
            f"only kind='linear' is supported, got {kind!r}."
        )

    x1s, y1s = _prepare_axis(x1, y1, "data set 1")
    x2s, y2s = _prepare_axis(x2, y2, "data set 2")

    # a1: region of overlap on the common independent-variable axis.
    fmin = max(float(x1s[0]), float(x2s[0]))
    fmax = min(float(x1s[-1]), float(x2s[-1]))
    if fmax <= fmin:
        raise ValueError(
            f"Input data sets do not overlap: set 1 spans "
            f"[{x1s[0]}, {x1s[-1]}], set 2 spans [{x2s[0]}, {x2s[-1]}]."
        )

    # a2-a3: identify and count each set's points inside the overlap.
    in1 = (x1s >= fmin) & (x1s <= fmax)
    in2 = (x2s >= fmin) & (x2s <= fmax)
    n1 = int(in1.sum())
    n2 = int(in2.sum())
    if n1 < MIN_OVERLAP_POINTS or n2 < MIN_OVERLAP_POINTS:
        raise ValueError(
            f"Each data set needs at least {MIN_OVERLAP_POINTS} points inside "
            f"the overlap [{fmin}, {fmax}]; got {n1} (set 1) and {n2} (set 2)."
        )

    # a4-a6: the fewest-points set defines the grid; interpolate the other.
    if n1 <= n2:
        reference = 1
        x_common = x1s[in1]
        y1_work = y1s[in1]
        y2_work = np.interp(x_common, x2s, y2s)
    else:
        reference = 2
        x_common = x2s[in2]
        y2_work = y2s[in2]
        y1_work = np.interp(x_common, x1s, y1s)

    return WorkingDataSets(
        x=x_common,
        y1=y1_work,
        y2=y2_work,
        overlap=(fmin, fmax),
        n_overlap=(n1, n2),
        reference=reference,
    )
