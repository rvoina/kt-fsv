"""
Select completed FSV measures and combine related one-dimensional comparisons.

Notes
-----
Related-component combination follows the adopted IEEE Std 1597.1-2022 §8.5(c)
profile. Window and mask utilities provide generic selection of completed measure
values; they do not implement the excluded standards-specific §8.5(d-g) workflows.

This module provides functions for windowing and masking FSV results to focus
on regions of interest, particularly useful for antenna patterns and scenarios
where certain regions should be excluded from the comparison.
"""

from typing import Sequence, Union

import numpy as np

from ._internal.value_validation import (
    _validate_component_weight,
    _validate_no_masked_values,
    _validate_recalculation_option,
)
from .analysis import FSVAnalysis, _snapshot_is_ready, _validate_boolean_mask
from . import measures
from .constants import DEFAULT_GRADE_SPREAD_THRESHOLD


def _selected_points(
    analysis: FSVAnalysis, name: str, selection: Union[slice, np.ndarray],
) -> np.ndarray:
    """Read only required measure values, rejecting unavailable selected data.

    Parameters
    ----------
    analysis : FSVAnalysis
        Source result.
    name : str
        Base point-array attribute.
    selection : slice or np.ndarray
        Validated full-domain slice or Boolean mask.

    Returns
    -------
    np.ndarray
        Selected finite real measure values.

    Raises
    ------
    ValueError
        If a required array has an incompatible layout or selected values
        are nonfinite/complex. Excluded values are not evaluated.
    """
    points = np.asarray(getattr(analysis, name))
    if points.ndim != 1 or len(points) != len(analysis.adm_point):
        raise ValueError(f"{name} must match the one-dimensional point domain.")
    selected = points[selection]
    if np.iscomplexobj(selected) or not np.all(np.isfinite(selected)):
        raise ValueError(
            f"Selected {name} values must be finite real numbers; requested points "
            "may be unavailable. Recompute from the original full results."
        )
    return selected


def _combined_active_index(
    result1: FSVAnalysis, result2: FSVAnalysis, k: float,
) -> Union[slice, np.ndarray]:
    """Choose an endpoint domain or require equal contributing selections.

    Parameters
    ----------
    result1, result2 : FSVAnalysis
        Component comparisons with recorded selections.
    k : float
        Validated component weight.

    Returns
    -------
    slice or np.ndarray
        Agreed active domain. None/full and explicit all-True are equivalent.

    Raises
    ------
    ValueError
        If contributing selections are unknown, invalid or different.

    Examples
    --------
    >>> from kt_fsv import FSV
    >>> full = FSV().fsv(np.ones(64), np.ones(64))
    >>> selected = apply_window(full, np.arange(64) < 32)
    >>> int(np.count_nonzero(_combined_active_index(selected, selected, 0.5)))
    32
    """
    if k == 0:
        return result2._active_index()
    if k == 1:
        return result1._active_index()
    first, second = result1._active_index(), result2._active_index()
    if isinstance(first, slice):
        if isinstance(second, slice) or np.all(second):
            return first
    elif isinstance(second, slice):
        if np.all(first):
            return first
    elif np.array_equal(first, second):
        return first
    raise ValueError(
        "Contributing results must have the same active selection; "
        "apply an explicit common mask before combining."
    )


def _combine_selected_measure(
    result1: FSVAnalysis, result2: FSVAnalysis, name: str,
    selection: Union[slice, np.ndarray], k: float,
) -> np.ndarray:
    """Evaluate the existing combination law only on contributing selected values.

    Parameters
    ----------
    result1, result2 : FSVAnalysis
        Component comparisons.
    name : str
        ADM or FDM point-array attribute.
    selection : slice or np.ndarray
        Validated common or endpoint domain.
    k : float
        Validated component weight.

    Returns
    -------
    np.ndarray
        Combined selected values, independent of parent array storage.

    Raises
    ------
    ValueError
        If required selected values are invalid or unavailable.
    """
    if k == 0:
        return _selected_points(result2, name, selection).copy()
    if k == 1:
        return _selected_points(result1, name, selection).copy()
    return (
        k * _selected_points(result1, name, selection)
        + (1 - k) * _selected_points(result2, name, selection)
    )


def _expand_selection(
    values: np.ndarray, selection: Union[slice, np.ndarray], length: int,
) -> np.ndarray:
    """Restore original indices with NaN placeholders for uncomputed values.

    Parameters
    ----------
    values : np.ndarray
        Computed active values.
    selection : slice or np.ndarray
        Their original-domain index.
    length : int
        Full point-array length, not the active count.

    Returns
    -------
    np.ndarray
        Full-domain values or the original array for an already full domain.

    Examples
    --------
    >>> expanded = _expand_selection(np.array([2.0, 4.0]), np.array([True, False, True]), 3)
    >>> expanded[[0, 2]].tolist()
    [2.0, 4.0]
    >>> bool(np.isnan(expanded[1]))
    True
    """
    if isinstance(selection, slice):
        return values
    result = np.full(length, np.nan, dtype=np.result_type(values.dtype, np.float64))
    result[selection] = values
    return result


def _effective_grade_spread_threshold(analysis: FSVAnalysis) -> float:
    """Resolve the recorded confidence threshold or the legacy default.

    Parameters
    ----------
    analysis : FSVAnalysis
        Result whose configuration may be unavailable.

    Returns
    -------
    float
        Configured threshold, or the reference default when config is None.

    Notes
    -----
    The fallback does not infer or fabricate the original configuration.
    """
    if analysis.config is None:
        return DEFAULT_GRADE_SPREAD_THRESHOLD
    return analysis.config.grade_spread_threshold


def _combined_grade_spread_threshold(
    result1: FSVAnalysis, result2: FSVAnalysis, k: float,
) -> float:
    """Select a compatible threshold for requested combined confidence statistics.

    Parameters
    ----------
    result1, result2 : FSVAnalysis
        Component results, with optional configurations.
    k : float
        Component weight already validated to lie in [0, 1].

    Returns
    -------
    float
        Active endpoint threshold or the common effective threshold.

    Raises
    ------
    ValueError
        If both components contribute and their effective thresholds differ.

    Notes
    -----
    Match represented numeric values exactly, using scalar arrays to avoid
    NumPy's weak Python/scalar comparison. For example, float32(0.85) differs
    from Python 0.85, although ordinary scalar equality can report True.
    Different scalar types remain compatible when their values agree exactly.
    The chosen threshold is returned unchanged; no tolerance, rounding or dtype
    conversion is applied to subsequent Grade/Spread calculations.

    Examples
    --------
    >>> from kt_fsv import FSV, FSVConfig
    >>> first = FSV(FSVConfig(grade_spread_threshold=0.85)).fsv(
    ...     np.ones(64), np.full(64, 2.0))
    >>> second = FSV(FSVConfig(grade_spread_threshold=np.float32(0.85))).fsv(
    ...     np.ones(64), np.full(64, 2.0))
    >>> _combined_grade_spread_threshold(first, second, 0.5)
    Traceback (most recent call last):
        ...
    ValueError: Grade/Spread thresholds must match when both components contribute; got 0.85 and 0.8500000238418579.
    >>> _combined_grade_spread_threshold(first, second, 1.0)
    0.85
    >>> _combined_grade_spread_threshold(first, second, 0.0) is second.config.grade_spread_threshold
    True
    """
    if k == 0:
        return _effective_grade_spread_threshold(result2)
    first = _effective_grade_spread_threshold(result1)
    if k == 1:
        return first
    second = _effective_grade_spread_threshold(result2)
    if not bool(np.equal(np.asarray(first), np.asarray(second))):
        raise ValueError(
            "Grade/Spread thresholds must match when both components contribute; "
            f"got {first} and {second}."
        )
    return first


def apply_window(
    analysis: FSVAnalysis,
    mask: Union[np.ndarray, Sequence[bool]],
    recalculate_grade_spread: bool = True,
) -> FSVAnalysis:
    """
    Select completed measure values and recalculate their summary statistics.

    Recalculates FSV statistics using only selected points.
    The caller chooses and justifies the region of interest. This generic
    selection does not rerun core processing on cropped input signals or
    implement the standards-specific deep-null/transient procedures.

    Parameters
    ----------
    analysis : FSVAnalysis
        Source result containing completed, available pointwise measures.
    mask : np.ndarray or sequence of bool
        Nonempty, one-dimensional Boolean selection with length
        len(analysis.adm_point), over the full original point domain. Its length
        is not the current active_count or the padded FFT length.
        Flat Boolean lists/tuples and read-only or strided arrays are supported.
        At least one point must be included. Numeric indices, numeric 0/1 masks,
        other non-Boolean dtypes and masked/unknown entries are rejected.
        - True: Include point in recalculated statistics
        - False: Exclude point from statistics
    recalculate_grade_spread : bool, optional
        Python or NumPy Boolean scalar. Recalculate existing Grade/Spread
        summaries when True (default); numeric, string and array substitutes
        are not accepted.
        Uses analysis.config.grade_spread_threshold when available, otherwise
        DEFAULT_GRADE_SPREAD_THRESHOLD (0.85). If the original analysis has
        adm_grade=None or recalculation is disabled, all Grade/Spread outputs
        and weighted GDM remain None. Weighted GDM is refreshed only when the
        input also has gdm_weighted and the new Spreads are available.

    Returns
    -------
    FSVAnalysis
        New immutable result snapshot with summaries for selected points.
        Base point arrays retain their full original indexing and values;
        summaries use only the selection. Weighted values, when present, are
        NaN outside the selection. Returned arrays are read-only.
        active_mask is a defensive read-only copy of the requested selection.

    Raises
    ------
    ValueError
        If recalculate_grade_spread is not a Boolean scalar, or the mask fails
        the Boolean/dimension/nonempty contract, contains
        unknown entries, has the wrong length, or includes no points. Also
        raised if requested values are unavailable, nonfinite or complex.
    FloatingPointError
        If numerical recovery cannot produce a finite required result.

    Warns
    -----
    RuntimeWarning
        After successful scaled mean/norm recovery. Mean validation is applied
        only to active points, not the excluded weighted NaN placeholders.

    Notes
    -----
    This windowing is a post-processing step that recalculates:
    - Mean values (ADM, FDM, GDM)
    - Six-category proportions in confidence histograms
    - Visual Rating Scale values
    - Grade and Spread when requested and analysis.adm_grade is present
    
    The base arrays (adm_point, fdm_point, gdm_point) retain their full-length
    context. Trusted snapshots share these immutable buffers; callers should
    not expect new writable copies. Configuration is retained without
    mutation. A missing configuration remains None; the .85 fallback describes
    this recalculation, not an inferred setting for the original comparison.
    Reweighting uses the selected points' new Spreads, so a genuine selection
    can legitimately change weighted values even though the threshold is kept.
    Each call explicitly replaces the recorded selection; it does not intersect
    with an earlier mask. Source base arrays retain their context, so replacement
    can widen the selection when the required values exist. A selected combined
    result may have unavailable NaN placeholders outside its active mask; to
    widen such a result, recompute from the original full component comparisons.
    No original sample indices, signal lengths or FFT computations are changed.
    The validated Boolean array is used consistently for selection and weighted
    expansion. Invalid mask inputs raise errors without a warning or implicit
    conversion to Boolean. A masked array with no masked entries is accepted.

    Configuration, lengths, breakpoint and debug data describe source context,
    not the effective optional-output decisions of this window. True refreshes
    existing Grade/Spread; it does not restore summaries dropped by an earlier
    operation. Keep the selection, recalculation request, actually used threshold
    and its config/fallback origin in a caller-held processing log. The helper
    stores no operation history. Record the reason for the selection and how the
    paired point locations correspond in the study report.

    Examples
    --------
    >>> from kt_fsv import FSV
    >>> from kt_fsv.postprocessing import apply_window
    >>> import numpy as np
    >>> 
    >>> # Compare synthetic responses on a common angular grid
    >>> angles = np.linspace(0, 180, 181)
    >>> fsv = FSV()
    >>> pattern1 = np.ones(angles.size)
    >>> pattern2 = np.linspace(1.0, 2.0, angles.size)
    >>> result = fsv.fsv(pattern1, pattern2)
    >>> 
    >>> # Select the inclusive 20-160 degree region for this illustration
    >>> mask = (angles >= 20) & (angles <= 160)
    >>> result_windowed = apply_window(result, mask)
    >>> 
    >>> print(f"Full comparison GDM: {result.gdm_mean:.3f}")
    Full comparison GDM: 2.224
    >>> print(f"Selected region GDM: {result_windowed.gdm_mean:.3f}")
    Selected region GDM: 1.093

    References
    ----------
    IEEE Std 1597.1-2022, Section 8.5(f): "Reducing the influence of deep nulls"
    describes a separate procedure outside this generic helper's scope.
    """
    _validate_recalculation_option(recalculate_grade_spread)
    # Validate mask
    mask = _validate_boolean_mask(mask)
    if len(mask) != len(analysis.adm_point):
        raise ValueError(
            f"Mask length ({len(mask)}) must match signal length "
            f"({len(analysis.adm_point)})"
        )

    if not np.any(mask):
        raise ValueError("Mask cannot be all False - no points to analyze")

    # Extract masked data
    adm_masked = _selected_points(analysis, "adm_point", mask)
    fdm_masked = _selected_points(analysis, "fdm_point", mask)
    gdm_masked = _selected_points(analysis, "gdm_point", mask)

    # Recalculate mean values
    adm_mean = measures._guarded_mean(adm_masked, "windowed ADM mean")
    fdm_mean = measures._guarded_mean(fdm_masked, "windowed FDM mean")
    gdm_mean = measures._guarded_mean(gdm_masked, "windowed GDM mean")

    # Recalculate confidence histograms
    adm_counts = measures.compute_confidence_counts(adm_masked)
    fdm_counts = measures.compute_confidence_counts(fdm_masked)
    gdm_counts = measures.compute_confidence_counts(gdm_masked)
    adm_conf_hist = measures._histogram_from_counts(adm_counts)
    fdm_conf_hist = measures._histogram_from_counts(fdm_counts)
    gdm_conf_hist = measures._histogram_from_counts(gdm_counts)

    # Recalculate VRS values
    adm_visual = float(measures.convert_to_vrs(np.array([adm_mean]))[0])
    fdm_visual = float(measures.convert_to_vrs(np.array([fdm_mean]))[0])
    gdm_visual = float(measures.convert_to_vrs(np.array([gdm_mean]))[0])

    # Recalculate Grade and Spread if original had them
    adm_grade = adm_spread = None
    fdm_grade = fdm_spread = None
    gdm_grade = gdm_spread = None

    if recalculate_grade_spread and analysis.adm_grade is not None:
        threshold = _effective_grade_spread_threshold(analysis)
        adm_grade, adm_spread = measures.compute_grade_spread(threshold=threshold, counts=adm_counts)
        fdm_grade, fdm_spread = measures.compute_grade_spread(threshold=threshold, counts=fdm_counts)
        gdm_grade, gdm_spread = measures.compute_grade_spread(threshold=threshold, counts=gdm_counts)

    # Recalculate weighted GDM if original had it
    gdm_weighted = gdm_weighted_mean = K_ADM = K_FDM = None
    if analysis.gdm_weighted is not None and adm_spread is not None:
        gdm_weighted, K_ADM, K_FDM = measures.compute_weighted_gdm(
            adm_masked, fdm_masked, adm_spread, fdm_spread
        )
        # Expand back to full length with NaN for excluded points
        gdm_weighted_full = np.full_like(analysis.adm_point, np.nan)
        gdm_weighted_full[mask] = gdm_weighted
        gdm_weighted = gdm_weighted_full
        gdm_weighted_mean = measures._guarded_mean(gdm_weighted[mask], "windowed weighted GDM mean")

    # Create new FSVAnalysis with updated statistics
    # Keep point-by-point arrays unchanged for plotting
    constructor = FSVAnalysis._from_owned_arrays if _snapshot_is_ready(analysis) else FSVAnalysis
    return constructor(
        adm_point=analysis.adm_point,
        adm_mean=adm_mean,
        adm_conf_hist=adm_conf_hist,
        adm_visual=adm_visual,
        fdm_point=analysis.fdm_point,
        fdm_mean=fdm_mean,
        fdm_conf_hist=fdm_conf_hist,
        fdm_visual=fdm_visual,
        gdm_point=analysis.gdm_point,
        gdm_mean=gdm_mean,
        gdm_conf_hist=gdm_conf_hist,
        gdm_visual=gdm_visual,
        len_orig=analysis.len_orig,
        len_fft=analysis.len_fft,
        break_point_lo_hi=analysis.break_point_lo_hi,
        config=analysis.config,
        adm_grade=adm_grade,
        adm_spread=adm_spread,
        fdm_grade=fdm_grade,
        fdm_spread=fdm_spread,
        gdm_grade=gdm_grade,
        gdm_spread=gdm_spread,
        gdm_weighted=gdm_weighted,
        gdm_weighted_mean=gdm_weighted_mean,
        K_ADM=K_ADM,
        K_FDM=K_FDM,
        debug=analysis.debug,
        active_mask=mask,
    )


def create_angular_mask(
    angles: np.ndarray,
    include_ranges: list[tuple[float, float]],
) -> np.ndarray:
    """
    Select angle values inside any supplied inclusive interval.

    Parameters
    ----------
    angles : np.ndarray
        One-dimensional array of real angle values. Use the same units for the
        array and interval bounds; the examples use degrees.
    include_ranges : list[tuple[float, float]]
        Sequence of (minimum, maximum) intervals. Their union is selected.
        Each interval includes both endpoints; an empty sequence selects nothing.

    Returns
    -------
    np.ndarray
        Newly allocated writable Boolean array of length len(angles).
        True includes a position. No sample order is changed.

    Raises
    ------
    TypeError
        If inputs do not support length, iteration or the requested comparisons.
    ValueError
        If angles or range bounds contain unknown NumPy masked entries.
        Also raised if a range cannot be unpacked into two endpoints or NumPy
        broadcasting fails. These errors are not comprehensive input validation.

    Notes
    -----
    Comparisons use the supplied values without angle conversion, normalization
    or cyclic wrapping. A reversed interval matches no finite real values;
    it does not denote a region crossing zero. For a 0-360 degree representation,
    express such a region with separate intervals, for example (350, 360)
    and (0, 10). Input arrays are not converted from Python lists.

    Unknown NumPy masked entries in angles and include_ranges raise a
    field-specific ValueError before they participate in comparisons, including
    entries nested in lists, tuples or object arrays. Angles are checked even
    with no ranges; bounds are checked even with empty data. Range iterables are
    checked as encountered; single-pass range iterators are consumed once.
    Each range is checked before unpacking, then its unpacked bounds are checked.
    Fully unmasked wrappers retain ordinary comparison behavior. No missing
    entries are filled, dropped or inferred.

    This convenience helper does not validate finiteness or bound ordering.
    Provide meaningful real values and bounds before calling.
    A generated mask does not establish coordinate or physical validity.
    Empty input returns an empty mask. apply_window and get_mask_statistics
    impose their own stricter consumer contracts.

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv.postprocessing import create_angular_mask
    >>> angles = np.linspace(0, 180, 181)
    >>> 
    >>> # Include main beam only (20-160°)
    >>> mask = create_angular_mask(angles, [(20, 160)])
    >>> int(np.count_nonzero(mask))
    141
    >>> 
    >>> # Include two separate regions
    >>> mask = create_angular_mask(angles, [(30, 80), (100, 150)])
    >>> int(np.count_nonzero(mask))
    102
    >>> crossing = create_angular_mask(
    ...     np.array([0, 10, 20, 350, 360]), [(350, 360), (0, 10)])
    >>> crossing.tolist()
    [True, True, False, True, True]
    >>> create_angular_mask(
    ...     np.ma.array([0., 45., 90.], mask=[False, True, False]), [(0, 90)])
    Traceback (most recent call last):
        ...
    ValueError: angles contains masked/unknown values; resolve missing samples explicitly before comparison.
    """
    _validate_no_masked_values(angles, "angles")
    _validate_no_masked_values(include_ranges, "include_ranges")
    mask = np.zeros(len(angles), dtype=bool)

    for bounds in include_ranges:
        _validate_no_masked_values(bounds, "include_ranges")
        min_angle, max_angle = bounds
        _validate_no_masked_values((min_angle, max_angle), "include_ranges")
        mask |= (angles >= min_angle) & (angles <= max_angle)

    return mask


def create_threshold_mask(
    signal: np.ndarray,
    min_value: float = None,
    max_value: float = None,
) -> np.ndarray:
    """
    Select supplied signal values between optional inclusive bounds.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional array of real values, in the caller's chosen units.
        Values are compared directly; no magnitude, power or dB conversion
        is performed.
    min_value : float or None, optional
        Inclusive lower bound in the signal's units. None (default) imposes no
        lower bound.
    max_value : float or None, optional
        Inclusive upper bound in the signal's units. None (default) imposes no
        upper bound.

    Returns
    -------
    np.ndarray
        Newly allocated writable Boolean array of length len(signal).
        True includes a position. With both bounds None, every position is True.

    Raises
    ------
    TypeError
        If the input does not support length or the requested comparisons.
    ValueError
        If signal, min_value or max_value contains unknown NumPy masked entries,
        or if NumPy broadcasting fails. These errors are not comprehensive
        input validation.

    Notes
    -----
    Thresholds operate on signed supplied values, not absolute amplitudes.
    Provide meaningful finite real data and scalar bounds. The helper does not
    validate finiteness or bound ordering; reversed bounds select no finite real
    values. With no bounds, even nonfinite entries receive True because no value
    comparison is performed. Python lists are not converted to arrays.

    Unknown NumPy masked entries in the signal or bounds raise a field-specific
    ValueError before comparisons, including entries nested in lists, tuples or
    object arrays. The signal is checked even with no bounds; supplied bounds
    are checked even with empty data. Fully unmasked wrappers retain ordinary
    comparison behavior. Resolve missing values explicitly; no samples are
    filled, dropped or inferred. This helper performs no data cleaning
    and does not implement a standards-specific deep-null procedure.
    Empty input returns an empty mask. A nonempty all-False result is valid for
    get_mask_statistics but is rejected by apply_window.

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv.postprocessing import create_threshold_mask
    >>> signal = np.array([1.0, 0.5, 0.001, 0.00001, 0.5, 1.0])
    >>> 
    >>> # Exclude very low values (< 0.01)
    >>> mask = create_threshold_mask(signal, min_value=0.01)
    >>> print(mask)
    [ True  True False False  True  True]
    >>> create_threshold_mask(np.array([0.1, 0.5]), min_value=np.ma.masked)
    Traceback (most recent call last):
        ...
    ValueError: min_value contains masked/unknown values; resolve missing samples explicitly before comparison.
    """
    _validate_no_masked_values(signal, "signal")
    _validate_no_masked_values(min_value, "min_value")
    _validate_no_masked_values(max_value, "max_value")
    mask = np.ones(len(signal), dtype=bool)

    if min_value is not None:
        mask &= signal >= min_value

    if max_value is not None:
        mask &= signal <= max_value

    return mask


def get_mask_statistics(mask: Union[np.ndarray, Sequence[bool]]) -> dict:
    """
    Count selected and excluded positions and report their domain fractions.

    Useful for documenting windowing in validation reports.

    Parameters
    ----------
    mask : np.ndarray or sequence of bool
        Nonempty, one-dimensional Boolean selection (True=include).
        Boolean lists/tuples, read-only/strided arrays and masked arrays without
        masked entries are supported. Numeric/object masks and unknown entries
        are rejected. A nonempty all-False mask is valid.

    Returns
    -------
    dict
        Native mapping with Python integer total_points, included_points and
        excluded_points. included_fraction and excluded_fraction are NumPy
        floating scalars in [0, 1], obtained by dividing the respective count
        by total_points. These are domain fractions, not FSV category confidence.

    Raises
    ------
    ValueError
        If the mask is empty, not one-dimensional or Boolean, cannot be converted
        to an array, or contains masked/unknown entries.

    Notes
    -----
    Both fractions require a nonempty domain. Unlike apply_window, this helper
    permits no points to be selected and has no external signal-length constraint.
    Invalid masks are not cast by truthiness or flattened, and no input is mutated.

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv.postprocessing import get_mask_statistics
    >>> mask = np.array([True, True, False, False, True])
    >>> stats = get_mask_statistics(mask)
    >>> print(stats['included_fraction'])
    0.6
    >>> float(get_mask_statistics((False, False))['included_fraction'])
    0.0
    """
    mask = _validate_boolean_mask(mask)
    total = len(mask)
    included = np.sum(mask)
    excluded = total - included

    return {
        "total_points": total,
        "included_points": int(included),
        "excluded_points": int(excluded),
        "included_fraction": included / total,
        "excluded_fraction": excluded / total,
    }


def combine_multidof(
    result1: FSVAnalysis,
    result2: FSVAnalysis,
    k: float = 0.5,
    component1_name: str = "Component 1",
    component2_name: str = "Component 2",
    recalculate_grade_spread: bool = True,
) -> FSVAnalysis:
    """
    Combine two related one-dimensional comparisons using the adopted §8.5(c) profile.

    For complex-valued data (S-parameters, impedance, E-fields), run FSV
    separately on each real-valued component (real/imag or mag/phase), then
    combine ADM and FDM pointwise. Derive GDM from those combined measures
    using Equation (11), rather than averaging the prior GDM arrays.

    Parameters
    ----------
    result1 : FSVAnalysis
        FSV results from first component (e.g., magnitude, real part)
    result2 : FSVAnalysis
        FSV results from second component (e.g., phase, imaginary part)
    k : real scalar, optional
        Finite Python int/float or NumPy integer/floating scalar (0 ≤ k ≤ 1).
        Default: 0.5.
        Containers, complex values and durations are rejected. Existing numeric
        Boolean endpoint aliases remain accepted; valid scalar types are not
        converted.
        - k=0.5: Equal weighting (IEEE default)
        - k>0.5: Emphasize component 1
        - k<0.5: Emphasize component 2
    component1_name : str, optional
        Unused compatibility label; not stored or rendered. Retain the first
        component's name in the caller's processing log/report. Default:
        "Component 1".
    component2_name : str, optional
        Unused compatibility label, with the same caller-recording requirement.
        Default: "Component 2".
    recalculate_grade_spread : bool, optional
        Python or NumPy Boolean scalar; other truthy/falsy inputs are rejected.
        Recalculate Grade/Spread and weighted GDM from the combined arrays
        (default: True), regardless of the parents' optional-output flags or
        presence. For 0<k<1, represented numeric confidence thresholds
        must match exactly. Scalar types may differ when the values are equal;
        Python 0.85 and float32(0.85) are distinct settings.
        A missing config uses DEFAULT_GRADE_SPREAD_THRESHOLD (0.85). At k=0
        or k=1, only the active result's threshold is used. When False, these
        optional outputs remain None and threshold matching is not required.

    Returns
    -------
    FSVAnalysis
        New immutable snapshot with combined ADM/FDM, derived pointwise GDM
        and summaries on the recorded active selection. Base and optional
        weighted arrays are read-only, independent of the parent point buffers,
        and retain the full index domain with NaN outside selected values.
        Configuration, lengths and breakpoint come from the first result for
        0<k<1, or the active result at an endpoint. A missing selected config
        remains None. Debug data are always None, including at endpoints.

    Raises
    ------
    ValueError
        If the recalculation request is not a Boolean scalar, k is not a
        supported finite scalar in [0, 1], ADM point-array lengths differ, or both components
        contribute and requested Grade/Spread recalculation has mismatched
        effective thresholds, or contributing active selections differ or are
        unknown/invalid, or a required selected operand is unavailable/nonfinite/complex.
    FloatingPointError
        If numerical recovery cannot produce a finite required result.

    Warns
    -----
    RuntimeWarning
        After successful scaled mean/norm recovery. Means remain reductions of
        the combined pointwise arrays, not combinations of parent scalar means.

    Notes
    -----
    Per IEEE §8.5(c), the component combination is:

    ⟨xDM_i⟩ = k · Component1{xDM_i} + (1-k) · Component2{xDM_i}

    Here xDM represents ADM or FDM at point i. The combined base GDM is
    sqrt(ADM_combined**2 + FDM_combined**2) pointwise, followed by its mean
    and histogram. It is not a mean of the previous GDM arrays or the norm
    of the scalar ADM/FDM means.

    Validation reports must document:
    1. Decomposition method (cartesian/polar)
    2. k value used
    3. Justification if k ≠ 0.5
    4. Pre-processing applied (unwrapping, normalization, etc.)

    The function recalculates statistics from the combined point-by-point
    measures. Confidence weighting uses the new ADM/FDM Spreads and the
    selected threshold; the original weighted GDM arrays are not averaged.
    Weighted GDM can exceed base GDM when the confidence rule amplifies ADM.

    The inherited config/length/breakpoint fields describe the selected source
    context, not a unique combined FFT or a complete processing recipe. For
    example, the returned config can say use_weighted_gdm=False while this
    operation has computed weighted GDM. Keep both source settings, component
    identities/names, k, the request and the effective threshold/origin in a
    caller-held log. Names and k are not retained in the returned result, and
    the name parameters currently have no effect on the calculation or display.

    Both inputs must represent aligned point locations. Threshold matching is
    a library contract, not proof that all other input settings are equivalent.
    A missing config cannot recover an unrecorded custom threshold, and no
    complete original configuration is fabricated for the fallback.
    For 0<k<1, active selections must match; None/full and an explicit all-True
    mask are equivalent. Apply a caller-chosen common mask to both results when
    necessary. No automatic intersection or coordinate alignment is performed.
    At k=0 or k=1 only the contributing parent's selection and values are used;
    inactive values are never multiplied by zero. Existing point-length checks
    still apply to both inputs. Selection and endpoint handling happen before
    arithmetic, so excluded invalid/range-limited values are not evaluated.
    All summaries and optional weighting are recomputed on the active domain.
    The output owns a read-only selection snapshot independent of both parents.
    Legacy objects without recorded selection require a known explicit mask or
    recomputation. Scalar scores and optional NaNs cannot reconstruct it reliably.
    The caller remains responsible for recording k, component names and
    preprocessing; these are not retained as new fields in FSVAnalysis.

    The deterministic example below illustrates the interface. Its k=0.6 is not
    a recommended physical weighting; justify k for the actual study. Printed
    component GDM means summarize the separate comparisons, rather than scalar
    contributions that can be averaged to obtain combined GDM.

    Examples
    --------
    >>> # S-parameter comparison (magnitude + phase)
    >>> from kt_fsv import FSV
    >>> from kt_fsv.postprocessing import combine_multidof
    >>> import numpy as np
    >>>
    >>> # Decompose S-parameter
    >>> angle = 2 * np.pi * np.arange(64) / 64
    >>> phase_sim = 1 + 0.2 * np.sin(6 * angle) + 0.05 * np.cos(20 * angle)
    >>> phase_meas = phase_sim + 0.02 + 0.01 * np.sin(6 * angle)
    >>> S11_sim = (
    ...     0.3 + 0.02 * np.sin(6 * angle) + 0.01 * np.cos(20 * angle)
    ... ) * np.exp(1j * phase_sim)
    >>> S11_meas = (
    ...     0.32 + 0.025 * np.sin(6 * angle) + 0.012 * np.cos(20 * angle)
    ... ) * np.exp(1j * phase_meas)
    >>>
    >>> mag_sim = np.abs(S11_sim)
    >>> mag_meas = np.abs(S11_meas)
    >>> phase_sim = np.unwrap(np.angle(S11_sim))
    >>> phase_meas = np.unwrap(np.angle(S11_meas))
    >>>
    >>> # Run FSV separately
    >>> fsv = FSV()
    >>> result_mag = fsv.fsv(mag_sim, mag_meas)
    >>> result_phase = fsv.fsv(phase_sim, phase_meas)
    >>>
    >>> # Combine with 60% magnitude, 40% phase
    >>> k = 0.6
    >>> result = combine_multidof(
    ...     result_mag, result_phase,
    ...     k=k,
    ...     component1_name="Magnitude",
    ...     component2_name="Phase"
    ... )
    >>>
    >>> print(f"Combined GDM: {result.gdm_mean:.3f}")
    Combined GDM: 0.160
    >>> print(f"  Magnitude comparison GDM: {result_mag.gdm_mean:.3f} "
    ...       f"(component weight: {k})")
      Magnitude comparison GDM: 0.236 (component weight: 0.6)
    >>> print(f"  Phase comparison GDM: {result_phase.gdm_mean:.3f} "
    ...       f"(component weight: {1-k})")
      Phase comparison GDM: 0.046 (component weight: 0.4)

    References
    ----------
    IEEE Std 1597.1-2022, Section 8.5(c): "Combining comparisons of related data set"
    """
    _validate_recalculation_option(recalculate_grade_spread)
    # Validate inputs
    _validate_component_weight(k)

    if len(result1.adm_point) != len(result2.adm_point):
        raise ValueError(
            f"Results must have same length: "
            f"{len(result1.adm_point)} vs {len(result2.adm_point)}"
        )

    metadata_source = result2 if k == 0 else result1
    selection = _combined_active_index(result1, result2, k)
    threshold = (
        _combined_grade_spread_threshold(result1, result2, k)
        if recalculate_grade_spread else DEFAULT_GRADE_SPREAD_THRESHOLD
    )

    # Combine point-by-point measures per IEEE formula
    adm_combined = _combine_selected_measure(result1, result2, "adm_point", selection, k)
    fdm_combined = _combine_selected_measure(result1, result2, "fdm_point", selection, k)
    gdm_combined = measures.compute_gdm(adm_combined, fdm_combined)

    # Recalculate statistics from combined arrays

    adm_mean = measures._guarded_mean(adm_combined, "combined ADM mean")
    fdm_mean = measures._guarded_mean(fdm_combined, "combined FDM mean")
    gdm_mean = measures._guarded_mean(gdm_combined, "combined GDM mean")

    adm_counts = measures.compute_confidence_counts(adm_combined)
    fdm_counts = measures.compute_confidence_counts(fdm_combined)
    gdm_counts = measures.compute_confidence_counts(gdm_combined)
    adm_conf_hist = measures._histogram_from_counts(adm_counts)
    fdm_conf_hist = measures._histogram_from_counts(fdm_counts)
    gdm_conf_hist = measures._histogram_from_counts(gdm_counts)

    adm_visual = float(measures.convert_to_vrs(np.array([adm_mean]))[0])
    fdm_visual = float(measures.convert_to_vrs(np.array([fdm_mean]))[0])
    gdm_visual = float(measures.convert_to_vrs(np.array([gdm_mean]))[0])

    # Optional: Grade and Spread
    adm_grade = adm_spread = None
    fdm_grade = fdm_spread = None
    gdm_grade = gdm_spread = None
    gdm_weighted = None
    gdm_weighted_mean = None
    K_ADM = None
    K_FDM = None

    if recalculate_grade_spread:
        adm_grade, adm_spread = measures.compute_grade_spread(threshold=threshold, counts=adm_counts)
        fdm_grade, fdm_spread = measures.compute_grade_spread(threshold=threshold, counts=fdm_counts)
        gdm_grade, gdm_spread = measures.compute_grade_spread(threshold=threshold, counts=gdm_counts)

        # Compute weighted GDM if spreads available
        if adm_spread is not None and fdm_spread is not None:
            # Pass arrays to compute_weighted_gdm (not scalars)
            gdm_weighted, K_ADM, K_FDM = measures.compute_weighted_gdm(
                adm_combined, fdm_combined, adm_spread, fdm_spread
            )
            gdm_weighted_mean = measures._guarded_mean(gdm_weighted, "combined weighted GDM mean")

    full_length = len(metadata_source.adm_point)
    if gdm_weighted is not None:
        gdm_weighted = _expand_selection(gdm_weighted, selection, full_length)

    # Use active endpoint metadata, or first-result metadata for an interior mix
    return FSVAnalysis._from_owned_arrays(
        adm_point=_expand_selection(adm_combined, selection, full_length),
        adm_mean=adm_mean,
        adm_conf_hist=adm_conf_hist,
        adm_visual=adm_visual,
        fdm_point=_expand_selection(fdm_combined, selection, full_length),
        fdm_mean=fdm_mean,
        fdm_conf_hist=fdm_conf_hist,
        fdm_visual=fdm_visual,
        gdm_point=_expand_selection(gdm_combined, selection, full_length),
        gdm_mean=gdm_mean,
        gdm_conf_hist=gdm_conf_hist,
        gdm_visual=gdm_visual,
        len_orig=metadata_source.len_orig,
        len_fft=metadata_source.len_fft,
        break_point_lo_hi=metadata_source.break_point_lo_hi,
        config=metadata_source.config,
        adm_grade=adm_grade,
        adm_spread=adm_spread,
        fdm_grade=fdm_grade,
        fdm_spread=fdm_spread,
        gdm_grade=gdm_grade,
        gdm_spread=gdm_spread,
        gdm_weighted=gdm_weighted,
        gdm_weighted_mean=gdm_weighted_mean,
        K_ADM=K_ADM,
        K_FDM=K_FDM,
        debug=None,  # Combined result has no debug data
        active_mask=None if isinstance(selection, slice) else selection,
    )
