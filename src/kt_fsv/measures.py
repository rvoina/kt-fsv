"""
FSV measure computation functions.

.. warning::
    This module contains internal implementation details and is not part
    of the public API. Functions may change without notice between versions.
    
    **For standard FSV analysis, use the `FSV` class instead.**
    
    Direct usage is only recommended for research, debugging, or advanced
    customization. If you need stable access to these functions, please
    open an issue on GitHub.

Provides functions for computing ADM, FDM, GDM measures and related statistics
per IEEE Std 1597.1-2022.
"""

import warnings
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._internal.confidence import (
    _confidence_binary_scale,
    _confidence_count_value,
    _confidence_fraction,
    _confidence_grade,
    _confidence_probability_format,
    _confidence_spread,
    _histogram_from_counts,
    _round_confidence_cdf,
    _validated_confidence_counts,
)
from ._internal.numeric_guards import (
    _guarded_gdm_norm,
    _guarded_mean,
    _guarded_reduction,
    _scaled_gdm_norm,
    _scaled_reduction,
    _validate_reduction_operands,
    _warn_scaled_arithmetic,
)
from ._internal.value_validation import (
    _probability_ulp,
    _reporting_array,
    _validate_probability_histogram,
    _validate_reporting_scalar,
    _validate_reporting_values,
)
from .config import _category_index, _validate_epsilon
from .constants import (
    DEFAULT_CONFIDENCE_BINS,
    DEFAULT_EPSILON_FLOOR,
    DEFAULT_GRADE_SPREAD_THRESHOLD,
)


def compute_derivatives(data: np.ndarray, n_d: int) -> np.ndarray:
    """
    Compute finite-difference derivative per IEEE Std 1597.1-2022.

    Uses central difference: f'(x) = f(x + n_d) - f(x - n_d)

    Parameters
    ----------
    data : np.ndarray
        Input signal or first-derivative vector.
    n_d : int
        Offset for this difference operation. FDM uses 2 on Lo/Hi and then
        3 on the already padded first Hi derivative.

    Returns
    -------
    np.ndarray
        Derivative with symmetric zero-padding to maintain length

    Notes
    -----
    Each call pads its output symmetrically to the input length. The FDM
    sequence retains that padding between derivative stages: first differences
    have two zeros per end, and the second Hi difference has three per end.
    This is the adopted endpoint convention; see compute_fdm for its support
    and normalization implications.

    This implementation uses vectorized NumPy slicing instead of Python loops.
    Benchmarking shows 60-70x speedup for typical signal sizes (100K-20M points).
    See tests/benchmark_vectorized_derivatives.py for validation.
    """
    n = len(data)
    derivative = np.zeros(n)

    # Vectorized central difference: d[i] = data[i + n_d] - data[i - n_d]
    # Equivalent to loop but ~60-70x faster for large arrays
    derivative[n_d:n - n_d] = data[2 * n_d:] - data[:n - 2 * n_d]

    # Leading and trailing points are zero (symmetric padding)
    return derivative


def _epsilon_floor_label(
    name: str, denominator: float, epsilon: float,
    first: np.ndarray, second: np.ndarray,
) -> Optional[str]:
    """Identify a floored normalization family with a nonzero magnitude difference.

    Parameters
    ----------
    name : str
        Denominator family name.
    denominator : float
        Finite raw denominator before max(denominator, epsilon).
    epsilon : float
        Validated absolute floor.
    first, second : np.ndarray
        Finite real operands defining the numerator abs(first)-abs(second).

    Returns
    -------
    str or None
        Affected family label, distinguishing an observed zero denominator with
        nonzero numerator, or None when no diagnostic is needed.

    Notes
    -----
    Magnitude comparison avoids recomputing a difference merely for diagnostics.
    The denominator check short-circuits ordinary unregularized families. A zero
    denominator alone is not evidence of an inactive component or of underflow.

    Examples
    --------
    >>> _epsilon_floor_label('beta', 0.5, 1.0, np.array([1.0]), np.array([2.0]))
    'beta'
    >>> _epsilon_floor_label('beta', 0.0, 1.0, np.zeros(2), np.zeros(2)) is None
    True
    """
    if not denominator < epsilon:
        return None
    if not np.any(np.abs(first) != np.abs(second)):
        return None
    if denominator == 0:
        return f"{name} (zero denominator with nonzero numerator)"
    return name


def _warn_epsilon_regularization(
    measure: str, epsilon: float, labels: Tuple[Optional[str], ...],
) -> None:
    """Summarize regularization without claiming numerical recovery or invalidity.

    Parameters
    ----------
    measure : str
        ADM or FDM calculation name.
    epsilon : float
        Validated absolute floor used by the unchanged equations.
    labels : tuple of str or None
        Affected denominator families, not individual samples.

    Returns
    -------
    None

    Warns
    -----
    RuntimeWarning
        Once per affected measure call. Standard caller warning filters apply.

    Notes
    -----
    No warning is emitted for zero numerators, even when their denominator is
    floored. The diagnostic records use of the existing regularized profile;
    it does not claim that the final score changed or that it is inaccurate.
    """
    affected = ", ".join(label for label in labels if label is not None)
    if not affected:
        return
    warnings.warn(
        f"{measure} epsilon regularization: absolute denominator floor "
        f"epsilon={float(epsilon)!r} was applied to {affected} with nonzero numerators. "
        "Scores are regularized and may depend on input units/scale. Review epsilon "
        "and the significance of weak or roundoff-level components; document the "
        "choice and use consistent units.",
        RuntimeWarning,
        stacklevel=3,
    )


def compute_adm(
    lo_data1: np.ndarray,
    lo_data2: np.ndarray,
    dc_data1: np.ndarray,
    dc_data2: np.ndarray,
    epsilon: float = DEFAULT_EPSILON_FLOOR,
) -> np.ndarray:
    """
    Compute ADM (Amplitude Difference Measure) per IEEE Std 1597.1-2022 Eq. 2.

    ADM(n) = |α/β| + |χ/δ_M| * exp(|χ/δ_M|)

    where:
        α = |Lo1(n)| - |Lo2(n)|
        β = (1/N) * Σ(|Lo1(i)| + |Lo2(i)|)
        χ = |DC1(n)| - |DC2(n)|
        δ_M = (1/N) * Σ(|DC1(i) + Lo1(i)| + |DC2(i) + Lo2(i)|)

    Notes
    -----
    The δ_M term (with subscript M for "Modified") was updated in IEEE Std 1597.1-2022
    compared to the 2008 version. The change incorporates the Lo component with the DC
    component as a "stability coefficient" to prevent disproportionate effects from small
    DC-only denominators. This modification improves correlation with visual assessment:
    
    - IEEE 1597.1-2008: δ = (1/N) * Σ(|DC1(i)| + |DC2(i)|)  [DC only]
    - IEEE 1597.1-2022: δ_M = (1/N) * Σ(|DC1(i)+Lo1(i)| + |DC2(i)+Lo2(i)|)  [DC+Lo combined]
    
    This ensures DC and Lo components are assessed together for trend information,
    and prevents numerical instability when DC values are small.
    
    References
    ----------
    Zhang, G., A. P. Duffy, H. Sasse, L. Wang, and R. Jauregui, "Improvement in the 
    Definition of ODM for FSV," IEEE Transactions on Electromagnetic Compatibility, 
    vol. 55, no. 4, pp. 773–779, 2013, http://dx.doi.org/10.1109/TEMC.2012.2231416.

    Parameters
    ----------
    lo_data1 : np.ndarray
        Lo-filtered time domain data for dataset 1
    lo_data2 : np.ndarray
        Lo-filtered time domain data for dataset 2
    dc_data1 : np.ndarray
        DC time domain data for dataset 1
    dc_data2 : np.ndarray
        DC time domain data for dataset 2
    epsilon : float, optional
        Absolute denominator floor. Default: 1e-10. Must be a Python/NumPy
        integer or floating scalar with finite, positive float64 conversion;
        booleans, complex values and arrays are rejected before arithmetic.
        Valid values and the max(denominator, epsilon) policy are unchanged.

    Returns
    -------
    np.ndarray
        Point-by-point ADM values

    Raises
    ------
    ValueError
        If epsilon fails the finite-positive-real-scalar contract, components
        are nonfinite/complex, or a required reduction is empty.
    FloatingPointError
        If a normalization denominator cannot be recovered as finite float64.
        This does not imply that the downstream ratio is unrepresentable.

    Warns
    -----
    RuntimeWarning
        Once per denominator successfully recovered using scaled arithmetic.
        Ordinary expressions are retained; recovery scales original components
        before forming sums, preserving signed DC+Lo recombination. The unchanged
        epsilon floor is applied only after restoring the denominator's units.
        Separately, once per call when a raw beta/delta_M denominator is below
        epsilon and its numerator is nonzero. This epsilon-regularization warning
        lists affected families without claiming numerical recovery or inaccuracy.
        A zero denominator with nonzero numerator is identified explicitly.

    Examples
    --------
    >>> zero = np.zeros(64)
    >>> adm = compute_adm(zero, zero, np.ones(64), np.full(64, 2.0))
    >>> bool(np.all(adm > 0))
    True
    >>> with warnings.catch_warnings(record=True) as diagnostics:
    ...     warnings.simplefilter('always')
    ...     adm = compute_adm(zero, zero, np.full(64, 1e-12), np.full(64, 2e-12))
    >>> len(diagnostics)
    1
    >>> 'ADM epsilon regularization' in str(diagnostics[0].message)
    True
    """
    _validate_epsilon(epsilon, "epsilon")
    _validate_reduction_operands((lo_data1, lo_data2, dc_data1, dc_data2), "ADM normalization")
    # Compute α (alpha): difference of Lo magnitudes
    alpha = np.abs(lo_data1) - np.abs(lo_data2)

    # Compute β (beta): mean of sum of Lo magnitudes
    # beta = np.mean(np.abs(lo_data1) + np.abs(lo_data2))
    beta = _guarded_reduction(
        lambda lo1, lo2: np.mean(np.abs(lo1) + np.abs(lo2)),
        (lo_data1, lo_data2), "ADM normalization denominator (beta)",
    )

    # Compute χ (chi): difference of DC magnitudes
    chi = np.abs(dc_data1) - np.abs(dc_data2)

    # Compute δ_M (delta_M): mean of combined DC+Lo magnitudes
    # Updated definition per IEEE Std 1597.1-2022 (Zhang et al.)
    #  delta_M = np.mean(
    #    np.abs(dc_data1 + lo_data1) + np.abs(dc_data2 + lo_data2)
    delta_M = _guarded_reduction(
        lambda dc1, lo1, dc2, lo2: np.mean(np.abs(dc1 + lo1) + np.abs(dc2 + lo2)),
        (dc_data1, lo_data1, dc_data2, lo_data2), "ADM normalization denominator (delta_M)",
    )

    floor_beta = _epsilon_floor_label("beta", beta, epsilon, lo_data1, lo_data2)
    floor_delta = _epsilon_floor_label("delta_M", delta_M, epsilon, dc_data1, dc_data2)

    # Avoid division by zero
    beta = max(beta, epsilon)
    delta_M = max(delta_M, epsilon)

    # Compute ADM
    # First term: |α/β|
    term1 = np.abs(alpha / beta)

    # Second term (ODM): |χ/δ_M| * exp(|χ/δ_M|)
    chi_ratio = np.abs(chi / delta_M)
    term2 = chi_ratio * np.exp(chi_ratio)

    adm = term1 + term2
    _warn_epsilon_regularization("ADM", epsilon, (floor_beta, floor_delta))

    return adm


def compute_fdm(
    lo_data1: np.ndarray,
    lo_data2: np.ndarray,
    hi_data1: np.ndarray,
    hi_data2: np.ndarray,
    N: int,
    epsilon: float = DEFAULT_EPSILON_FLOOR,
) -> np.ndarray:
    """
    Compute FDM (Feature Difference Measure) per IEEE Std 1597.1-2022 Eq. 7-10.

    FDM combines three components based on derivatives:
    - FDM1: Based on first derivative of Lo signals
    - FDM2: Based on first derivative of Hi signals
    - FDM3: Based on second derivative of Hi signals

    Parameters
    ----------
    lo_data1 : np.ndarray
        Lo-filtered time domain data for dataset 1
    lo_data2 : np.ndarray
        Lo-filtered time domain data for dataset 2
    hi_data1 : np.ndarray
        Hi-filtered time domain data for dataset 1
    hi_data2 : np.ndarray
        Hi-filtered time domain data for dataset 2
    N : int
        Original signal length
    epsilon : float, optional
        Absolute denominator floor. Default: 1e-10. Must be a Python/NumPy
        integer or floating scalar with finite, positive float64 conversion;
        booleans, complex values and arrays are rejected before arithmetic.
        Valid values and the max(denominator, epsilon) policy are unchanged.

    Returns
    -------
    np.ndarray
        Point-by-point FDM values

    Raises
    ------
    ValueError
        If epsilon fails the finite-positive-real-scalar contract, original
        components or required derivative operands are nonfinite/complex,
        or a required reduction is empty.
    FloatingPointError
        If scaled recovery cannot produce a finite required denominator, even if
        a broader scaled-ratio implementation could represent the final measure.

    Warns
    -----
    RuntimeWarning
        Once per successfully recovered derivative-family denominator.
        Separately, one epsilon-regularization diagnostic per call summarizes
        FDM1/FDM2/FDM3 families whose raw denominator is below epsilon and whose
        magnitude-difference numerator is nonzero. Zero numerators stay quiet.
        Regularization is not arithmetic recovery and need not lower the final
        FDM, because the three signed terms can cancel differently.

    Notes
    -----
    The adopted endpoint convention applies symmetric zero padding after each
    derivative stage. First Lo/Hi differences use offset 2 and are padded to N;
    the offset-3 second operation uses the padded first Hi derivative, then
    pads its own output to N with three zeros at each end.

    At zero-based positions 3, 4, N-5 and N-4, the second derivative can
    therefore depend on inserted first-derivative zeros. Positions 5 through
    N-6 have complete original support and match the composed eleven-entry
    stencil. Zeroing the four additional positions would be a different
    endpoint policy and could change FDM elsewhere through normalization.

    Denominator guards retain the ordinary (c/N)*sum expression. Recovery scales
    both derivative arrays by the same power of two before their magnitude sum,
    retaining c and the supplied original N, then restores units before epsilon.
    No derivative stencil/padding or epsilon policy is changed. This does not
    recover values already lost in earlier FFT or derivative operations.

    Examples
    --------
    >>> zero = np.zeros(64)
    >>> np.array_equal(compute_fdm(zero, zero, zero, zero, 64), zero)
    True
    """
    _validate_epsilon(epsilon, "epsilon")
    _validate_reduction_operands((lo_data1, lo_data2, hi_data1, hi_data2), "FDM normalization")
    # Compute first derivatives of Lo data (n_d = 2)
    lo1_prime = compute_derivatives(lo_data1, n_d=2)
    lo2_prime = compute_derivatives(lo_data2, n_d=2)

    # Compute first derivatives of Hi data (n_d = 2)
    hi1_prime = compute_derivatives(hi_data1, n_d=2)
    hi2_prime = compute_derivatives(hi_data2, n_d=2)

    # Compute second derivatives of Hi signals (n_d = 3 on first derivative)
    hi1_double_prime = compute_derivatives(hi1_prime, n_d=3)
    hi2_double_prime = compute_derivatives(hi2_prime, n_d=3)

    # FDM1: Lo' component (Equation 7)
    # Denominator factor: 2/N
    # denom1 = (2.0 / N) * np.sum(np.abs(lo1_prime) + np.abs(lo2_prime))
    denom1 = _guarded_reduction(
        lambda first, second: (2.0 / N) * np.sum(np.abs(first) + np.abs(second)),
        (lo1_prime, lo2_prime), "FDM1 normalization denominator",
    )
    floor1 = _epsilon_floor_label("FDM1", denom1, epsilon, lo1_prime, lo2_prime)
    denom1 = max(denom1, epsilon)
    fdm1 = (np.abs(lo1_prime) - np.abs(lo2_prime)) / denom1

    # FDM2: Hi' component (Equation 8)
    # Denominator factor: 6/N
    # denom2 = (6.0 / N) * np.sum(np.abs(hi1_prime) + np.abs(hi2_prime))
    denom2 = _guarded_reduction(
        lambda first, second: (6.0 / N) * np.sum(np.abs(first) + np.abs(second)),
        (hi1_prime, hi2_prime), "FDM2 normalization denominator",
    )
    floor2 = _epsilon_floor_label("FDM2", denom2, epsilon, hi1_prime, hi2_prime)
    denom2 = max(denom2, epsilon)
    fdm2 = (np.abs(hi1_prime) - np.abs(hi2_prime)) / denom2

    # FDM3: Hi'' component (Equation 9)
    # Denominator factor: 7.2/N
    # denom3 = (7.2 / N) * np.sum(np.abs(hi1_double_prime) + np.abs(hi2_double_prime))
    denom3 = _guarded_reduction(
        lambda first, second: (7.2 / N) * np.sum(np.abs(first) + np.abs(second)),
        (hi1_double_prime, hi2_double_prime), "FDM3 normalization denominator",
    )
    floor3 = _epsilon_floor_label("FDM3", denom3, epsilon, hi1_double_prime, hi2_double_prime)
    denom3 = max(denom3, epsilon)
    fdm3 = (np.abs(hi1_double_prime) - np.abs(hi2_double_prime)) / denom3

    # Combined FDM (Equation 10 in document, combining 7-9)
    fdm = 2.0 * np.abs(fdm1 + fdm2 + fdm3)
    _warn_epsilon_regularization("FDM", epsilon, (floor1, floor2, floor3))

    return fdm


def compute_gdm(adm: np.ndarray, fdm: np.ndarray) -> np.ndarray:
    """
    Compute GDM (Global Difference Measure) per IEEE Std 1597.1-2022 Eq. 11.

    GDM(f) = sqrt(ADM(f)^2 + FDM(f)^2)

    Parameters
    ----------
    adm : np.ndarray
        Finite real floating-point ADM values, normally float64 in the pipeline.
    fdm : np.ndarray
        Finite real floating-point FDM values, broadcast-compatible with adm.

    Returns
    -------
    np.ndarray
        Pointwise GDM from the ordinary expression or scaled recovery.

    Raises
    ------
    ValueError
        If operands are nonfinite or complex.
    FloatingPointError
        If recovery cannot produce a finite result or loses a nonzero norm to zero.

    Warns
    -----
    RuntimeWarning
        When the documented scaled-norm implementation successfully recovers
        after a numerical range event. Standard Python warning filters apply.

    Notes
    -----
    Ordinary arithmetic is retained. If squaring or summing encounters overflow
    or underflow, both components are scaled by a common power of two, the same
    norm is evaluated, and the exponent is restored with ldexp. This avoids
    constructing an unrepresentable scale factor and does not change Eq. 11.
    A mixed-range call is reevaluated as a whole; last-bit differences can occur.

    Recovery is local to this norm. It cannot repair FFT, denominator or
    scalar-reduction errors elsewhere. Preprocessing is not required solely to
    silence a recovery warning and is not guaranteed to prevent one. Review units/range and validate
    any consistent transformation of both datasets. Inputs are not cast or modified;
    generic integer/object precision policies are outside this floating-point guard.

    Examples
    --------
    >>> compute_gdm(np.array([3.0, 0.0]), np.array([4.0, 0.0]))
    array([5., 0.])
    """
    return _guarded_gdm_norm(adm, fdm)


def compute_confidence_counts(values: ArrayLike) -> np.ndarray:
    """Count the six confidence categories without discarding population data.

    Parameters
    ----------
    values : array_like
        Nonempty one-dimensional finite, nonnegative real measure values.
        For selected results, supply only active values.

    Returns
    -------
    np.ndarray
        Owned six-element integer array in the standard category order.

    Raises
    ------
    ValueError
        If samples are empty, malformed, nonfinite, negative, complex or unknown.

    Notes
    -----
    Counts use the canonical stored float64 boundaries and actual represented
    sample values. Floating arrays are compared at float64 or their wider
    native precision; object entries use exact scalar comparisons. Standard
    cutoffs are all below two, so integer/Boolean samples retain their exact
    category despite any floating comparison of larger integers.
    Samples are not mutated, rounded for display or discarded. The histogram
    helper shares these counts; custom scalar limits do not change these bins.

    Examples
    --------
    >>> compute_confidence_counts([0.0, 0.15, 0.15]).tolist()
    [1, 2, 0, 0, 0, 0]
    """
    values = _reporting_array(values, "FSV values")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Confidence histogram requires a nonempty one-dimensional sample array.")
    bins = DEFAULT_CONFIDENCE_BINS
    counts = np.zeros(6, dtype=np.intp)
    if values.dtype.kind == "O":
        for value in values:
            counts[_category_index(value, bins[1:])] += 1
        return counts
    if values.dtype.kind == "f":
        values = values.astype(np.result_type(values.dtype, np.float64), copy=False)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        counts[i] = np.sum((values >= low) & (values < high))
    return counts


def compute_confidence_histogram(values: ArrayLike) -> np.ndarray:
    """
    Compute confidence histogram per IEEE Std 1597.1-2022 Table 3.

    Categorizes FSV values into 6 bins:
    - Excellent: < 0.1
    - Very Good: 0.1 - 0.2
    - Good: 0.2 - 0.4
    - Fair: 0.4 - 0.8
    - Poor: 0.8 - 1.6
    - Very Poor: >= the stored 1.6 cutoff

    Parameters
    ----------
    values : array_like
        Nonempty one-dimensional array or flat numeric sequence of finite,
        nonnegative real FSV values (ADM, FDM or GDM). For selected results,
        pass active values explicitly; this helper has no selection metadata.

    Returns
    -------
    np.ndarray
        6-element array with proportion of points in each category

    Raises
    ------
    ValueError
        If the sample population is empty, not one-dimensional, or contains
        negative, nonfinite, complex, unsupported or masked/unknown values.
    
    Notes
    -----
    Input validation neither discards invalid points nor changes the denominator.
    There is no clipping or implicit histogram renormalization.
    Categories follow the same actual-value comparisons as standard scalar
    lookup. Stored float64 cutoffs are upper-exclusive, and custom scalar
    limits do not change this standard histogram. See compute_confidence_counts
    for the precision/container policy.
    Use compute_confidence_counts when Grade/Spread must reflect the original
    population; rounded proportions alone cannot always recover its crossings.
    The benchmark statements below describe the historical counting kernel;
    input-validation overhead has not been rebenchmarked in this correction.

    This implementation uses manual loops with boolean masking rather than 
    np.histogram(). Benchmarking shows the current approach is 3-5× faster 
    for FSV use case due to:
    
    1. Small bin count (only 6 bins) - loop overhead is negligible
    2. Boolean masking already vectorized - each np.sum() executes in C
    3. np.histogram() has setup/validation overhead not needed here
    
    Performance tested up to 20M points confirms this pattern holds at scale.
    
    Benchmark validation (tests/benchmark_confidence_histogram.py) compared:
    - np.histogram: 3.3× slower at 20M points
    - np.digitize + bincount: 1.2× slower at 20M points  
    - np.searchsorted + bincount: 1.4× slower at 20M points
    - Broadcasting approach: 3.7× slower at 20M points
    
    Those measurements do not characterize the current validated helper as a whole.

    Examples
    --------
    >>> compute_confidence_histogram(np.array([0.0, 0.15])).tolist()
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    """
    return _histogram_from_counts(compute_confidence_counts(values))


def compute_grade_spread(
    histogram: Optional[ArrayLike] = None,
    threshold: float = DEFAULT_GRADE_SPREAD_THRESHOLD,
    *,
    counts: Optional[ArrayLike] = None,
) -> tuple[int, int]:
    """Compute strict-exceed Grade/Spread for the declared IEEE 8.5(b) profile.

    Parameters
    ----------
    histogram : array_like, optional
        Six finite nonnegative supplied proportions, summing to one within the
        documented input-rounding allowance. Order is Excellent through Very Poor.
        Mutually exclusive with counts; original populations are not inferred.
    threshold : float, optional
        Finite supported real scalar in (0,1), default 0.85. Its actual
        represented value is preserved, including native NumPy scalar precision.
    counts : array_like, optional
        Six nonnegative Python/NumPy integer counts with positive total.
        Boolean, floating and unknown entries are rejected before coercion.
        Use this mode when the original population is available.

    Returns
    -------
    tuple[int, int]
        Grade and greedy Spread in 1-6, both requiring strict exceed.

    Raises
    ------
    ValueError
        If neither or both input modes are supplied, entries/shape/total/cutoff
        are invalid, or the supplied histogram total does not exceed the cutoff.

    Notes
    -----
    Counts are added exactly, then each cumulative count fraction is rounded
    once to binary64. Probability-only inputs instead accumulate the supplied
    represented values exactly and round once to at least binary64, retaining
    wider floating formats. Comparison never narrows the threshold.

    Histogram-total validation uses 2*fsum(ulp(bin)) + 2*eps64 as before; it does not
    normalize probabilities or provide a tolerance for the strict comparison.
    Rounded bars can produce different crossings from their original counts.

    Grade starts at Excellent. Spread starts at the first maximum and greedily
    adds the larger adjacent category value, choosing left on ties. This retained profile
    is not guaranteed to find the shortest interval or be reflection-invariant.
    Neither input is modified. Histogram/Boolean 0/1 behavior is retained, but
    threshold one (including True) is outside the new calculation domain.

    Examples
    --------
    >>> compute_grade_spread([0.85, 0.15, 0, 0, 0, 0])
    (2, 2)
    >>> compute_grade_spread(counts=[7, 18, 67, 8, 0, 0])
    (3, 3)
    >>> compute_grade_spread(np.array([7, 18, 67, 8, 0, 0]) / 100)
    (3, 2)
    """
    if (histogram is None) == (counts is None):
        raise ValueError("Provide exactly one of histogram or counts.")
    _validate_reporting_scalar(threshold, "threshold")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1).")
    cutoff = _confidence_fraction(threshold)
    if counts is not None:
        values = _validated_confidence_counts(counts)
        category_values = tuple(Fraction(value) for value in values)
        total = sum(values)
        precision, minimum = 53, -1074
    else:
        _validate_probability_histogram(histogram)
        category_values = tuple(_confidence_fraction(value) for value in histogram)
        total = 1
        precision, minimum = _confidence_probability_format(histogram)

    def crossed(subtotal: Fraction) -> bool:
        """Compare one cumulative proportion with the unmodified cutoff.

        Parameters
        ----------
        subtotal : Fraction
            Exact count sum or represented probability sum.

        Returns
        -------
        bool
            Whether the once-rounded cumulative value strictly exceeds cutoff.
        """
        return _round_confidence_cdf(subtotal / total, precision, minimum) > cutoff

    if not crossed(sum(category_values, Fraction())):
        raise ValueError(
            "The supplied histogram total does not exceed the requested threshold; "
            "supply original counts or correct the probability distribution."
        )
    return _confidence_grade(category_values, crossed), _confidence_spread(category_values, crossed)


def _validate_spread_count(value: object, name: str) -> None:
    """Require an integer category count without converting a valid argument.

    Parameters
    ----------
    value : object
        Candidate Python or NumPy integer scalar.
    name : str
        Spread parameter name for an error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the input is Boolean, noninteger, a duration/container, or outside
        the supported count range from 1 through 6.
    """
    message = f"{name} must be an integer count from 1 to 6."
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(message)
    if isinstance(value, np.generic) and value.dtype.kind not in "iu":
        raise ValueError(message)
    if not 1 <= value <= 6:
        raise ValueError(message)


def compute_weighted_gdm(
    adm: np.ndarray,
    fdm: np.ndarray,
    spread_adm: int,
    spread_fdm: int,
) -> tuple[np.ndarray, float, float]:
    """
    Compute weighted GDM per IEEE Std 1597.1-2022 §8.5(b).

    The Spread ratio reduces FDM when ADM Spread is smaller, amplifies ADM
    when ADM Spread is larger, and gives unit coefficients for equal Spreads.
    Weighted GDM can therefore exceed the unweighted value.

    Parameters
    ----------
    adm : np.ndarray
        Point-by-point ADM values
    fdm : np.ndarray
        Point-by-point FDM values
    spread_adm : int
        ADM Spread: a Python or NumPy integer count from 1 through 6.
        Booleans, floating values, arrays and durations are not accepted.
    spread_fdm : int
        FDM Spread, with the same integer-count requirements as spread_adm.

    Returns
    -------
    tuple[np.ndarray, float, float]
        (weighted_gdm, K_ADM, K_FDM) where K values are weighting coefficients.

    Raises
    ------
    ValueError
        If either Spread is not a supported integer count in 1..6, identifying
        the invalid parameter before coefficient arithmetic. Also raised if
        norm operands or calculated coefficients are nonfinite or complex.
    FloatingPointError
        If scaled recovery cannot represent a finite, nonzero weighted norm.

    Warns
    -----
    RuntimeWarning
        Once per call when scaled-norm recovery succeeds after a numerical event.

    Notes
    -----
    Spread counts covered categories, so its minimum is one even when the
    measure values are zero. Counts are checked without rounding, clipping,
    squeezing arrays or replacing valid scalar types. This preserves the
    existing coefficient arithmetic and NumPy promotion behavior.

    Weighting formula from IEEE standard:
    - If SpreadADM < SpreadFDM: K_ADM = 1, K_FDM = SpreadADM / SpreadFDM
    - If SpreadADM > SpreadFDM: K_FDM = 1, K_ADM = SpreadADM / SpreadFDM
    - If SpreadADM = SpreadFDM: K_ADM = K_FDM = 1

    GDM_weighted = sqrt((K_ADM * ADM)² + (K_FDM * FDM)²)

    For valid Spread counts from 1 to 6, K_ADM is in [1, 6] and K_FDM is
    in [1/6, 1]. These are not probabilities and are not normalized to sum
    to one. The scalar weighted result is obtained by averaging the returned
    pointwise values, not by taking the norm of the input means.

    For finite real floating-point operands, the original weighted expression
    is attempted first. The shared GDM guard includes the weight multiplications;
    on recovery it scales the original operands before weighting, preserving
    contributions that premature product rounding could lose. See compute_gdm
    for the scaled formula, preprocessing caveats and warning behavior. A zero
    norm for zero weighted operands is valid and does not trigger a warning.

    Examples
    --------
    >>> weighted, k_adm, k_fdm = compute_weighted_gdm(
    ...     np.array([1.0]), np.array([0.0]), 3, 2
    ... )
    >>> (k_adm, k_fdm, float(weighted[0]))
    (1.5, 1.0, 1.5)
    >>> compute_weighted_gdm(np.zeros(1), np.zeros(1), 0, 1)
    Traceback (most recent call last):
        ...
    ValueError: spread_adm must be an integer count from 1 to 6.
    """
    _validate_spread_count(spread_adm, "spread_adm")
    _validate_spread_count(spread_fdm, "spread_fdm")
    # Compute weighting coefficients based on spread
    if spread_adm < spread_fdm:
        K_ADM = 1.0
        K_FDM = spread_adm / spread_fdm
    elif spread_adm > spread_fdm:
        K_FDM = 1.0
        K_ADM = spread_adm / spread_fdm
    else:
        K_ADM = 1.0
        K_FDM = 1.0

    # Compute weighted GDM
    gdm_weighted = _guarded_gdm_norm(adm, fdm, (K_ADM, K_FDM))

    return gdm_weighted, K_ADM, K_FDM


def convert_to_vrs(x: ArrayLike) -> np.ndarray:
    """
    Convert FSV values to Visual Rating Scale per IEEE Std 1597.1-2022 Table 5.

    Piecewise linear conversion:
    - X ≤ 0.1: V = 1 + 10X
    - 0.1 < X ≤ 0.2: V = 2 + 10(X - 0.099)
    - 0.2 < X ≤ 0.4: V = 3 + 5(X - 0.199)
    - 0.4 < X ≤ 0.8: V = 4 + 2.5(X - 0.399)
    - 0.8 < X ≤ 1.6: V = 5 + 1.25(X - 0.799)
    - X > 1.6: V = 6

    Parameters
    ----------
    x : array_like
        Finite nonnegative real ADM, FDM or GDM values, not category indices.
        Numeric arrays, scalars and flat numeric sequences are accepted.
        Explicit arrays may be multidimensional, zero-dimensional or empty.
        Integer arrays are supported without truncating fractional output.
        The input is not modified. Excluded result placeholders must be
        removed by selecting active values before calling this helper.

    Returns
    -------
    np.ndarray
        Float64 VRS values with the same shape as x. Ratings can be fractional.
        The nominal scale is 1-6, but the literal formula at X=1.6 gives
        V=6.00125; results are not clipped to six.

    Raises
    ------
    ValueError
        If values are negative, nonfinite, complex, unsupported or masked/unknown.

    Notes
    -----
    Output storage is float64 regardless of the input array dtype. Validation
    does not cast supplied arrays to another arithmetic dtype, so intermediate
    expressions and comparisons retain their existing NumPy behavior.
    Float64 output does not imply promoted input arithmetic. Numeric object
    arrays and Boolean 0/1 values retain their supported numeric behavior.

    The literal offsets produce these right-hand jumps (right-hand limit
    minus the value at the boundary):

    - X=0.1: 2 to 2.01, a jump of +0.01.
    - X=0.2: 3.01 to 3.005, a jump of -0.005.
    - X=0.4: 4.005 to 4.0025, a jump of -0.0025.
    - X=0.8: 5.0025 to 5.00125, a jump of -0.00125.
    - X=1.6: 6.00125 to 6, a jump of -0.00125.

    These values describe the displayed Table-5 equations, without attributing
    a purpose to their offsets. No rounding, smoothing or clipping is applied.

    Examples
    --------
    >>> convert_to_vrs(np.array([0, 1, 2], dtype=np.int64)).tolist()
    [1.0, 5.25125, 6.0]
    >>> convert_to_vrs(np.array([1.6, 2.0])).tolist()
    [6.00125, 6.0]

    References
    ----------
    IEEE Std 1597.1-2022, Table 5: Piecewise visual conversion
    """
    x = _reporting_array(x, "VRS values")
    v = np.zeros_like(x, dtype=np.float64)

    # Excellent: X <= 0.1
    mask = x <= 0.1
    v[mask] = 1 + 10 * x[mask]

    # Very Good: 0.1 < X <= 0.2
    mask = (x > 0.1) & (x <= 0.2)
    v[mask] = 2 + 10 * (x[mask] - 0.099)

    # Good: 0.2 < X <= 0.4
    mask = (x > 0.2) & (x <= 0.4)
    v[mask] = 3 + 5 * (x[mask] - 0.199)

    # Fair: 0.4 < X <= 0.8
    mask = (x > 0.4) & (x <= 0.8)
    v[mask] = 4 + 2.5 * (x[mask] - 0.399)

    # Poor: 0.8 < X <= 1.6
    mask = (x > 0.8) & (x <= 1.6)
    v[mask] = 5 + 1.25 * (x[mask] - 0.799)

    # Very Poor: X > 1.6
    mask = x > 1.6
    v[mask] = 6

    return v
