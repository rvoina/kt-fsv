"""
Filter creation functions for FSV analysis.

.. warning::

    This module contains internal implementation details and is not part
    of the public API. Functions may change without notice between versions.

    **For standard FSV analysis, use the `FSV` class instead.**

    Direct usage is only recommended for research, debugging, or advanced
    customization. If you need stable access to these functions, please
    open an issue on GitHub.

Provides functions for creating DC, Lo (low-frequency), and Hi (high-frequency)
filters per IEEE Std 1597.1-2022.
"""

import numpy as np

from .measures import _validate_reduction_operands, _warn_scaled_arithmetic

def create_dc_filter(fft_length: int, break_point_dc: int) -> np.ndarray:
    """
    Create a DC filter array with ones at both ends of the spectrum.

    Sets the first break_point_dc bins and last break_point_dc - 1 bins to 1,
    handling the full FFT spectrum symmetrically without duplicating DC.

    Parameters
    ----------
    fft_length : int
        Length of the FFT result
    break_point_dc : int
        Number of nonnegative-frequency bins in the DC region, starting at
        index zero. The reference setting is 5; a value of 1 retains DC only.

    Returns
    -------
    np.ndarray
        Filter array with:
        - 1s for first break_point_dc points from the start
        - 1s for last break_point_dc - 1 points from the end
        - 0s elsewhere

    Notes
    -----
    For a full spectrum FFT, this creates a symmetric filter that
    preserves both positive and negative low frequencies around DC.
    DC is self-paired, so break_point_dc=1 needs no negative-frequency bins.
    FSVConfig reports experimental settings at construction; this internal
    helper does not emit configuration warnings.
    """
    dc_filter = np.zeros(fft_length)

    # Set ones at the start (low positive frequencies including DC)
    # DC region: indices 0 to break_point_dc-1 (i.e., 0-4 for break_point_dc=5)
    dc_filter[:break_point_dc] = 1

    # Set ones at the end (low negative frequencies) for symmetric application
    # Mirror: last (break_point_dc - 1) points to maintain conjugate symmetry
    if break_point_dc > 1:
        dc_filter[-(break_point_dc - 1) :] = 1

    return dc_filter

def create_lo_filter(fft_length: int, breakpoint: int, break_point_dc: int) -> np.ndarray:
    """
    Create Lo (low-frequency) filter per IEEE Std 1597.1-2022 Table 1.

    The Lo filter passes low-frequency content (excluding DC) up to the breakpoint,
    with a smooth transition zone.

    Parameters
    ----------
    fft_length : int
        Length of the FFT result
    breakpoint : int
        The breakpoint index (i_bp) separating Lo from Hi frequencies
    break_point_dc : int
        DC break point (start of Lo passband)

    Returns
    -------
    np.ndarray
        Lo filter array for full spectrum

    Notes
    -----
    Filter structure (0-based indexing):
    - Indices 0-4: 0 (DC region, excluded)
    - Indices 5 to i_bp-4: 1 (passband)
    - Indices i_bp-3 to i_bp+3: transition per Table 1
    - Indices i_bp+4 to end: 0 (stopband)
    """
    # Transition values from Table 1
    transition = np.array([1.000, 0.834, 0.667, 0.500, 0.334, 0.167, 0.000])

    # Create single-sided filter
    n_single = fft_length // 2 + 1
    lo_filter = np.zeros(n_single)

    # Passband: indices 5 to i_bp-4 (value = 1)
    passband_start = break_point_dc  # Index 5
    passband_end = breakpoint - 3  # i_bp - 4 + 1 = i_bp - 3 (exclusive)
    if passband_end > passband_start:
        lo_filter[passband_start:passband_end] = 1.0

    # Transition zone: i_bp-3 to i_bp+3
    for j, val in enumerate(transition):
        idx = breakpoint - 3 + j
        if 0 <= idx < n_single:
            lo_filter[idx] = val

    # Mirror to full spectrum for symmetric application
    lo_filter_full = np.zeros(fft_length)
    lo_filter_full[:n_single] = lo_filter

    # Mirror (excluding DC and Nyquist)
    if fft_length % 2 == 0:
        lo_filter_full[n_single:] = lo_filter[-2:0:-1]
    else:
        lo_filter_full[n_single:] = lo_filter[-1:0:-1]

    return lo_filter_full


def create_hi_filter( fft_length: int, breakpoint: int) -> np.ndarray:
    """
    Create Hi (high-frequency) filter per IEEE Std 1597.1-2022 Table 2.

    The Hi filter passes high-frequency content from the breakpoint onwards,
    with a smooth transition zone.

    Parameters
    ----------
    fft_length : int
        Length of the FFT result
    breakpoint : int
        The breakpoint index (i_bp) separating Lo from Hi frequencies

    Returns
    -------
    np.ndarray
        Hi filter array for full spectrum

    Notes
    -----
    Filter structure (0-based indexing):
    - Indices 0 to i_bp-3: 0 (stopband)
    - Indices i_bp-3 to i_bp+2: transition per Table 2
    - Indices i_bp+3 to end: 1 (passband)
    """
    # Transition values from Table 2
    transition = np.array([0.000, 0.167, 0.334, 0.500, 0.667, 0.834, 1.000])

    # Create single-sided filter
    n_single = fft_length // 2 + 1
    hi_filter = np.zeros(n_single)

    # Transition zone: i_bp-3 to i_bp+3
    for j, val in enumerate(transition):
        idx = breakpoint - 3 + j
        if 0 <= idx < n_single:
            hi_filter[idx] = val

    # Passband: i_bp+3 to end (value = 1)
    passband_start = breakpoint + 3
    if passband_start < n_single:
        hi_filter[passband_start:] = 1.0

    # Mirror to full spectrum for symmetric application
    hi_filter_full = np.zeros(fft_length)
    hi_filter_full[:n_single] = hi_filter
    # Mirror (excluding DC and Nyquist)
    if fft_length % 2 == 0:
        hi_filter_full[n_single:] = hi_filter[-2:0:-1]
    else:
        hi_filter_full[n_single:] = hi_filter[-1:0:-1]

    return hi_filter_full

def _scaled_breakpoint_accumulation(
    region: np.ndarray, threshold: float,
) -> tuple[float, np.ndarray]:
    """Recover finite relative accumulation values without restoring absolute units.

    Parameters
    ----------
    region : np.ndarray
        Validated finite real magnitude region, possibly empty or all zero.
    threshold : float
        Finite fraction used by the original breakpoint comparison.

    Returns
    -------
    tuple of float and np.ndarray
        Target and cumulative magnitudes expressed in the same scaled units.

    Raises
    ------
    FloatingPointError
        If recovery cannot produce finite comparison operands.

    Notes
    -----
    Divide by the largest magnitude only after the ordinary accumulation fails.
    Since both sides of the comparison share the scale, absolute totals need not
    fit float64. Zero/empty regions retain their original first-crossing behavior.
    Negligible scaled contributions may underflow; no bins or thresholds are changed.

    Examples
    --------
    >>> target, cumulative = _scaled_breakpoint_accumulation(np.array([2.0, 2.0]), 0.5)
    >>> target
    1.0
    >>> cumulative
    array([1., 2.])
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            largest = np.max(np.abs(region), initial=0.0)
            scaled = region / largest if largest != 0 else region
            target = float(np.sum(scaled) * threshold)
            cumulative = np.cumsum(scaled)
    except FloatingPointError as error:
        raise FloatingPointError(
            "Lo/Hi breakpoint scaled-arithmetic recovery could not produce finite comparison values."
        ) from error
    if not np.isfinite(target) or not np.all(np.isfinite(cumulative)):
        raise FloatingPointError("Lo/Hi breakpoint scaled-arithmetic recovery produced nonfinite comparison values.")
    return target, cumulative


def compute_breakpoint(
    magnitude1: np.ndarray,
    magnitude2: np.ndarray,
    len_fft: int,
    break_point_dc: int,
    threshold_lo_hi: float,
) -> tuple[int, int, int]:
    """
    Compute the Lo/Hi transition coordinate for the adopted IEEE 8.4(c2-c4) profile.

    The statistic is a cumulative sum of Fourier magnitudes, not squared
    magnitudes. It uses the undoubled unique half-spectrum through the final
    bin at floor(len_fft/2), including Nyquist once for even lengths.

    Algorithm:
    1. Sum each dataset separately from index break_point_dc-1 through
       floor(len_fft/2), inclusive; the default start is the fifth point.
    2. Find the first cumulative sum greater than or equal to the configured
       fraction of that total (default 0.4).
    3. Add offset of 5 to get final breakpoint
    4. Use minimum of both datasets' breakpoints

    Parameters
    ----------
    magnitude1 : np.ndarray
        Full undoubled magnitude spectrum for dataset 1.
    magnitude2 : np.ndarray
        Full undoubled magnitude spectrum for dataset 2.
    len_fft : int
        Length of the FFT result
    break_point_dc : int
        DC cutoff setting; the cumulative calculation starts at this value
        minus one. Default 5 therefore starts at zero-based index 4.
    threshold_lo_hi : float
        Cumulative spectral-magnitude fraction, typically 0.4.

    Returns
    -------
    tuple[int, int, int]
        (break_point_lo_hi, break_point_lo_hi_1, break_point_lo_hi_2)
        Combined breakpoint and individual breakpoints for each dataset

    Raises
    ------
    ValueError
        If a used magnitude region or threshold is nonfinite or complex.
    FloatingPointError
        If scaled recovery cannot produce finite comparison operands.

    Warns
    -----
    RuntimeWarning
        Once per call after successful scaled-arithmetic recovery.

    Notes
    -----
    This declared scalar/endpoint convention interprets the source's sum of
    transformed ordinates; it is not a uniquely specified physical-energy
    definition. No conjugate-pair doubling or endpoint omission is performed.
    The full complex coefficients are used separately for symmetric filtering.

    The crossing is inclusive, unlike Grade/Spread's strict comparison.
    The +5 offset and subsequent transition windows do not guarantee an exact
    40:60 energy division in the reconstructed Lo/Hi signals. The returned
    transition coordinate can exceed the final unique FFT bin; it is not
    clamped. A zero selected spectrum gives the earliest qualifying index.

    Ordinary totals, targets and cumulative sums retain their evaluation order.
    On a numerical failure, each original finite magnitude region is divided by
    its own largest magnitude before reevaluation. Only relative units are needed
    for threshold crossings. Both spectra are reevaluated on fallback; rounding
    near a crossing can affect its index. Zero/empty regions retain their behavior.
    Index bounds, magnitude convention, inclusive search and final offset are
    unchanged; the range guard does not establish a unique source interpretation.

    This implementation uses vectorized NumPy operations (np.cumsum and 
    np.searchsorted) instead of manual Python loops. Benchmarking shows 
    8-11× speedup for typical signal sizes (10K-100K points) and scales 
    well to very large signals (10M-20M points maintain 8-9× speedup).   

    The original loop-based implementation is preserved in 
    tools/benchmark_vectorization.py for validation purposes.

    Examples
    --------
    >>> magnitude = np.ones(64)
    >>> tuple(int(v) for v in compute_breakpoint(magnitude, magnitude, 64, 5, 0.4))
    (20, 20, 20)
    """
    # Calculate single-sided spectrum length (up to Nyquist)
    n_single = len_fft // 2 + 1

    # 1. Compute total sum for each dataset (single-sided, starting at the 5th
    #    data point). Standard §8.4.c.2 is 1-based, so the 5th point is index 4.
    # Per standard: S = Σ_{i=5}^{N} TWDS(i) - using single-sided spectrum
    start_idx = break_point_dc - 1  # 5th data point = numpy index 4 (§8.4.c.2)
    _validate_reduction_operands(
        (magnitude1[start_idx:n_single], magnitude2[start_idx:n_single], np.asarray(threshold_lo_hi)),
        "Lo/Hi breakpoint",
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            total_sum1 = np.sum(magnitude1[start_idx:n_single])
            total_sum2 = np.sum(magnitude2[start_idx:n_single])

            # 2. Calculate target threshold values (40% of total)
            target1 = total_sum1 * threshold_lo_hi
            target2 = total_sum2 * threshold_lo_hi

            # 3-4. Find 40% location for both datasets (vectorized implementation)
            # Extract single-sided spectrum regions for efficient cumsum
            mag1_region = magnitude1[start_idx:n_single]
            mag2_region = magnitude2[start_idx:n_single]

            # Vectorized cumulative sum (NumPy C implementation)
            cumsum1 = np.cumsum(mag1_region)
            cumsum2 = np.cumsum(mag2_region)
    except FloatingPointError:
        target1, cumsum1 = _scaled_breakpoint_accumulation(magnitude1[start_idx:n_single], threshold_lo_hi)
        target2, cumsum2 = _scaled_breakpoint_accumulation(magnitude2[start_idx:n_single], threshold_lo_hi)
        _warn_scaled_arithmetic("Lo/Hi breakpoint", stacklevel=3)

    # Binary search for threshold crossing (O(log n) vs O(n) linear scan)
    # searchsorted finds first index where cumsum >= target
    idx1 = np.searchsorted(cumsum1, target1, side='left')
    idx2 = np.searchsorted(cumsum2, target2, side='left')   

    # Adjust indices back to full spectrum coordinates
    i_40_1 = start_idx + idx1
    i_40_2 = start_idx + idx2

    # 5. Breakpoint = i_40% + 5 (per IEEE Std 1597.1-2022 §8.4.c.4)
    break_point_lo_hi_1 = i_40_1 + 5
    break_point_lo_hi_2 = i_40_2 + 5

    # 6. Final breakpoint is the minimum of the two
    break_point_lo_hi = min(break_point_lo_hi_1, break_point_lo_hi_2)

    return break_point_lo_hi, break_point_lo_hi_1, break_point_lo_hi_2