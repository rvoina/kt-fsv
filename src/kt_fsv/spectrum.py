"""
FFT and spectrum computation helpers for FSV analysis.

.. warning::
    This module contains internal implementation details and is not part
    of the public API. Functions may change without notice between versions.
    
    **For standard FSV analysis, use the `FSV` class instead.**
    
    Direct usage is only recommended for research, debugging, or advanced
    customization. If you need stable access to these functions, please
    open an issue on GitHub.

Provides pure functions for FFT normalization, magnitude spectrum computation,
and inverse FFT operations.
"""

from typing import Tuple

import numpy as np

from ._internal.fft_support import (
    NormalizationType,
    _check_fft_output,
    _checked_fft_scale,
    _complex_ldexp,
    _fft_scaling_shifts,
    _recover_transform,
    _require_finite_transform,
    _restore_fft_component,
    _restore_fft_output,
    _warn_transform_recovery,
)
from .constants import DEFAULT_ZERO_PAD


def next_power_of_2(n: int) -> int:
    """
    Find the next power of 2 greater than or equal to n.

    Parameters
    ----------
    n : int
        The input number.

    Returns
    -------
    int
        The next power of 2 greater than or equal to n.
    """
    return int(2 ** np.ceil(np.log2(n)))


def compute_magnitude_spectrum(
    data: np.ndarray,
    normalization: NormalizationType,
    zero_pad: bool = DEFAULT_ZERO_PAD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute the full FFT spectrum and the magnitude of each coefficient.

    Use ``normalization`` to choose how the FFT values are scaled:

    - ``"none"`` returns the FFT coefficients without rescaling.
    - ``"N"`` divides them by the original number of input samples.
    - ``"sqrt_N"`` divides them by the square root of that sample count.
      This is the default used by ``FSV``.

    The complex coefficients retain phase information for filtering and
    reconstruction. The magnitude array contains their absolute values, with
    both positive and negative frequency bins included. Pass the same
    normalization setting to ``inverse_spectrum`` when reconstructing components.

    Parameters
    ----------
    data : np.ndarray
        Finite input-domain data, normally float64 in the public FSV pipeline.
        Internal complex transforms remain supported; the public FSV input
        contract still requires explicit real-valued component comparisons.
    normalization : NormalizationType
        Normalization method: 'none', 'N', or 'sqrt_N'
    zero_pad : bool, optional
        Whether to enlarge to the next power of two (default: False). The
        target profile keeps the original length; IEEE 8.4(b2) says this
        working-data enlargement shall be avoided.

    Returns
    -------
    k : np.ndarray
        Indices of all FFT bins, from zero to the transform length minus one.
    magnitude : np.ndarray
        Full magnitude spectrum (both positive and negative frequencies)
    fft_result : np.ndarray
        Normalized FFT result (complex values)
    N_orig : int
        Original signal length before zero padding

    Raises
    ------
    ValueError
        If an input component is nonfinite.
    FloatingPointError
        If recovery cannot preserve operand information or represent required
        coefficients/magnitudes, or a nonzero forward transform collapses to zero.

    Warns
    -----
    RuntimeWarning
        Once per successfully recovered call, prefixed "FFT range recovery:".
        Failed attempts do not emit success warnings. Caller filters still apply.

    Notes
    -----
    For example, consider a real cosine with amplitude 1 whose frequency falls
    exactly on an FFT bin other than DC or Nyquist. Without padding, ``"N"``
    normalization gives magnitude 0.5 in each of its positive and negative
    frequency bins. The function returns both bins without doubling their
    magnitudes. DC and the even-length Nyquist component each occupy a single bin.

    For the scaling identities, let N be the original sample count, L the
    transform length and c the selected divisor (1, N or sqrt(N)). If X is the
    raw FFT, the returned coefficients are F = X/c. In exact arithmetic:

    ``sum(abs(F)**2) = (L/c**2) * sum(abs(data)**2)``.

    With no padding, L=N, so ``"sqrt_N"`` preserves the sum of squared magnitudes.
    The corresponding factors for ``"none"`` and ``"N"`` are N and 1/N.
    These relationships describe numerical scaling; a physical energy
    interpretation also depends on the input quantity, units and sampling.

    - Optional zero padding changes the transform grid and can change FSV
      results. The divisor still uses original N, not the enlarged length L.
      This differs from zero-filling selected coefficients for an N-point
      inverse reconstruction.
    - The returned spectrum includes both positive and negative frequencies
    - For real signals, negative frequencies are complex conjugates of positive ones
    - Breakpoint selection uses an undoubled unique half of abs(F), including
      the even Nyquist endpoint once. Filtering retains the full complex F.
      A common positive normalization factor cancels in the ideal relative
      crossing; floating rounding near a crossing can still change the index.
    - The ordinary FFT, normalization and magnitude expressions are unchanged.
      Local guards detect range events and explicit output checks catch silent
      nonfinite magnitudes. Recovery tries at most four power-of-two scales,
      verifies operand round trips, and rejects intermediate FFT underflow.
      Normalization and exponent restoration are combined before final rounding.
    - Padding still uses original N for normalization. No raw-data preprocessing,
      normalization-mode change, clipping or epsilon adjustment is performed.
      A failed recovery can reflect a limited safe dynamic range even if a wider
      arithmetic implementation could produce some or all mathematical outputs.

    Examples
    --------
    >>> _, magnitude, coefficients, original = compute_magnitude_spectrum(
    ...     np.array([1.0, 0.0, 0.0, 0.0]), "N")
    >>> magnitude
    array([0.25, 0.25, 0.25, 0.25])
    >>> original
    4
    """
    data = np.asarray(data)
    _require_finite_transform(data, "FFT")
    N_orig = len(data)

    # Zero padding if requested
    if zero_pad:
        N_pad = next_power_of_2(N_orig)
        data = np.pad(data, (0, N_pad - N_orig), "constant")
    else:
        N_pad = N_orig

    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            # Compute FFT
            fft_result = np.fft.fft(data)

            # Apply configured normalization using original length N
            if normalization == "N":
                fft_result = fft_result / N_orig
            elif normalization == "sqrt_N":
                fft_result = fft_result / np.sqrt(N_orig)
            # else: no normalization applied ('none' option)

            # Get full spectrum frequency indices (0 to N_pad-1)
            k = np.arange(N_pad)

            # Calculate magnitude spectrum
            magnitude = np.abs(fft_result)
            _check_fft_output(fft_result, magnitude, bool(np.any(data)))
    except FloatingPointError:
        fft_result = _recover_transform(data, N_orig, normalization)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            magnitude = np.abs(fft_result)
        k = np.arange(N_pad)
        _warn_transform_recovery("FFT", normalization)

    return k, magnitude, fft_result, N_orig


def inverse_spectrum(
    fft_result: np.ndarray,
    N_orig: int,
    normalization: NormalizationType,
) -> np.ndarray:
    """
    Compute inverse FFT to recover time domain signal, using configured normalization.

    This function reverses the FFT operation, accounting for the normalization that was
    applied in the forward transform. The normalization used is determined by the
    normalization parameter:

    1. No normalization ('none'):
       - Assumes input FFT was not normalized
       - Applies standard IFFT scaling (1/N)

    2. Normalization by N ('N'):
       - Assumes input FFT was divided by N
       - Multiplies result by N to recover original amplitudes

    3. Normalization by √N ('sqrt_N'):
       - Assumes input FFT was divided by √N
       - Multiplies result by √N to recover original amplitudes

    Parameters
    ----------
    fft_result : np.ndarray
        Finite complex FFT coefficients from compute_magnitude_spectrum.
    N_orig : int
        Original length of time domain signal before zero padding
    normalization : NormalizationType
        Normalization method used in forward transform

    Returns
    -------
    np.ndarray
        Reconstructed time domain signal

    Raises
    ------
    ValueError
        If an input coefficient has a nonfinite real or imaginary component.
    FloatingPointError
        If checked recovery cannot preserve information or represent required
        reconstructed samples.

    Warns
    -----
    RuntimeWarning
        Once after successful recovery, prefixed "IFFT range recovery:".

    Notes
    -----
    - The function automatically handles the IFFT's 1/N scaling factor
    - For zero-padded signals, only the original length is returned
    - Output is forced to real values, assuming real input signal
    - Local range guards include the IFFT and normalization restoration.
      Recovery preserves original-N normalization and the returned real prefix.
      Unreturned imaginary values or samples outside the crop do not acquire a
      new validity/selection policy. Nonzero inverse input can legitimately
      return zeros, unlike complete loss of a nonzero forward transform.
    - Temporary scaling must preserve operands; the scaled transform is guarded
      against intermediate underflow as well as overflow. Final subnormal/zero
      rounding is permitted. This does not guarantee correctly-rounded values
      at every cancellation or representation boundary.

    Examples
    --------
    >>> inverse_spectrum(np.array([4.0, 0.0, 0.0, 0.0], dtype=complex), 4, "none")
    array([1., 1., 1., 1.])
    """
    fft_result = np.asarray(fft_result)
    _require_finite_transform(fft_result, "IFFT")
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            # Compute inverse FFT
            original_signal = np.fft.ifft(fft_result)

            # Apply normalization correction based on configured normalization
            if normalization == "N":
                original_signal = original_signal * N_orig
            elif normalization == "sqrt_N":
                original_signal = original_signal * np.sqrt(N_orig)
            # else: no correction needed for 'none'

            # Return only the original length
            result = np.real(original_signal[:N_orig])
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("Nonfinite reconstructed samples.")
        return result
    except FloatingPointError:
        result = _recover_transform(fft_result, N_orig, normalization, inverse=True)
    _warn_transform_recovery("IFFT", normalization)
    return result
