"""Private FFT/IFFT validation, scaling and recovery support.

Notes
-----
These helpers retain the existing guarded transform behavior. The calculation
entry points and compatibility imports remain in kt_fsv.spectrum.
"""

import warnings
from typing import Literal, Tuple

import numpy as np


NormalizationType = Literal["none", "N", "sqrt_N"]


def _require_finite_transform(values: np.ndarray, name: str) -> None:
    """Validate real or complex transform operands before numerical evaluation.

    Parameters
    ----------
    values : np.ndarray
        Input signal or complex coefficients.
    name : str
        FFT or IFFT diagnostic name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any real or imaginary component is nonfinite.
    """
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} requires finite input components.")


def _complex_ldexp(values: np.ndarray, shift: int) -> np.ndarray:
    """Scale real and imaginary parts without materializing a power-of-two factor.

    Parameters
    ----------
    values : np.ndarray
        Finite real or complex values.
    shift : int
        Binary exponent adjustment.

    Returns
    -------
    np.ndarray
        Complex128 scaled values; the input is not mutated.
    """
    result = np.empty(values.shape, dtype=np.complex128)
    result.real = np.ldexp(values.real, shift)
    result.imag = np.ldexp(values.imag, shift)
    return result


def _fft_scaling_shifts(values: np.ndarray) -> Tuple[int, ...]:
    """Select a bounded set of range-aware scaling attempts.

    Parameters
    ----------
    values : np.ndarray
        Finite nonempty transform input.

    Returns
    -------
    tuple of int
        At most four distinct binary shifts.

    Notes
    -----
    Try a modest change with estimated accumulation headroom first, then the
    upper safe target, more downward headroom, and order-one values. The estimate
    is not a bound on every FFT backend: each actual attempt is guarded and must
    preserve its operands. Avoid always shrinking the largest input to order one,
    which can erase small components exposed by cancellation.
    """
    largest = max(np.max(np.abs(values.real)), np.max(np.abs(values.imag)))
    exponent = int(np.frexp(largest)[1])
    headroom = max(1, (values.shape[-1] - 1).bit_length()) + 4
    upper_shift = np.finfo(np.float64).maxexp - headroom - exponent
    modest_shift = min(headroom, upper_shift)
    return tuple(dict.fromkeys((modest_shift, upper_shift, modest_shift - headroom, -exponent)))


def _checked_fft_scale(values: np.ndarray, shift: int) -> np.ndarray:
    """Require a lossless operand round trip before trying a scaled transform.

    Parameters
    ----------
    values : np.ndarray
        Original finite operands.
    shift : int
        Proposed binary scaling.

    Returns
    -------
    np.ndarray
        Scaled complex128 operands.

    Raises
    ------
    FloatingPointError
        If scaling overflows or loses any represented operand information.

    Notes
    -----
    Underflow during scaling is permitted only if the exact value comparison
    after inverse scaling succeeds. Partial subnormal precision loss is rejected,
    not merely nonzero-to-zero loss. The transform itself is separately guarded.

    Examples
    --------
    >>> values = np.array([3.0, 4j])
    >>> np.array_equal(_complex_ldexp(_checked_fft_scale(values, -3), 3), values)
    True
    """
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        scaled = _complex_ldexp(values, shift)
        if not np.array_equal(_complex_ldexp(scaled, -shift), values):
            raise FloatingPointError("Temporary FFT scaling loses operand information.")
    return scaled


def _restore_fft_component(
    values: np.ndarray, shift: int, factor: float, inverse: bool,
) -> np.ndarray:
    """Combine normalization and exponent restoration before final rounding.

    Parameters
    ----------
    values : np.ndarray
        Real or imaginary component of a successfully evaluated scaled transform.
    shift : int
        Scaling previously applied to the transform input.
    factor : float
        Original-length normalization factor.
    inverse : bool
        Multiply by the factor for inverse reconstruction rather than divide.

    Returns
    -------
    np.ndarray
        Component restored to the configured output units.

    Raises
    ------
    FloatingPointError
        If the required restored component is outside the finite output range.

    Notes
    -----
    frexp separates each component and the normalization factor. Mantissa
    arithmetic stays away from underflow/overflow; ldexp performs final rounding.
    Correctly rounded subnormal or zero individual components are allowed here.
    This is not a universal correctly-rounded FFT guarantee.

    Examples
    --------
    >>> _restore_fft_component(np.array([8.0]), 1, 2.0, False)
    array([2.])
    """
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        mantissa, exponent = np.frexp(values)
        factor_mantissa, factor_exponent = np.frexp(factor)
        if inverse:
            mantissa = mantissa * factor_mantissa
            exponent = exponent - shift + factor_exponent
        else:
            mantissa = mantissa / factor_mantissa
            exponent = exponent - shift - factor_exponent
        return np.ldexp(mantissa, exponent)


def _restore_fft_output(
    transformed: np.ndarray, shift: int, original_length: int,
    normalization: NormalizationType, inverse: bool,
) -> np.ndarray:
    """Restore the required forward spectrum or inverse real/cropped result.

    Parameters
    ----------
    transformed : np.ndarray
        Finite scaled transform output.
    shift : int
        Operand scaling exponent.
    original_length : int
        Original length used by the package normalization and cropping.
    normalization : NormalizationType
        Existing convention.
    inverse : bool
        Select inverse reconstruction.

    Returns
    -------
    np.ndarray
        Required complex coefficients or real reconstructed prefix.

    Raises
    ------
    FloatingPointError
        If a required output component cannot be restored.
    """
    factor = 1.0
    if normalization == "N":
        factor = float(original_length)
    elif normalization == "sqrt_N":
        factor = float(np.sqrt(original_length))
    if inverse:
        return _restore_fft_component(transformed.real[:original_length], shift, factor, True)
    result = np.empty(transformed.shape, dtype=np.complex128)
    result.real = _restore_fft_component(transformed.real, shift, factor, False)
    result.imag = _restore_fft_component(transformed.imag, shift, factor, False)
    return result


def _check_fft_output(
    coefficients: np.ndarray, magnitude: np.ndarray, input_nonzero: bool,
) -> None:
    """Reject silent nonfinite outputs and complete forward information loss.

    Parameters
    ----------
    coefficients, magnitude : np.ndarray
        Required normalized output arrays.
    input_nonzero : bool
        Whether the original forward input contains a nonzero value.

    Returns
    -------
    None

    Raises
    ------
    FloatingPointError
        If an output is nonfinite, a nonzero coefficient loses its magnitude,
        or a nonzero forward input collapses to an entirely zero spectrum.
    """
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(magnitude)):
        raise FloatingPointError("Required FFT coefficients or magnitudes are nonfinite.")
    if np.any((magnitude == 0) & (coefficients != 0)):
        raise FloatingPointError("FFT magnitude calculation lost a nonzero coefficient.")
    if input_nonzero and not np.any(coefficients):
        raise FloatingPointError("Nonzero forward input collapsed to an entirely zero FFT.")


def _recover_transform(
    values: np.ndarray, original_length: int, normalization: NormalizationType,
    inverse: bool = False,
) -> np.ndarray:
    """Try checked scaling without accepting loss in the transform intermediates.

    Parameters
    ----------
    values : np.ndarray
        Original finite operands, padded already for the forward case.
    original_length : int
        Length before padding.
    normalization : NormalizationType
        Unchanged normalization convention.
    inverse : bool, optional
        Recover inverse reconstruction rather than a forward spectrum.

    Returns
    -------
    np.ndarray
        Safely recovered required output within the checked numerical domain.

    Raises
    ------
    FloatingPointError
        If every bounded attempt loses information, encounters an intermediate
        range failure, or cannot represent the required output.

    Notes
    -----
    An input scaling round trip alone is insufficient: intermediate FFT
    underflow is also an error. Only final output rounding may underflow.
    No success warning is issued by an unsuccessful attempt. The caller warns
    once after completed recovery. Nonzero inverse input may legitimately have
    a zero real/cropped result, so forward-collapse checks are not reused there.

    Examples
    --------
    >>> recovered = _recover_transform(np.full(64, 1e307), 64, "N")
    >>> bool(np.all(np.isfinite(recovered)))
    True
    """
    last_error = None
    for shift in _fft_scaling_shifts(values):
        try:
            scaled = _checked_fft_scale(values, shift)
            with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
                transformed = np.fft.ifft(scaled) if inverse else np.fft.fft(scaled)
                if not np.all(np.isfinite(transformed)):
                    raise FloatingPointError("Nonfinite scaled transform.")
            result = _restore_fft_output(transformed, shift, original_length, normalization, inverse)
            if inverse:
                if not np.all(np.isfinite(result)):
                    raise FloatingPointError("Nonfinite reconstructed samples.")
            else:
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    _check_fft_output(result, np.abs(result), bool(np.any(values)))
            return result
        except FloatingPointError as error:
            last_error = error
    name = "IFFT" if inverse else "FFT"
    raise FloatingPointError(
        f"{name} range recovery failed: cannot preserve operand information and "
        "produce finite representable required outputs with safe intermediate arithmetic."
    ) from last_error


def _warn_transform_recovery(name: str, normalization: NormalizationType) -> None:
    """Report successful recovery separately from epsilon regularization.

    Parameters
    ----------
    name : str
        FFT or IFFT calculation.
    normalization : NormalizationType
        Configured normalization that was preserved.

    Returns
    -------
    None

    Warns
    -----
    RuntimeWarning
        Once per recovered helper call, subject to the caller's warning filters.
    """
    warnings.warn(
        f"{name} range recovery: floating-point range limits were encountered. "
        "The same transform was recomputed using temporary power-of-two scaling, "
        f"preserving normalization={normalization!r} and output units. Review input "
        "units and numerical range; consider consistent rescaling of both datasets "
        "and the absolute epsilon floor where appropriate, and validate its effect.",
        RuntimeWarning,
        stacklevel=3,
    )
