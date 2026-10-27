"""Numerical range guards used by FSV calculations.

Notes
-----
These private helpers retain ordinary arithmetic and provide the established
scaled recovery paths. They depend only on NumPy and the standard library.
"""

import warnings
from typing import Callable, Optional, Tuple

import numpy as np


def _validate_reduction_operands(operands: Tuple[np.ndarray, ...], name: str) -> None:
    """Reject invalid arithmetic inputs before forming reduction terms.

    Parameters
    ----------
    operands : tuple of np.ndarray
        Real numeric operands; empty regions are allowed for breakpoint searches.
    name : str
        Calculation identified in the diagnostic.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an operand is nonfinite or complex.
    """
    for operand in operands:
        if np.iscomplexobj(operand) or not np.all(np.isfinite(operand)):
            raise ValueError(f"{name} requires finite real operands.")


def _warn_scaled_arithmetic(name: str, stacklevel: int = 4) -> None:
    """Report successful recovery without implying automatic input preprocessing.

    Parameters
    ----------
    name : str
        Affected calculation, not necessarily a norm.
    stacklevel : int, optional
        Warning frame depth for the caller of the calculation.

    Returns
    -------
    None

    Warns
    -----
    RuntimeWarning
        Once per successfully recovered calculation; caller filters still apply.
    """
    warnings.warn(
        f"Numerical range limits were encountered while computing {name}. "
        "The same expression was re-evaluated using the documented scaled-arithmetic "
        "implementation. Review input units and numerical range; consider consistent "
        "preprocessing of both datasets where appropriate, and validate its effect.",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


def _scaled_reduction(
    operation: Callable[..., float], operands: Tuple[np.ndarray, ...], name: str,
) -> float:
    """Recover a degree-one homogeneous scalar expression from original operands.

    Parameters
    ----------
    operation : callable
        Original mean or denominator expression, before epsilon regularization.
    operands : tuple of np.ndarray
        Validated, nonempty finite floating-point arrays.
    name : str
        Required scalar identified in a failure diagnostic.

    Returns
    -------
    float
        Recovered scalar in the original units.

    Raises
    ------
    FloatingPointError
        If the required scalar cannot be recovered as a finite representable value.

    Notes
    -----
    All operands share the exponent selected by frexp from their largest absolute
    value. ldexp scales them before compound additions/absolute values and restores
    the exponent only after reduction. Powers of two avoid unnecessary rounding of
    signed DC+Lo cancellation. Epsilon is applied by the caller after restoration.
    Negligible normalized terms and final subnormal rounding may underflow, including
    a positive mean correctly rounding to zero. This is not a general compensated
    summation or correctly-rounded guarantee for all boundary/cancellation cases.

    Examples
    --------
    >>> _scaled_reduction(np.mean, (np.array([2.0, 4.0]),), 'example mean')
    3.0
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            largest = max(np.max(np.abs(operand)) for operand in operands)
            exponent = int(np.frexp(largest)[1])
            scaled = tuple(np.ldexp(operand, -exponent) for operand in operands)
            result = float(np.ldexp(operation(*scaled), exponent))
    except FloatingPointError as error:
        raise FloatingPointError(
            f"{name} scaled-arithmetic recovery could not produce a finite representable "
            "required scalar."
        ) from error
    if not np.isfinite(result):
        raise FloatingPointError(f"{name} scaled-arithmetic recovery did not produce a finite representable value.")
    return result


def _guarded_reduction(
    operation: Callable[..., float], operands: Tuple[np.ndarray, ...], name: str,
) -> float:
    """Retain ordinary reduction arithmetic, recovering and warning only on failure.

    Parameters
    ----------
    operation : callable
        Scalar degree-one homogeneous expression, with original evaluation order.
    operands : tuple of np.ndarray
        Nonempty real floating-point arrays, normally float64 in the pipeline.
    name : str
        Denominator or summary name for warnings/errors.

    Returns
    -------
    float
        Ordinary or recovered result; no input mutation or epsilon substitution.

    Raises
    ------
    ValueError
        If operands are empty, nonfinite or complex.
    FloatingPointError
        If scaled recovery cannot produce a finite required scalar.

    Warns
    -----
    RuntimeWarning
        Once after successful scaled recovery.

    Notes
    -----
    Overflow, invalid operations and division failures are guarded locally; ordinary
    underflow follows the caller's NumPy setting. Recovery permits subnormal rounding.
    All settings are restored on exit. This guard covers constituent operations inside
    the expression, not just the final reduction. It does not repair already-invalid
    FFT/derivative arrays or introduce a general integer/object arithmetic policy.

    Examples
    --------
    >>> _guarded_reduction(np.mean, (np.array([1.0, 3.0]),), 'example mean')
    2.0
    """
    _validate_reduction_operands(operands, name)
    if any(operand.size == 0 for operand in operands):
        raise ValueError(f"{name} requires nonempty operands.")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = float(operation(*operands))
            if not np.isfinite(result):
                raise FloatingPointError("Nonfinite reduction result.")
        return result
    except FloatingPointError:
        result = _scaled_reduction(operation, operands, name)
    _warn_scaled_arithmetic(name)
    return result


def _guarded_mean(values: np.ndarray, name: str) -> float:
    """Average active measure points without avoidable intermediate sum overflow.

    Parameters
    ----------
    values : np.ndarray
        Nonempty finite real floating-point values, normally nonnegative measures.
        Window callers must select active points before passing weighted arrays.
    name : str
        Summary identified in diagnostics.

    Returns
    -------
    float
        Ordinary mean or scaled recovery of the same arithmetic mean.

    Raises
    ------
    ValueError
        If operands are empty, nonfinite or complex.
    FloatingPointError
        If recovery fails.

    Warns
    -----
    RuntimeWarning
        After successful recovery; a correctly rounded zero mean is permitted.

    Examples
    --------
    >>> _guarded_mean(np.array([1.0, 2.0, 3.0]), 'example mean')
    2.0
    """
    return _guarded_reduction(np.mean, (values,), name)


def _scaled_gdm_norm(
    adm: np.ndarray, fdm: np.ndarray, coefficients: Tuple[float, float], name: str,
) -> np.ndarray:
    """Recover a norm with power-of-two scaling before weighting and squaring.

    Parameters
    ----------
    adm, fdm : np.ndarray
        Finite real floating-point operands validated by the caller.
    coefficients : tuple of float
        Unit coefficients or the approved Spread-based weights.
    name : str
        Norm name used in range-error diagnostics.

    Returns
    -------
    np.ndarray
        Scaled evaluation of the same pointwise norm, with broadcasting preserved.

    Raises
    ------
    FloatingPointError
        If recovery overflows, is nonfinite, or loses a nonzero norm to zero.

    Notes
    -----
    For each point, frexp selects e so the largest raw magnitude divided by
    2**e is in [0.5,1). With u=ADM/2**e and v=FDM/2**e, evaluate
    2**e * sqrt((K_ADM*u)**2 + (K_FDM*v)**2). ldexp applies these powers without
    materializing 2**e, which could itself be out of range. Zero operands need
    no division. Weighting occurs after scaling, not on already damaged products.

    Underflow in negligible normalized terms and final subnormal rounding is
    allowed locally. With valid Spread weights, a nonzero normalized component
    remains at least 1/12 in magnitude. Final nonzero-to-zero loss is rejected.
    This is not a new raw-input normalization or a correctly-rounded guarantee
    for every subnormal/upper-range boundary.

    Examples
    --------
    >>> _scaled_gdm_norm(np.array([3.0]), np.array([4.0]), (1.0, 1.0), 'GDM')
    array([5.])
    """
    ka, kf = coefficients
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            largest = np.maximum(np.abs(adm), np.abs(fdm))
            _, exponent = np.frexp(largest)
            adm_scaled = np.ldexp(adm, -exponent)
            fdm_scaled = np.ldexp(fdm, -exponent)
            scaled_norm = np.sqrt((ka * adm_scaled)**2 + (kf * fdm_scaled)**2)
            result = np.ldexp(scaled_norm, exponent)
    except FloatingPointError as error:
        raise FloatingPointError(
            f"{name} scaled-norm recovery could not produce a finite representable result."
        ) from error
    nonzero = ((adm != 0) & (ka != 0)) | ((fdm != 0) & (kf != 0))
    if not np.all(np.isfinite(result)) or np.any((result == 0) & nonzero):
        raise FloatingPointError(
            f"{name} scaled-norm recovery could not produce a finite representable "
            "nonzero result for nonzero weighted operands."
        )
    return result


def _guarded_gdm_norm(
    adm: np.ndarray, fdm: np.ndarray, weights: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Keep ordinary norm arithmetic and warn only after successful scaled recovery.

    Parameters
    ----------
    adm, fdm : np.ndarray
        Real floating-point measure arrays, normally float64 in the FSV pipeline.
    weights : tuple of float or None, optional
        Spread-based coefficients, or None to retain the original base expression.

    Returns
    -------
    np.ndarray
        Ordinary or recovered pointwise GDM, with NumPy broadcasting preserved.

    Raises
    ------
    ValueError
        If an operand or coefficient is nonfinite or complex.
    FloatingPointError
        If scaled recovery cannot represent the result.

    Warns
    -----
    RuntimeWarning
        Once per affected call after successful scaled-norm recovery.

    Notes
    -----
    The guard covers weighting products as well as squares, addition and square
    root. NumPy error settings are local and restored on every exit. A mixed-range
    array is reevaluated as a whole on fallback, so ordinary elements in that
    call can differ in their last bits. No input dtype coercion is introduced;
    integer/object arithmetic policies are outside this floating-point guard.

    Examples
    --------
    >>> _guarded_gdm_norm(np.array([3.0]), np.array([4.0]))
    array([5.])
    """
    name = "GDM" if weights is None else "weighted GDM"
    coefficients = (1.0, 1.0) if weights is None else weights
    for operand in (adm, fdm, *coefficients):
        if np.iscomplexobj(operand) or not np.all(np.isfinite(operand)):
            raise ValueError(f"{name} requires finite real operands and coefficients.")
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            if weights is None:
                result = np.sqrt(adm**2 + fdm**2)
            else:
                ka, kf = coefficients
                result = np.sqrt((ka * adm)**2 + (kf * fdm)**2)
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("Nonfinite norm result.")
        return result
    except FloatingPointError:
        result = _scaled_gdm_norm(adm, fdm, coefficients, name)
    warnings.warn(
        f"Numerical range limits were encountered during {name} evaluation. "
        "The same GDM formula was re-evaluated using the documented scaled-norm "
        "implementation. Review input units and numerical range; consider consistent "
        "preprocessing of both datasets where appropriate, and validate its effect.",
        RuntimeWarning,
        stacklevel=3,
    )
    return result
