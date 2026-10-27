"""Private validation of numeric values, selections and control options.

Notes
-----
These helpers retain the existing scalar-storage, array, histogram and option checks.
Consumers retain their own comparison and shape contracts.
"""

import math
from typing import Sequence, Union

import numpy as np


def _validate_immutable_numeric_scalar(value: object, name: str) -> None:
    """Require supported immutable real scalar storage without numerical coercion.

    Parameters
    ----------
    value : object
        Python/NumPy integer, floating or Boolean scalar.
    name : str
        Configuration field identified in the error.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If storage is a container, complex or another unsupported representation.

    Notes
    -----
    This checks storage/type only. Existing range/order rules still apply;
    accepted scalar values and precision are retained unchanged.
    """
    if not isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
        raise ValueError(f"{name} must be an immutable Python or NumPy real numeric scalar.")


def _validate_reporting_scalar(value: object, name: str) -> None:
    """Require a finite nonnegative real scalar without coercing its arithmetic.

    Parameters
    ----------
    value : object
        Python/NumPy integer, floating or Boolean scalar. Boolean 0/1 values
        retain their existing numeric meaning. Containers are not scalars here.
    name : str
        Input name used in the diagnostic.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the value is not a supported real scalar, is negative or is nonfinite.

    Notes
    -----
    Integers are finite without conversion to float, including large Python ints.
    Validation preserves the supplied value and dtype. Each consumer applies
    its own documented comparison contract.
    """
    message = f"{name} must be a finite, nonnegative real scalar."
    if not isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
        raise ValueError(message)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(message)
    if value < 0:
        raise ValueError(message)


def _validate_reporting_values(values: np.ndarray, name: str) -> None:
    """Check real numeric values without casting complex or object data to float.

    Parameters
    ----------
    values : np.ndarray
        Array view used for validation.
    name : str
        Diagnostic input name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any element is unsupported, nonfinite, complex or negative.
    """
    if values.dtype.kind == "O":
        for value in values.flat:
            _validate_reporting_scalar(value, name)
        return
    if values.dtype.kind not in "buif":
        raise ValueError(f"{name} must contain finite, nonnegative real values.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite, nonnegative real values.")


def _reporting_array(values: object, name: str) -> np.ndarray:
    """Obtain a validated array without dtype promotion or unknown-value loss.

    Parameters
    ----------
    values : object
        Numeric array, scalar or flat sequence. Native arrays may have any shape;
        individual reporting functions impose their own shape requirements.
    name : str
        Diagnostic input name.

    Returns
    -------
    np.ndarray
        Array preserving the supplied array dtype; the input is not mutated.

    Raises
    ------
    ValueError
        If values are unsupported or include invalid/unknown numeric entries.

    Notes
    -----
    Unknown masked values are rejected before conversion can discard metadata.
    Flat sequences are checked before conversion, including masked/complex
    elements. Explicit numeric ndarrays support multidimensional VRS conversion.
    """
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} contains masked/unknown values; select or resolve them explicitly.")
    if isinstance(values, Sequence):
        for value in values:
            _validate_reporting_scalar(value, name)
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite, nonnegative real values.") from error
    _validate_reporting_values(array, name)
    return array


def _probability_ulp(value: object) -> float:
    """Bound one representable step at the bin's precision and validation precision.

    Parameters
    ----------
    value : object
        Validated real probability in [0,1].

    Returns
    -------
    float
        Greater of the native floating spacing and float64 checking spacing.

    Notes
    -----
    Integers/Booleans have no native rounding allowance. Float64 checking still
    accounts for conversion/summation. Harmless subnormal spacing calculations
    are locally permitted; caller error settings and histogram arithmetic remain
    unchanged.
    """
    with np.errstate(under="ignore"):
        native = float(np.spacing(value)) if isinstance(value, (float, np.floating)) else 0.0
        checking = float(np.spacing(float(value)))
    return max(native, checking)


def _validate_probability_histogram(histogram: object) -> None:
    """Require six normalized probabilities within a documented rounding budget.

    Parameters
    ----------
    histogram : object
        One-dimensional six-bin distribution. Its original container/dtype is
        retained for the subsequent Grade/Spread calculation.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If shape, entries or the total of the proportions are invalid.

    Notes
    -----
    The total is checked with math.fsum after temporary float64 conversion.
    The absolute allowance is two ULPs per bin at the coarser of its native and
    float64 checking precision, plus two float64 epsilons for summation:
    2*fsum(ulp(bin)) + 2*eps64. This permits limited representation error, not
    arbitrary missing or excess proportions. Values are never renormalized.
    Compensated summation is only a validity check. The Grade/Spread kernel
    separately evaluates the supplied proportions with its declared cumulative
    precision and strict comparison; this allowance is not a cutoff tolerance.

    Examples
    --------
    >>> _validate_probability_histogram(np.array([.7, .2, .1, 0, 0, 0]))
    >>> _validate_probability_histogram(np.full(6, 1/6, dtype=np.float32))
    """
    array = _reporting_array(histogram, "histogram")
    if array.shape != (6,):
        raise ValueError("histogram must be a one-dimensional six-bin probability distribution.")
    if np.any(array > 1):
        raise ValueError("histogram probabilities must be in [0, 1].")
    total = math.fsum(float(value) for value in histogram)
    tolerance = 2 * math.fsum(_probability_ulp(value) for value in histogram)
    tolerance += 2 * float(np.finfo(np.float64).eps)
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            "histogram probabilities must sum to 1 within the documented rounding allowance."
        )


def _validate_boolean_mask(mask: Union[np.ndarray, Sequence[bool]]) -> np.ndarray:
    """Return a nonempty one-dimensional Boolean selection without dtype coercion.

    Parameters
    ----------
    mask : np.ndarray or sequence of bool
        Boolean array or flat sequence of Python/NumPy Boolean scalars.
        Masked arrays are supported only when no entries are masked.

    Returns
    -------
    np.ndarray
        Canonical Boolean array to use for all indexing and counting.
        The caller's input is not mutated; array inputs may be shared.

    Raises
    ------
    ValueError
        If the mask is empty, not one-dimensional, has a non-Boolean dtype or
        element, cannot be converted to an array, or has masked/unknown entries.

    Notes
    -----
    Check unknown entries before array conversion can discard their metadata.
    Sequence elements must already be Boolean scalars; this also prevents
    converting masked sequence elements into NaN. No truthiness conversion,
    dtype coercion or flattening is performed. This helper does not choose
    window length, require any True entries, or compose previous selections.

    Examples
    --------
    >>> _validate_boolean_mask((True, False, True))
    array([ True, False,  True])
    >>> _validate_boolean_mask([False, False]).dtype == np.dtype(bool)
    True
    """
    message = (
        "Mask must be a one-dimensional Boolean selection "
        "(True=include, False=exclude). Numeric indices and non-Boolean "
        "entries are not accepted; construct a Boolean mask explicitly."
    )
    if np.ma.isMaskedArray(mask) and np.any(np.ma.getmaskarray(mask)):
        raise ValueError(
            "Mask cannot contain masked/unknown entries; resolve the selection explicitly."
        )
    if isinstance(mask, Sequence) and any(
        not isinstance(value, (bool, np.bool_)) for value in mask
    ):
        raise ValueError(message)
    try:
        array = np.asarray(mask)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if array.ndim != 1:
        raise ValueError(message)
    if array.size == 0:
        raise ValueError("Mask cannot be empty.")
    if array.dtype != np.dtype(bool):
        raise ValueError(message)
    return array


def _validated_display_scalar(value: object, name: str) -> float:
    """Check a reporting scalar without changing its stored value or dtype.

    Parameters
    ----------
    value : object
        Required numeric scalar.
    name : str
        Field name for diagnostics.

    Returns
    -------
    float
        Temporary finite float64-compatible value for checks and formatting.

    Raises
    ------
    ValueError
        If the scalar is invalid or cannot be represented finitely for reporting.

    Notes
    -----
    The result caches are never assigned this temporary conversion. Native mean
    dtype is retained for category lookup and VRS correspondence.
    """
    _validate_reporting_scalar(value, name)
    try:
        checked = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as a finite reporting scalar.") from error
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be representable as a finite reporting scalar.")
    return checked


def _validate_recalculation_option(value: object) -> None:
    """Require a Boolean scalar without interpreting arbitrary truthy inputs.

    Parameters
    ----------
    value : object
        Candidate recalculation request.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the request is not a Python or NumPy Boolean scalar.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("recalculate_grade_spread must be a Boolean scalar.")


def _validate_component_weight(k: object) -> None:
    """Check the scalar weight while preserving its existing arithmetic type.

    Parameters
    ----------
    k : object
        Candidate finite Python or NumPy real scalar in [0, 1].
        Boolean endpoint aliases retain their existing numeric interpretation.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the weight has unsupported storage/type or is nonfinite/out of range.
    """
    message = "k must be a finite real scalar in [0, 1]."
    if not isinstance(k, (int, float, bool, np.integer, np.floating, np.bool_)):
        raise ValueError(message)
    if isinstance(k, np.generic) and k.dtype.kind not in "buif":
        raise ValueError(message)
    if isinstance(k, (float, np.floating)) and not np.isfinite(k):
        raise ValueError(message)
    if not 0 <= k <= 1:
        raise ValueError(f"k must be in [0, 1], got {k}")


def _validate_epsilon(value: object, name: str = "epsilon_floor") -> None:
    """Require a positive finite real scalar usable by the float64 pipeline.

    Parameters
    ----------
    value : object
        Python int/float or NumPy integer/floating scalar to validate.
        Booleans, complex values, arrays and other containers are not supported.
    name : str, optional
        Parameter name included in the error message.

    Returns
    -------
    None
        Accepted values are not replaced, clamped or coerced for computation.

    Raises
    ------
    ValueError
        If the type is unsupported or float64 conversion is nonfinite/nonpositive.

    Notes
    -----
    A temporary Python float checks range only; the caller retains the original
    scalar. This does not alter the absolute-floor formula or make it scale invariant.

    Examples
    --------
    >>> _validate_epsilon(1e-10)
    >>> _validate_epsilon(float('inf'))
    Traceback (most recent call last):
        ...
    ValueError: epsilon_floor must be a finite, strictly positive real scalar within the float64 range.
    """
    message = f"{name} must be a finite, strictly positive real scalar within the float64 range."
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(message)
    try:
        checked = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(message) from error
    if not math.isfinite(checked) or checked <= 0:
        raise ValueError(message)


def _validate_no_masked_values(values: object, name: str) -> None:
    """Reject unknown NumPy masked entries before input conversion.

    Parameters
    ----------
    values : object
        Raw input, including NumPy arrays and Python list/tuple containers.
    name : str
        Input name included in the diagnostic.

    Returns
    -------
    None
        Fully known inputs are not converted or modified.

    Raises
    ------
    ValueError
        If a NumPy masked array or masked scalar has any unknown entries,
        including entries nested in lists, tuples or object arrays.

    Notes
    -----
    Fully unmasked wrappers retain subsequent dtype, shape and value checks.
    Numeric ndarrays use their mask metadata without scanning their values.
    Object containers are traversed iteratively. Retaining visited containers
    avoids repeated traversal, recursion and reuse of temporary object identities.
    This helper neither resolves missing data nor implements foreign missing-data
    protocols. Other unsupported inputs remain the caller's validation concern.

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv._internal.value_validation import _validate_no_masked_values
    >>> _validate_no_masked_values(np.ma.array([1., 2.], mask=False), "data")
    >>> _validate_no_masked_values([1., np.ma.masked], "data")
    Traceback (most recent call last):
        ...
    ValueError: data contains masked/unknown values; resolve missing samples explicitly before comparison.
    """
    end = object()
    pending = [iter((values,))]
    visited = {}
    while pending:
        value = next(pending[-1], end)
        if value is end:
            pending.pop()
            continue
        if np.ma.isMaskedArray(value) and np.ma.is_masked(value):
            raise ValueError(
                f"{name} contains masked/unknown values; "
                "resolve missing samples explicitly before comparison."
            )
        if isinstance(value, np.ndarray):
            if value.dtype.kind != "O":
                continue
            children = value.flat
        elif isinstance(value, (list, tuple)):
            children = iter(value)
        else:
            continue
        identity = id(value)
        if identity in visited:
            continue
        visited[identity] = value
        pending.append(iter(children))
