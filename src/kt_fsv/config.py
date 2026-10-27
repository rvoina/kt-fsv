"""
FSV configuration dataclass.

Provides reference defaults and experimental overrides for FSV analysis.
Using defaults alone does not establish IEEE Std 1597.1-2022 conformance.
"""

import warnings
from dataclasses import dataclass, field, fields
from typing import Dict, Iterable, Literal, Mapping, Tuple, Union

import numpy as np

from . import constants
from ._internal.category_rules import (
    _category_index,
    _category_less,
    _category_ratio,
    _validate_category_limit_values,
    _validated_category_limits,
    _validated_category_names,
)
from ._internal.value_validation import (
    _validate_epsilon,
    _validate_immutable_numeric_scalar,
)


# =============================================================================
# Default Constants (IEEE Std 1597.1-2022)
# =============================================================================

# Default epsilon floor for numerical stability in ADM/FDM calculations
DEFAULT_EPSILON_FLOOR: float = constants.DEFAULT_EPSILON_FLOOR
DEFAULT_BREAK_POINT_DC: int = constants.DEFAULT_BREAK_POINT_DC

# Default category thresholds per IEEE Std 1597.1-2022 Table 3
DEFAULT_CATEGORY_THRESHOLDS: Dict[str, float] = constants.DEFAULT_CATEGORY_THRESHOLDS


class _ReadOnlyCategoryThresholds(dict):
    """Own dictionary-compatible category data with no ordinary mutation paths.

    Parameters
    ----------
    values : Mapping or iterable of (str, float)
        Input to copy, including the iterable used by dataclasses.asdict.

    Returns
    -------
    _ReadOnlyCategoryThresholds
        Read-only snapshot retaining input order and native scalar values.

    Raises
    ------
    ValueError
        If a value is not a supported immutable real scalar.
    TypeError
        If ordinary mutation or reinitialization is attempted.

    Notes
    -----
    This private dict-compatible representation supports standard dataclass
    replacement/export, copying and same-version pickle round trips. It is not
    a security boundary against explicit low-level Python bypasses.

    Examples
    --------
    >>> from dataclasses import replace
    >>> original = FSVConfig()
    >>> limits = dict(original.category_thresholds)
    >>> limits["Good"] = 0.6
    >>> updated = replace(original, category_thresholds=limits)
    >>> original.category_thresholds["Good"], updated.category_thresholds["Good"]
    (0.4, 0.6)
    >>> updated.category_thresholds["Good"] = 0.7
    Traceback (most recent call last):
        ...
    TypeError: category_thresholds are read-only; create a new FSVConfig.
    """

    __slots__ = ("_sealed",)

    def __init__(
        self, values: Union[Mapping[str, float], Iterable[Tuple[str, float]]],
    ) -> None:
        """Copy and seal category storage.

        Parameters
        ----------
        values : Mapping or iterable of pairs
            Values to snapshot without dtype conversion.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If this instance is already initialized.
        ValueError
            If a category value is an unsupported scalar representation.
        """
        if getattr(self, "_sealed", False):
            raise TypeError("category_thresholds are read-only; create a new FSVConfig.")
        snapshot = dict(values)
        for name, value in snapshot.items():
            _validate_immutable_numeric_scalar(value, f"category_thresholds[{name!r}]")
        dict.__init__(self, snapshot)
        object.__setattr__(self, "_sealed", True)

    def _deny_mutation(self, *args: object, **kwargs: object) -> None:
        """Reject a write before it can alter the snapshot.

        Parameters
        ----------
        *args, **kwargs : object
            Arguments of the blocked dictionary/attribute mutation.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            Always; an explicit new configuration is required.
        """
        raise TypeError("category_thresholds are read-only; create a new FSVConfig.")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _deny_mutation
    __setattr__ = __delattr__ = _deny_mutation

    def __copy__(self) -> "_ReadOnlyCategoryThresholds":
        """Return this immutable snapshot for shallow copying.

        Returns
        -------
        _ReadOnlyCategoryThresholds
            The unchanged read-only value.
        """
        return self

    def __deepcopy__(self, memo: Dict[int, object]) -> "_ReadOnlyCategoryThresholds":
        """Retain immutable scalar values during deep copying.

        Parameters
        ----------
        memo : dict
            Standard copy-protocol memo; no mutable child values require copying.

        Returns
        -------
        _ReadOnlyCategoryThresholds
            The unchanged read-only value.
        """
        return self

    def __reduce__(self) -> Tuple[object, Tuple[Dict[str, float]]]:
        """Reconstruct a sealed snapshot for same-version pickle compatibility.

        Returns
        -------
        tuple
            Constructor and a detached ordinary dictionary argument.

        Notes
        -----
        This does not define a stable cross-version archival format.
        """
        return type(self), (dict(self),)


def _same_config_setting(first: object, second: object) -> bool:
    """Compare a supported setting without weak scalar promotion.

    Parameters
    ----------
    first, second : object
        Validated scalar settings or category mappings.

    Returns
    -------
    bool
        Whether scalar types and exact values agree. Mapping insertion order
        is ignored; corresponding category values retain the same scalar rule.

    Notes
    -----
    This is configuration identity, not approximate numerical equivalence.
    Distinct NaN-valued settings do not match. No values are coerced or changed.
    """
    if isinstance(first, Mapping):
        if not isinstance(second, Mapping) or first.keys() != second.keys():
            return False
        return all(
            type(first[name]) is type(second[name]) and bool(first[name] == second[name])
            for name in first
        )
    return type(first) is type(second) and bool(first == second)


@dataclass(frozen=True, eq=False)
class FSVConfig:
    """
    Configuration for FSV algorithm parameters.

    Defaults follow the reference configuration used by this implementation.
    They do not certify conformance of a complete analysis. Modifications
    should be documented in validation reports with technical justification.

    Parameters
    ----------
    normalization : {'none', 'N', 'sqrt_N'}, optional
        Forward FFT divisor: 1 ('none'), original N ('N'), or sqrt(N)
        ('sqrt_N', the default), with matching inverse scaling. Only sqrt_N
        equates the unweighted input/spectral sums of squares when no padding
        is used; single-sided amplitude doubling is not applied.
    zero_pad : bool, optional
        Whether to zero-pad the data to next power of 2. Default False.
        IEEE 8.4(b2) says this working-data enlargement shall be avoided.
        It is separate from zero-filling coefficients for inverse reconstruction.
    break_point_dc : int, optional
        Number of nonnegative-frequency bins in the DC region, including
        bin zero (indices 0 to break_point_dc-1). Default: 5, the reference
        setting for IEEE Std 1597.1-2022. Other positive integer settings
        are experimental overrides and issue UserWarning on construction.
        A value of 1 retains only true DC; mirrored nonzero-frequency bins
        are included for larger values.
    threshold_lo_hi : float, optional
        Cumulative spectral-magnitude fraction for the Lo/Hi breakpoint.
        Must be in (0, 1). Default: 0.4. The adopted profile uses undoubled unique FFT bins from
        break_point_dc-1 through floor(N_fft/2), including the even Nyquist
        endpoint, and selects the first cumulative sum at least this fraction
        of the total. It does not guarantee a 40:60 filtered-energy partition.
    compute_grade_spread : bool, optional
        Whether to compute the category-distribution Grade and Spread
        descriptors in IEEE section 8.5(b). Default: True. When False, their
        result fields are None and weighted GDM is not computed. Standard
        category histograms are still computed.
    use_weighted_gdm : bool, optional
        Whether to compute weighted GDM per IEEE Std 1597.1-2022 §8.5(b).
        Default: False. Spread-based coefficients reduce FDM when ADM Spread
        is smaller and amplify ADM when ADM Spread is larger. The weighted
        score is not necessarily lower than the base GDM. Both this setting
        and compute_grade_spread must be True to produce weighted output;
        base GDM remains available and unchanged by this output option.
    grade_spread_threshold : float, optional
        Strict cumulative threshold for Grade and Spread metrics (§8.5(b)).
        Must be in (0,1); including all analyzed points gives a cumulative
        proportion of one, which cannot be strictly exceeded.
        Default: 0.85 (85% per IEEE recommendation). The represented scalar
        value is retained rather than rounded to a displayed percentage.
        This is a category-proportion threshold, not a statistical confidence level.
    category_thresholds : Mapping, optional
        Exclusive upper limits for the configured scalar interpretation. Keys must be exactly
        Excellent, Very Good, Good, Fair, Poor and Very Poor; insertion order
        is ignored. The first five limits must be positive, finite and strictly
        increasing by actual numeric value; Very Poor must end at +inf.
        Invalid names or limits raise ValueError before classification.
        Custom limits affect the existing result category properties. Standard
        category properties, histograms, Grade/Spread and Table5 VRS use the
        canonical scale; reports identify both interpretations for custom limits.
        The mapping is defensively copied and stored read-only. Bounds must be
        supported immutable Python/NumPy real numeric scalars; their native
        types and values are preserved. Use dict(config.category_thresholds)
        to obtain an editable copy for a new configuration.
        Default: reference values from Table 3.
    epsilon_floor : float, optional
        Absolute denominator floor for ADM/FDM division. Default: 1e-10.
        Must be a Python or NumPy integer/floating scalar whose float64 conversion
        is finite and strictly positive. Booleans, complex values, strings and
        arrays (including zero-dimensional arrays) are rejected. Accepted values
        are retained without replacement. Small positive denominators are also
        floored; validation does not remove the resulting unit/scale dependence.

    Returns
    -------
    FSVConfig
        Configuration with immutable scalar settings and a detached, read-only
        category mapping.

    Raises
    ------
    ValueError
        If configuration validation fails, including an invalid epsilon value/type.

    Warns
    -----
    UserWarning
        After successful validation, when break_point_dc differs from 5.
        Users must document and justify the custom setting and validate its
        effect. The warning does not replace input validation or certify
        the remaining configuration.

    Notes
    -----
    Scalar settings reject containers, including zero-dimensional arrays.
    Supported Python/NumPy scalar values are retained without promotion; this
    ownership check does not change existing range, order or flag semantics.
    Category snapshots are read-only under ordinary dictionary operations.
    Create a new configuration with dataclasses.replace or a constructor for
    intentional updates; existing analyses retain their original configuration.

    Standard copying and same-version pickle round trips preserve read-only
    storage. dataclasses.asdict produces a detached dict-compatible nested
    snapshot; use dict(...) when an editable mapping is needed. No strict JSON,
    hashability or cross-version pickle guarantee is introduced. Reconstruct
    legacy/unvalidated configurations through the current constructor before
    relying on the ownership contract.

    Configuration ownership is separate from result snapshot protection and
    cache consistency. It does not support concurrent engine reconfiguration.
    The mutable legacy default lookup in constants retains its existing
    compatibility policy.

    Record changed settings, their rationale and their observed effects in the
    study report. Package choices such as FFT normalization, optional output
    switches and the absolute epsilon floor are not additional IEEE requirements.

    The DC warning is issued during construction, not on each analysis that
    reuses this configuration. Its display follows Python warning filters;
    applications treating UserWarning as an error must handle the deviation
    explicitly. Other default changes do not trigger this DC-specific warning.

    Examples
    --------
    >>> import warnings
    >>> from kt_fsv import FSV, FSVConfig
    >>> # Use the package reference configuration
    >>> config = FSVConfig()
    >>> fsv = FSV(config=config)
    >>>
    >>> # Change the cumulative spectral-magnitude target, not an energy partition
    >>> config = FSVConfig(threshold_lo_hi=0.5)
    >>> fsv = FSV(config=config)
    >>>
    >>> # Record the expected warnings for two experimental DC settings
    >>> with warnings.catch_warnings(record=True) as diagnostics:
    ...     warnings.filterwarnings(
    ...         "always", message=r"^break_point_dc=(7|10) differs ",
    ...         category=UserWarning,
    ...     )
    ...     wider_dc = FSVConfig(break_point_dc=10)
    ...     custom = FSVConfig(
    ...         break_point_dc=7, threshold_lo_hi=0.45,
    ...         grade_spread_threshold=0.90,
    ...     )
    >>> len(diagnostics)
    2
    >>> fsv = FSV(config=custom)
    >>>
    >>> # Enable weighted GDM
    >>> config = FSVConfig(use_weighted_gdm=True)
    >>> fsv = FSV(config=config)
    >>>
    >>> # Nondefault transform settings; padding warns when an analysis is run
    >>> config = FSVConfig(normalization='N', zero_pad=True)
    >>> fsv = FSV(config=config)

    References
    ----------
    IEEE Std 1597.1-2022: Standard for Validation of Computational
    Electromagnetics Computer Modeling and Simulations
    """

    normalization: Literal["none", "N", "sqrt_N"] = constants.DEFAULT_NORMALIZATION
    zero_pad: bool = constants.DEFAULT_ZERO_PAD
    break_point_dc: int = DEFAULT_BREAK_POINT_DC
    threshold_lo_hi: float = constants.DEFAULT_THRESHOLD_LO_HI
    compute_grade_spread: bool = constants.DEFAULT_COMPUTE_GRADE_SPREAD
    use_weighted_gdm: bool = constants.DEFAULT_USE_WEIGHTED_GDM
    grade_spread_threshold: float = constants.DEFAULT_GRADE_SPREAD_THRESHOLD
    category_thresholds: Mapping[str, float] = field(
        default_factory=lambda: dict(constants.DEFAULT_CATEGORY_ITEMS)
    )
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        """Compare exact configuration values and their scalar representations.

        Parameters
        ----------
        other : object
            Configuration of the same concrete class, or an unrelated value.

        Returns
        -------
        bool or NotImplemented
            True for self identity or matching setting types/values.
            NotImplemented for a different concrete class, allowing Python's
            ordinary reflected-comparison protocol.

        Notes
        -----
        Mapping order is irrelevant. Disabled options still participate.
        Across distinct configurations, a NaN setting never establishes equality;
        self identity does not certify configuration validity. Same-type signed
        zeros retain ordinary numeric equality. Instances remain unhashable.

        This policy does not replace the value-only compatibility checks for
        individual confidence thresholds during combination or plotting.

        Examples
        --------
        >>> from dataclasses import replace
        >>> import numpy as np
        >>> from kt_fsv import FSVConfig
        >>> default = FSVConfig()
        >>> default == replace(default)
        True
        >>> native = replace(default, grade_spread_threshold=np.float32(0.85))
        >>> default == native
        False
        >>> native == replace(default, grade_spread_threshold=float(np.float32(0.85)))
        False
        >>> native == replace(native)
        True
        """
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return all(
            _same_config_setting(getattr(self, setting.name), getattr(other, setting.name))
            for setting in fields(self)
        )

    def __post_init__(self) -> None:
        """
        Validate configuration before warning about a custom DC setting.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a parameter fails configuration validation.

        Warns
        -----
        UserWarning
            If validation succeeds and break_point_dc differs from 5.
        """
        self._validate_scalar_storage()
        # Validate normalization
        valid_normalizations = ["none", "N", "sqrt_N"]
        if self.normalization not in valid_normalizations:
            raise ValueError(
                f"normalization must be one of {valid_normalizations}, "
                f"got '{self.normalization}'"
            )

        if self.break_point_dc < 1:
            raise ValueError(
                f"break_point_dc must be >= 1, got {self.break_point_dc}"
            )

        if not 0.0 < self.threshold_lo_hi < 1.0:
            raise ValueError(
                f"threshold_lo_hi must be in (0, 1), got {self.threshold_lo_hi}"
            )

        if not 0.0 < self.grade_spread_threshold < 1.0:
            raise ValueError(
                f"grade_spread_threshold must be in (0, 1), "
                f"got {self.grade_spread_threshold}"
            )

        _validate_epsilon(self.epsilon_floor)

        if not isinstance(self.category_thresholds, Mapping):
            raise ValueError("category_thresholds must be a mapping")

        snapshot = _ReadOnlyCategoryThresholds(self.category_thresholds)
        required_categories = _validated_category_names(snapshot)

        thresholds = [
            snapshot[cat]
            for cat in required_categories[:-1]  # Exclude "Very Poor" (inf)
        ]
        _validate_category_limit_values(thresholds, snapshot[required_categories[-1]])

        object.__setattr__(self, "category_thresholds", snapshot)
        self._warn_nonstandard_dc()

    def _validate_scalar_storage(self) -> None:
        """Reject mutable containers in fields whose contracts require scalars.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If normalization is not a string scalar or a numerical/switch field
            uses an unsupported representation, including zero-dimensional arrays.

        Notes
        -----
        No flag coercion or new numerical boundary policy is introduced.
        Epsilon retains its separately approved, stricter scalar validation.

        Examples
        --------
        >>> FSVConfig(zero_pad=np.array(False))
        Traceback (most recent call last):
            ...
        ValueError: zero_pad must be an immutable Python or NumPy real numeric scalar.
        """
        if not isinstance(self.normalization, (str, np.str_)):
            raise ValueError("normalization must be an immutable string scalar.")
        for name in (
            "zero_pad", "break_point_dc", "threshold_lo_hi", "compute_grade_spread",
            "use_weighted_gdm", "grade_spread_threshold",
        ):
            _validate_immutable_numeric_scalar(getattr(self, name), name)

    def _warn_nonstandard_dc(self) -> None:
        """Report a DC-setting deviation at the configuration caller.

        Returns
        -------
        None

        Warns
        -----
        UserWarning
            When the accepted DC setting differs from the reference value.

        Notes
        -----
        The stack level accounts for this helper, __post_init__ and the
        generated dataclass constructor before reaching a direct caller.
        """
        if self.break_point_dc == DEFAULT_BREAK_POINT_DC:
            return
        warnings.warn(
            f"break_point_dc={self.break_point_dc} differs from the IEEE Std "
            f"1597.1-2022 DC setting of {DEFAULT_BREAK_POINT_DC}. "
            "This configuration uses a modified FSV procedure. Document and justify "
            "this choice and validate its effect before relying on the results.",
            UserWarning,
            stacklevel=4,
        )
