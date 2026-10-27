"""
FSV Analysis results container.

Contains the FSVAnalysis class - the primary output of FSV analysis
that provides results and debug data. Visualization is provided through
the separate FSVVisualizer class accessible via the .visualizer property.
"""

from dataclasses import MISSING, InitVar, dataclass, fields
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, ClassVar, Tuple, Type

import numpy as np

import kt_fsv

if TYPE_CHECKING:
    from .visualization import FSVVisualizer
    from .config import FSVConfig


# =============================================================================
# Import default thresholds from config
# =============================================================================

from ._internal.result_storage import (
    _SnapshotT,
    _check_snapshot_array,
    _freeze_snapshot_arrays,
    _from_owned_snapshot,
    _readonly_snapshot_buffer,
    _snapshot_array,
    _snapshot_getattribute,
    _validate_snapshot_storage,
)
from ._internal.value_validation import (
    _validate_boolean_mask,
    _validated_display_scalar,
)
from .config import (
    DEFAULT_CATEGORY_THRESHOLDS, FSVConfig, _ReadOnlyCategoryThresholds,
    _category_index, _category_less, _category_ratio,
    _validate_epsilon, _validated_category_limits,
)
from .constants import DEFAULT_CATEGORY_ITEMS
from .measures import (
    _reporting_array,
    _validate_probability_histogram,
    _validate_reporting_scalar,
    convert_to_vrs,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_cached_vrs(mean: object, visual: object, name: str) -> None:
    """Check the deterministic mean/VRS relationship without repairing the cache.

    Parameters
    ----------
    mean : object
        Validated mean, retaining its original numeric type.
    visual : object
        Stored rating to check.
    name : str
        Rating field name for diagnostics.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the stored rating is invalid or disagrees with the mean's conversion.

    Notes
    -----
    Use the existing converter with the mean's native dtype, not a promoted mean.
    Comparison permits eight float64 output ULPs for final rounding. No hard cap
    of six is applied. This checks one scalar relationship, not correspondence
    between all cached statistics and the underlying active point arrays.

    Examples
    --------
    >>> _validate_cached_vrs(0.15, 2.51, "gdm_visual")
    >>> _validate_cached_vrs(np.float32(1.6), 6.001250267028809, "gdm_visual")
    """
    actual = _validated_display_scalar(visual, name)
    expected = float(convert_to_vrs(np.asarray(mean)))
    allowance = 8 * float(np.spacing(expected))
    if abs(actual - expected) > allowance:
        raise ValueError(f"{name} is inconsistent with the stored mean; recompute the result explicitly.")


def get_category(value: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """
    Get FSV category for a given value.

    Uses thresholds from DEFAULT_CATEGORY_THRESHOLDS per IEEE Std 1597.1-2022,
    or custom thresholds if provided.

    Parameters
    ----------
    value : float
        Finite nonnegative real FSV measure scalar (ADM, FDM or GDM mean).
        Python/NumPy integer and floating values are compared without narrowing
        either operand. Numeric Boolean 0/1 values retain their meaning.
    thresholds : dict, optional
        Custom category upper thresholds. If None, uses the reference defaults.
        Keys must be exactly Excellent, Very Good, Good, Fair, Poor and
        Very Poor. Insertion order is ignored; numerical limits come from
        this mapping rather than the default values. First-five limits must
        be positive, finite and strictly increasing; Very Poor must have
        positive infinity as its upper marker.

    Returns
    -------
    str
        Category name: 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor', 'Very Poor'

    Raises
    ------
    ValueError
        If the value is invalid, nonscalar, negative, nonfinite or complex,
        or the supplied category names or numeric limits are invalid.
        An empty mapping is invalid; use None for the default thresholds.

    Notes
    -----
    The complete scale is checked before classification. Comparisons retain
    the strict upper-bound '<' rule and use actual represented values.
    The supplied mapping is not reordered or otherwise modified.
    Positive infinity is the required final cutoff; it is not accepted as a
    measured score. There is no clipping or automatic conversion of missing scores into
    an ordinary category.
    Standard limits use their stored float64 values; equality belongs to the
    next category, including Very Poor at the stored 1.6 cutoff. A float16
    value displayed as 1.6 is actually 1.599609375 and therefore belongs to Poor.
    This is the declared interval/precision profile, not a claim that Table3
    uniquely specifies all machine endpoints. None retains the live legacy
    default lookup; standard result properties use immutable canonical limits.

    Examples
    --------
    >>> get_category(0.15)
    'Very Good'
    >>> get_category(np.float16(1.6))
    'Poor'
    >>> get_category(float("nan"))
    Traceback (most recent call last):
        ...
    ValueError: value must be a finite, nonnegative real scalar.
    """
    _validate_reporting_scalar(value, "value")
    if thresholds is None:
        thresholds = DEFAULT_CATEGORY_THRESHOLDS
    limits = _validated_category_limits(thresholds)
    for (category, _), limit in zip(DEFAULT_CATEGORY_ITEMS, limits):
        if _category_less(value, limit):
            return category
    return "Very Poor"  # Fallback (should never reach due to inf threshold)


def _get_standard_category(value: object) -> str:
    """Classify a score using the immutable standard scale.

    Parameters
    ----------
    value : object
        Finite nonnegative Python/NumPy real scalar.

    Returns
    -------
    str
        Canonical category, independent of custom or legacy lookup settings.

    Raises
    ------
    ValueError
        If the score is invalid.
    """
    _validate_reporting_scalar(value, "value")
    index = _category_index(value, tuple(upper for _, upper in DEFAULT_CATEGORY_ITEMS))
    return DEFAULT_CATEGORY_ITEMS[index][0]


# =============================================================================
# Debug Data
# =============================================================================

def _validate_snapshot_config(config: Optional[FSVConfig]) -> None:
    """Require current immutable configuration storage without rebuilding history.

    Parameters
    ----------
    config : FSVConfig or None
        Recorded settings, or unavailable metadata.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If metadata is mutable, unvalidated or unsupported.
    """
    if config is None:
        return
    message = "config must be a current immutable FSVConfig or None; reconstruct legacy settings explicitly."
    if not isinstance(config, FSVConfig):
        raise ValueError(message)
    try:
        if not isinstance(config.category_thresholds, _ReadOnlyCategoryThresholds):
            raise ValueError(message)
        config._validate_scalar_storage()
        _validate_epsilon(config.epsilon_floor)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(message) from error


def _validate_snapshot_metadata(record: _SnapshotT, *, sealed_mask: bool = False) -> None:
    """Require immutable configuration/debug storage and a primitive mask.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Record whose nonnumerical references must be stable.
    sealed_mask : bool, optional
        Require the constructed ndarray mask rather than an incoming sequence.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If result metadata can retain mutable unsupported storage.
    """
    if not isinstance(record, FSVAnalysis):
        return
    _validate_snapshot_config(record.config)
    if record.debug is not None and (
        not isinstance(record.debug, FSVDebugData) or not _snapshot_is_ready(record.debug)
    ):
        raise ValueError("debug must be a current immutable FSVDebugData snapshot or None.")
    if sealed_mask and record.active_mask is not None:
        _check_snapshot_array(record.active_mask, "active_mask")
        _validate_boolean_mask(record.active_mask)


def _snapshot_is_ready(record: _SnapshotT) -> bool:
    """Recognize current snapshots for safe sharing/copying, not cache correctness.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Candidate producer/parent record.

    Returns
    -------
    bool
        Whether the snapshot marker, storage and recorded domain are intact.
    """
    if vars(record).get("_snapshot_ready") is not True:
        return False
    if isinstance(record, FSVAnalysis) and "active_mask" not in vars(record):
        return False
    try:
        _validate_snapshot_storage(record)
        _validate_snapshot_metadata(record, sealed_mask=True)
    except ValueError:
        return False
    return all(
        _readonly_snapshot_buffer(value)
        for value in vars(record).values() if isinstance(value, np.ndarray)
    )


def _copy_snapshot(record: _SnapshotT) -> _SnapshotT:
    """Return the same supported immutable snapshot.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Current snapshot.

    Returns
    -------
    FSVAnalysis or FSVDebugData
        The same object; copy is not an editing escape hatch.

    Raises
    ------
    ValueError
        If legacy/modified storage is not a current snapshot.
    """
    if not _snapshot_is_ready(record):
        raise ValueError("Reconstruct a current snapshot with known selection before copying.")
    return record


def _deepcopy_snapshot(record: _SnapshotT, memo: Dict[int, Any]) -> _SnapshotT:
    """Retain immutable identity and register it in the copy memo.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Current snapshot.
    memo : dict
        Standard deepcopy memo.

    Returns
    -------
    FSVAnalysis or FSVDebugData
        Same immutable object.

    Raises
    ------
    ValueError
        If the record cannot safely retain identity.
    """
    result = _copy_snapshot(record)
    memo[id(record)] = result
    return result


def _restore_snapshot(snapshot_type: Type[_SnapshotT], state: Dict[str, Any]) -> _SnapshotT:
    """Restore supported state through defensive public construction.

    Parameters
    ----------
    snapshot_type : type
        Result/debug class.
    state : dict
        Stored public fields, with explicit selection for an analysis.

    Returns
    -------
    FSVAnalysis or FSVDebugData
        Independently owned snapshot, with cached values unchanged.

    Raises
    ------
    ValueError
        If state is incompatible, incomplete or lacks selection provenance.
    """
    if not isinstance(state, dict):
        raise ValueError("Snapshot state must be a field dictionary.")
    definitions = fields(snapshot_type)
    names = {item.name for item in definitions}
    required = {item.name for item in definitions if item.default is MISSING and item.default_factory is MISSING}
    if issubclass(snapshot_type, FSVAnalysis) and "active_mask" not in state:
        raise ValueError("Snapshot state lacks recorded selection; reconstruct with a known mask.")
    if state.keys() - names or required - state.keys():
        raise ValueError("Snapshot state contains unknown fields or lacks required fields.")
    return snapshot_type(**state)


def _reduce_snapshot(record: _SnapshotT) -> Tuple[Any, Tuple[Type[_SnapshotT], Dict[str, Any]]]:
    """Serialize public fields without the private transfer mode.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Current snapshot.

    Returns
    -------
    tuple
        Defensive restoration callable and its arguments.

    Raises
    ------
    ValueError
        If storage/provenance is not a supported current snapshot.
    """
    _copy_snapshot(record)
    return _restore_snapshot, (type(record), {item.name: getattr(record, item.name) for item in fields(record)})


def _set_snapshot_state(record: _SnapshotT, state: Dict[str, Any]) -> None:
    """Accept compatible legacy field state only through defensive restoration.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Uninitialized object supplied by pickle or a restoration caller.
    state : dict
        Public field state.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the target is already initialized or state cannot be restored safely.
    """
    if vars(record):
        raise ValueError("Snapshot restoration requires an uninitialized target.")
    restored = _restore_snapshot(type(record), state)
    for item in fields(restored):
        object.__setattr__(record, item.name, getattr(restored, item.name))
    object.__setattr__(record, "_snapshot_ready", True)


@dataclass(frozen=True, eq=False)
class FSVDebugData:
    """
    Immutable snapshot of working inputs and intermediate computation arrays.

    Obtain this record from ``analysis.debug`` when the core calculation uses
    ``FSV(store_debug=True)``. Direct construction stores supplied arrays;
    it does not run an FFT or calculate filtered components.

    Parameters
    ----------
    data1, data2 : np.ndarray
        Working inputs. Core-produced records have shape ``(N,)``, where
        ``N = analysis.len_orig``.
    magnitude1, magnitude2 : np.ndarray
        Full, unshifted, undoubled FFT magnitudes under the source normalization.
        Core-produced records have shape ``(analysis.len_fft,)``.
    dc_data1, dc_data2 : np.ndarray
        Inverse-filtered DC components on the working sample domain, shape
        ``(N,)`` for core-produced records.
    lo_data1, lo_data2 : np.ndarray
        Inverse-filtered Lo components on the same working domain.
    hi_data1, hi_data2 : np.ndarray
        Inverse-filtered Hi components on the same working domain.

    Returns
    -------
    FSVDebugData
        Owned, read-only snapshot of the supplied numerical arrays.

    Raises
    ------
    ValueError
        If an array does not use supported primitive numerical ndarray storage.
        Masked arrays and object-containing storage are rejected.

    Notes
    -----
    Fields and numerical buffers are read-only. Public construction copies
    borrowed arrays; core producers copy the two retained inputs and transfer
    their already computed intermediate allocations. Caller input edits cannot
    rewrite this debug history. For fsv_from_xy these are working/resampled
    inputs, not a complete archive of raw coordinates and preprocessing.
    Array access returns a fresh read-only view header; caller shape/dtype
    reinterpretations of that view cannot rewrite stored snapshot metadata.

    Filtered components return to the input's working sample domain, which may
    represent frequency, angle or time; they are not necessarily physical
    time-domain signals. This record contains no coordinate array, complex FFT
    coefficients, filter coefficients or complete processing history.
    Windowing retains the source debug record; component combination supplies
    no combined debug record.

    Construction preserves dtype and shape but does not verify that the supplied
    arrays form a consistent calculation. Copy/deepcopy retain immutable identity;
    equality is identity-based and instances are unhashable. Explicit array
    copies provide editable numerical data. ``dataclasses.asdict`` detaches
    fields but is not a JSON conversion. Same-version pickle restoration makes
    owned read-only buffers; no general cross-version archive contract is given.

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv import FSV
    >>> analysis = FSV(store_debug=True).fsv(np.ones(64), np.full(64, 2.0))
    >>> debug = analysis.debug
    >>> assert debug is not None
    >>> debug.data1.shape, debug.magnitude1.shape
    ((64,), (64,))
    >>> debug.data1.flags.writeable
    False
    >>> editable = debug.data1.copy()
    >>> editable[:] = 0
    >>> bool(np.all(debug.data1 == 1.0))
    True
    """

    data1: np.ndarray
    data2: np.ndarray
    magnitude1: np.ndarray
    magnitude2: np.ndarray
    dc_data1: np.ndarray
    dc_data2: np.ndarray
    lo_data1: np.ndarray
    lo_data2: np.ndarray
    hi_data1: np.ndarray
    hi_data2: np.ndarray

    _adopt_owned_arrays: InitVar[bool] = False
    _snapshot_ready: ClassVar[bool] = False
    _snapshot_array_names: ClassVar[Tuple[str, ...]] = (
        "data1", "data2", "magnitude1", "magnitude2", "dc_data1", "dc_data2",
        "lo_data1", "lo_data2", "hi_data1", "hi_data2",
    )
    __hash__ = None
    __getattribute__ = _snapshot_getattribute
    __copy__ = _copy_snapshot
    __deepcopy__ = _deepcopy_snapshot
    __reduce__ = _reduce_snapshot
    __setstate__ = _set_snapshot_state
    _from_owned_arrays = classmethod(_from_owned_snapshot)

    def __post_init__(self, _adopt_owned_arrays: bool) -> None:
        """Own and seal debug buffers without recalculating their values.

        Parameters
        ----------
        _adopt_owned_arrays : bool
            Private internal-producer mode.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If storage is unsupported.
        """
        _validate_snapshot_storage(self)
        _freeze_snapshot_arrays(self, _adopt_owned_arrays)
        object.__setattr__(self, "_snapshot_ready", True)


# =============================================================================
# Main Analysis Class
# =============================================================================

@dataclass(frozen=True, eq=False)
class FSVAnalysis:
    """
    Immutable result snapshot with summaries and access to a visualizer.

    ``FSV.fsv`` and ``FSV.fsv_from_xy`` produce these records. Direct construction
    stores supplied results; it does not perform FSV or recompute cached values.

    Parameters
    ----------
    adm_point : np.ndarray
        Pointwise ADM on the working sample domain, normally shape ``(N,)``.
    adm_mean : float
        Stored mean ADM over the active points.
    adm_conf_hist : np.ndarray
        Six canonical category proportions, shape ``(6,)``; not integer counts.
    adm_visual : float
        Stored Table-5 VRS conversion of adm_mean; not a category index.
    fdm_point : np.ndarray
        Pointwise FDM on the same working domain.
    fdm_mean : float
        Stored mean FDM over the active points.
    fdm_conf_hist : np.ndarray
        Six canonical FDM category proportions.
    fdm_visual : float
        Stored Table-5 VRS conversion of fdm_mean.
    gdm_point : np.ndarray
        Pointwise base GDM on the same working domain.
    gdm_mean : float
        Stored mean of active pointwise GDM values.
    gdm_conf_hist : np.ndarray
        Six canonical GDM category proportions.
    gdm_visual : float
        Stored Table-5 VRS conversion of gdm_mean. The nominal scale is 1-6;
        literal formula values slightly above six are retained.
    len_orig : int
        Original core working-signal length, retained after postprocessing.
        Use active_count for the current statistical population.
    len_fft : int
        FFT length of the recorded source context (may differ if zero-padded).
        Component combination does not perform a new FFT.
    break_point_lo_hi : int
        Lo/Hi breakpoint of the recorded core/source comparison. An inherited
        value does not define a common breakpoint for combined components.
    config : FSVConfig or None, optional
        Original core or selected-parent configuration. After postprocessing,
        its flags alone do not identify which optional outputs were calculated.
        Default None means unavailable source settings, not reconstructed history.
    adm_grade, fdm_grade, gdm_grade : int or None, optional
        Optional Grade category counts under the operation's strict cumulative
        cutoff. Each defaults to None.
    adm_spread, fdm_spread, gdm_spread : int or None, optional
        Optional Spread category counts. Each defaults to None. Library-produced
        counts use the canonical categories, independently of custom scalar labels.
    gdm_weighted : np.ndarray or None, optional
        Optional pointwise Spread-weighted GDM, default None. Selected operations
        retain original indexing and place NaN outside their active region.
    gdm_weighted_mean : float or None, optional
        Stored mean of active weighted values, default None.
    K_ADM, K_FDM : float or None, optional
        Spread-based weighting coefficients, each defaulting to None.
    debug : FSVDebugData or None, optional
        Source intermediate-data snapshot, default None. Windows can share source
        debug data; component combinations do not create a combined debug record.
    active_mask : np.ndarray or sequence of bool or None, optional
        Boolean selection used for the stored statistics. None declares the full
        point domain. An explicit mask is validated and copied to a read-only
        array; it must select at least one point and match the base array length.
        The stored value is an ndarray or None; the default is None.

    Returns
    -------
    FSVAnalysis
        Owned snapshot of supplied result fields and supported source metadata.

    Raises
    ------
    ValueError
        If storage/metadata is unsupported or an explicit active mask violates
        the Boolean selection contract.

    Notes
    -----
    Selection is indexed against the original point arrays, not FFT bins or
    physical coordinates. A mask does not establish coordinate alignment.
    Excluded combined values may be NaN placeholders. Original signal/FFT lengths
    are not replaced by the active count. Fields and numerical buffers are
    read-only under ordinary operations. Public construction owns copies;
    windowing shares stable base buffers without modifying them. Each array
    access returns a fresh read-only header over the same stable data, so shape
    or dtype edits to a retrieved view do not modify stored array metadata.
    Legacy results without recorded selection require recomputation or an
    explicit known mask; selection is never inferred from optional NaNs.

    Copy/deepcopy return this immutable instance. dataclasses.replace creates
    a distinct owned record but does not recompute caches. Identity comparison
    replaces generated array equality; instances remain unhashable. Use explicit
    array copies for editable numerical data; dataclasses.asdict detaches fields
    but is not a JSON conversion.

    Construction and restoration establish storage ownership, not deep cache
    consistency or proof of historical provenance. Primitive numerical arrays
    retain dtype, shape and valid excluded NaNs; views still validate consumed
    fields. Same-version restoration requires recorded selection and compatible
    metadata. This is not a security boundary against low-level Python bypasses.

    Core Grade/Spread requires ``compute_grade_spread=True``. Core weighted GDM
    requires both ``compute_grade_spread`` and ``use_weighted_gdm``. Postprocessing
    follows its own documented recalculation gates. Inspect optional fields for
    availability instead of treating inherited configuration flags as a recipe.
    Keep coordinates, operation arguments and source settings in caller records.

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv import FSV
    >>> analysis = FSV(store_debug=False).fsv(np.ones(64), np.full(64, 2.0))
    >>> print(analysis.gdm_category)
    Fair
    >>> analysis.gdm_point.shape, analysis.active_count
    ((64,), 64)
    >>> analysis.gdm_point.flags.writeable
    False
    >>> analysis.debug is None
    True
    """

    # Core ADM results
    adm_point: np.ndarray
    adm_mean: float
    adm_conf_hist: np.ndarray
    adm_visual: float

    # Core FDM results
    fdm_point: np.ndarray
    fdm_mean: float
    fdm_conf_hist: np.ndarray
    fdm_visual: float

    # Core GDM results
    gdm_point: np.ndarray
    gdm_mean: float
    gdm_conf_hist: np.ndarray
    gdm_visual: float

    # Metadata
    len_orig: int
    len_fft: int
    break_point_lo_hi: int
    config: Optional["FSVConfig"] = None

    # Optional Grade and Spread (IEEE §8.5(b))
    adm_grade: Optional[int] = None
    adm_spread: Optional[int] = None
    fdm_grade: Optional[int] = None
    fdm_spread: Optional[int] = None
    gdm_grade: Optional[int] = None
    gdm_spread: Optional[int] = None

    # Optional Weighted GDM (IEEE §8.5(b))
    gdm_weighted: Optional[np.ndarray] = None
    gdm_weighted_mean: Optional[float] = None
    K_ADM: Optional[float] = None
    K_FDM: Optional[float] = None

    # Optional debug data
    debug: Optional[FSVDebugData] = None

    active_mask: Optional[np.ndarray] = None

    _adopt_owned_arrays: InitVar[bool] = False
    _snapshot_ready: ClassVar[bool] = False
    _snapshot_array_names: ClassVar[Tuple[str, ...]] = (
        "adm_point", "fdm_point", "gdm_point", "adm_conf_hist", "fdm_conf_hist",
        "gdm_conf_hist", "gdm_weighted",
    )
    __hash__ = None
    __getattribute__ = _snapshot_getattribute
    __copy__ = _copy_snapshot
    __deepcopy__ = _deepcopy_snapshot
    __reduce__ = _reduce_snapshot
    __setstate__ = _set_snapshot_state
    _from_owned_arrays = classmethod(_from_owned_snapshot)

    def __post_init__(self, _adopt_owned_arrays: bool) -> None:
        """Own result storage and preserve explicit selection provenance.

        Parameters
        ----------
        _adopt_owned_arrays : bool
            Private transfer mode for audited library producers.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If storage/metadata is unsupported or selection is malformed.

        Examples
        --------
        >>> from copy import copy
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV().fsv(np.ones(64), np.full(64, 2.0))
        >>> result.gdm_point.flags.writeable
        False
        >>> copy(result) is result
        True
        >>> editable = result.gdm_point.copy()
        >>> editable[:] = 0
        >>> bool(np.all(result.gdm_point > 0))
        True
        """
        _validate_snapshot_storage(self)
        _validate_snapshot_metadata(self)
        selection = self._active_index()
        _freeze_snapshot_arrays(self, _adopt_owned_arrays)
        if not isinstance(selection, slice):
            object.__setattr__(self, "active_mask", _snapshot_array(selection, "active_mask"))
        object.__setattr__(self, "_snapshot_ready", True)

    def _active_index(self) -> Union[slice, np.ndarray]:
        """Resolve the recorded domain without guessing historical selections.

        Returns
        -------
        slice or np.ndarray
            Full slice for a declared full domain, otherwise a Boolean index.

        Raises
        ------
        ValueError
            If state is absent or invalid. Legacy or deliberately corrupted
            metadata is checked before selection-aware operations use it.
        """
        if "active_mask" not in self.__dict__:
            raise ValueError(
                "Result has no recorded selection. Recompute it or apply an "
                "explicit known mask before combining, exporting or plotting."
            )
        if self.active_mask is None:
            return slice(None)
        mask = _validate_boolean_mask(self.active_mask)
        if len(mask) != len(self.adm_point):
            raise ValueError("Active mask length must match the point-array domain.")
        if not np.any(mask):
            raise ValueError("Active mask must select at least one point.")
        return mask

    @property
    def active_count(self) -> int:
        """Return the number of points represented by the stored statistics.

        Returns
        -------
        int
            Active point count, independent of the original-length metadata.

        Raises
        ------
        ValueError
            If the result has no valid recorded selection.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> from kt_fsv.postprocessing import apply_window
        >>> full = FSV().fsv(np.ones(64), np.ones(64))
        >>> selected = apply_window(full, np.arange(64) < 32)
        >>> selected.active_count
        32
        >>> selected.active_mask.flags.writeable
        False
        """
        selection = self._active_index()
        return len(self.adm_point) if isinstance(selection, slice) else int(np.count_nonzero(selection))

    def _validate_report_selection(self, *, allow_unknown: bool = False) -> None:
        """Check the point-domain anchor and the selection needed by a view.

        Parameters
        ----------
        allow_unknown : bool, optional
            Permit the existing legacy diagnostic repr to label selection unknown.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the point-domain anchor or required recorded selection is invalid.
        """
        if not isinstance(self.adm_point, np.ndarray) or self.adm_point.ndim != 1 or self.adm_point.size == 0:
            raise ValueError("adm_point must define a nonempty one-dimensional point domain.")
        if allow_unknown and "active_mask" not in self.__dict__:
            return
        self._active_index()

    def _validate_summary_fields(
        self, *, visuals: bool = False, standard_categories: bool = False,
        allow_unknown: bool = False,
    ) -> None:
        """Preflight consumed scalar fields and labels before any output is emitted.

        Parameters
        ----------
        visuals : bool, optional
            Check VRS correspondence when the output consumes stored ratings.
        standard_categories : bool, optional
            Resolve standard histogram labels rather than configured scalar labels.
        allow_unknown : bool, optional
            Preserve the explicit unknown-selection legacy repr behavior.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If required scalars, category labels, ratings or selection are invalid.

        Notes
        -----
        This does not read unused histograms, optional counts or point values.
        It neither recomputes cached means nor mutates any result/configuration.

        Examples
        --------
        >>> from kt_fsv import FSV
        >>> from dataclasses import replace
        >>> result = FSV().fsv(np.ones(64), np.ones(64))
        >>> result = replace(result, gdm_visual=np.nan)
        >>> result._validate_summary_fields()
        >>> result._validate_summary_fields(visuals=True)
        Traceback (most recent call last):
            ...
        ValueError: gdm_visual must be a finite, nonnegative real scalar.
        """
        self._validate_report_selection(allow_unknown=allow_unknown)
        for name in ("adm", "fdm", "gdm"):
            mean = getattr(self, f"{name}_mean")
            _validated_display_scalar(mean, f"{name}_mean")
            try:
                if standard_categories:
                    _get_standard_category(mean)
                else:
                    getattr(self, f"{name}_category")
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name}_category cannot be resolved: {error}") from error
            if visuals:
                _validate_cached_vrs(mean, getattr(self, f"{name}_visual"), f"{name}_visual")

    def _validate_point_view(self) -> None:
        """Validate layouts and active measures before any point plot is drawn.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a required layout, active value, stored mean or label is invalid.

        Notes
        -----
        Only active values are checked; excluded placeholders remain permitted.
        No cached mean is recomputed from the point arrays.
        """
        self._validate_summary_fields()
        selection = self._active_index()
        length = len(self.adm_point)
        for name in ("adm_point", "fdm_point", "gdm_point"):
            points = getattr(self, name)
            if not isinstance(points, np.ndarray) or points.ndim != 1 or len(points) != length:
                raise ValueError(f"{name} must be a one-dimensional array matching the point domain.")
            _reporting_array(points[selection], f"{name} active values")

    def _validate_histogram_view(self) -> None:
        """Validate cached probability arrays and the scalar labels a histogram uses.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any required histogram, mean, label or selection is invalid.

        Notes
        -----
        Stored histogram fields are arrays; multiplying a Python list by 100
        repeats it rather than producing percentages. Probability validation
        uses the existing rounding allowance and never renormalizes the cache.
        """
        self._validate_summary_fields(standard_categories=True)
        for name in ("adm_conf_hist", "fdm_conf_hist", "gdm_conf_hist"):
            histogram = getattr(self, name)
            if not isinstance(histogram, np.ndarray):
                raise ValueError(f"{name} must be a six-bin probability array.")
            try:
                _validate_probability_histogram(histogram)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} is invalid: {error}") from error

    def _validate_grade_pair(self, name: str) -> bool:
        """Check one optional Grade/Spread pair when a view consumes its markers.

        Parameters
        ----------
        name : str
            Lowercase measure prefix.

        Returns
        -------
        bool
            True for a complete valid pair, False if both fields are absent.

        Raises
        ------
        ValueError
            If the pair is partial or contains noninteger/out-of-range counts.

        Notes
        -----
        Python/NumPy integers are count values. Boolean flags and floating-point
        values, including integral-valued floats, are not accepted as counts.
        """
        grade, spread = getattr(self, f"{name}_grade"), getattr(self, f"{name}_spread")
        if grade is None and spread is None:
            return False
        for value in (grade, spread):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= 6:
                raise ValueError(
                    f"{name} Grade/Spread must be a complete pair of integer counts "
                    "in 1..6, or both None when absent."
                )
        return True

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def adm_category(self) -> str:
        """Get the configured ADM interpretation.

        Returns
        -------
        str
            Category using configured limits or the current default lookup.

        Raises
        ------
        ValueError
            If the score or effective scale is invalid.
        """
        thresholds = self.config.category_thresholds if self.config else None
        return get_category(self.adm_mean, thresholds)

    @property
    def fdm_category(self) -> str:
        """Get the configured FDM interpretation.

        Returns
        -------
        str
            Category using configured limits or the current default lookup.

        Raises
        ------
        ValueError
            If the score or effective scale is invalid.
        """
        thresholds = self.config.category_thresholds if self.config else None
        return get_category(self.fdm_mean, thresholds)

    @property
    def gdm_category(self) -> str:
        """Get the configured GDM interpretation.

        Returns
        -------
        str
            Category using configured limits or the current default lookup.

        Raises
        ------
        ValueError
            If the score or effective scale is invalid.
        """
        thresholds = self.config.category_thresholds if self.config else None
        return get_category(self.gdm_mean, thresholds)

    @property
    def category(self) -> str:
        """Return the configured GDM category.

        Returns
        -------
        str
            Alias of gdm_category, including its current-default fallback.

        Raises
        ------
        ValueError
            If the stored GDM mean or effective category scale is invalid.
        """
        return self.gdm_category

    @property
    def adm_standard_category(self) -> str:
        """Get the standard ADM category, independent of custom limits.

        Returns
        -------
        str
            Category of the stored ADM mean on the canonical scale.

        Raises
        ------
        ValueError
            If the stored mean is invalid.
        """
        return _get_standard_category(self.adm_mean)

    @property
    def fdm_standard_category(self) -> str:
        """Get the standard FDM category, independent of custom limits.

        Returns
        -------
        str
            Category of the stored FDM mean on the canonical scale.

        Raises
        ------
        ValueError
            If the stored mean is invalid.
        """
        return _get_standard_category(self.fdm_mean)

    @property
    def gdm_standard_category(self) -> str:
        """Get the standard GDM category, independent of custom limits.

        Returns
        -------
        str
            Category of the stored GDM mean on the canonical scale.

        Raises
        ------
        ValueError
            If the stored mean is invalid.
        """
        return _get_standard_category(self.gdm_mean)

    def _category_view(self) -> Dict[str, Any]:
        """Describe the effective configured lookup without inventing its history.

        Returns
        -------
        dict
            Scale identity, a copied limit mapping and its current source.
            Native scalar values are retained; this is not JSON conversion.

        Raises
        ------
        ValueError
            If the effective lookup has invalid names or numerical limits.

        Notes
        -----
        Absent config uses the current legacy default lookup, which need not
        describe the settings of an earlier calculation.
        """
        thresholds = self.config.category_thresholds if self.config else DEFAULT_CATEGORY_THRESHOLDS
        limits = _validated_category_limits(thresholds)
        standard = all(
            _category_ratio(limit) == _category_ratio(reference)
            for limit, (_, reference) in zip(limits[:-1], DEFAULT_CATEGORY_ITEMS[:-1])
        )
        return {
            "category_scale": "standard" if standard else "custom",
            "category_thresholds": {
                name: limit for (name, _), limit in zip(DEFAULT_CATEGORY_ITEMS, limits)
            },
            "category_thresholds_source": "config" if self.config else "current_defaults",
        }

    def _display_category(self, measure: str) -> str:
        """Identify both interpretations when a custom scalar scale is active.

        Parameters
        ----------
        measure : str
            Internal measure name: adm, fdm or gdm.

        Returns
        -------
        str
            Ordinary standard label, or explicitly named standard/custom labels.

        Raises
        ------
        ValueError
            If the score or effective category scale is invalid.
        """
        configured = getattr(self, f"{measure}_category")
        if self._category_view()["category_scale"] == "standard":
            return configured
        standard = getattr(self, f"{measure}_standard_category")
        return f"standard: {standard}; custom: {configured}"

    @property
    def visualizer(self) -> "kt_fsv.FSVVisualizer":
        """
        Create a visualizer for this result snapshot.

        Returns
        -------
        FSVVisualizer
            A new helper instance referencing this analysis.

        Notes
        -----
        Each access creates a separate visualizer with its own figure tracking.
        Use one instance or a context manager to save and close its figures.
        Measures and histograms do not require debug data; input/spectrum/filter
        views require a suitable source debug snapshot.
        The example writes a PNG in the working directory.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=False).fsv(np.ones(64), np.full(64, 2.0))
        >>> with result.visualizer as viz:
        ...     figure = viz.plot_measures()
        ...     figure.savefig("result_measures.png", dpi=150, bbox_inches="tight")
        """
        from .visualization import FSVVisualizer
        return FSVVisualizer(self)

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        """Return a validated detailed representation.

        Returns
        -------
        str
            Stored means, ratings and metadata.

        Raises
        ------
        ValueError
            If a consumed scalar, rating or category label is invalid.

        Notes
        -----
        Legacy selection can still be labelled unknown. Scalar validation does
        not recompute all caches or certify correspondence with point arrays.
        """
        self._validate_summary_fields(visuals=True, allow_unknown=True)
        active = self.active_count if "active_mask" in self.__dict__ else "unknown"
        return (
            f"FSVAnalysis(\n"
            f"  ADM: mean={self.adm_mean:.4f} ({self._display_category('adm')}), "
            f"visual={self.adm_visual:.2f}\n"
            f"  FDM: mean={self.fdm_mean:.4f} ({self._display_category('fdm')}), "
            f"visual={self.fdm_visual:.2f}\n"
            f"  GDM: mean={self.gdm_mean:.4f} ({self._display_category('gdm')}), "
            f"visual={self.gdm_visual:.2f}\n"
            f"  breakpoint={self.break_point_lo_hi}, "
            f"len={self.len_orig}, active={active}, debug={'yes' if self.debug else 'no'}\n"
            f")"
        )

    def summary(self) -> str:
        """
        Return a text summary of core measures and source/selection metadata.

        Returns
        -------
        str
            Core means, categories, VRS values, lengths, active count and
            available source breakpoint/normalization settings.

        Raises
        ------
        ValueError
            If required selection, mean, rating or category data are invalid.

        Notes
        -----
        Mean/VRS correspondence is checked without replacing stored fields.
        This output does not audit point-derived means or histogram caches.
        Length/breakpoint/normalization fields describe source context that can
        be inherited after postprocessing; the selection describes the current
        statistical domain. The summary is not a complete processing history.
        It omits point arrays, histograms, Grade/Spread and weighted-GDM fields.
        Read those attributes directly when the operation produced them.
        """
        self._validate_summary_fields(visuals=True)
        lines = [
            "=" * 60,
            "FSV Analysis Summary",
            "=" * 60,
            f"ADM: {self.adm_mean:.4f} ({self._display_category('adm')}) VRS={self.adm_visual:.2f}",
            f"FDM: {self.fdm_mean:.4f} ({self._display_category('fdm')}) VRS={self.fdm_visual:.2f}",
            f"GDM: {self.gdm_mean:.4f} ({self._display_category('gdm')}) VRS={self.gdm_visual:.2f}",
            "-" * 60,
            f"Signal length: {self.len_orig}",
            f"Active points: {self.active_count}/{len(self.adm_point)}",
            f"FFT length: {self.len_fft}",
            f"Breakpoint Lo/Hi: {self.break_point_lo_hi}",
            f"Breakpoint DC: {self.config.break_point_dc if self.config else 'N/A'}",
            f"Normalization: {self.config.normalization if self.config else 'N/A'}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_results(self) -> None:
        """
        Print the core-measure summary to the console.

        Equivalent to ``print(self.summary())``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If summary validation fails, before any text is printed.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV().fsv(np.ones(64), np.full(64, 2.0))
        >>> result.print_results()
        ============================================================
        FSV Analysis Summary
        ============================================================
        ADM: 0.4652 (Fair) VRS=4.17
        FDM: 0.0000 (Excellent) VRS=1.00
        GDM: 0.4652 (Fair) VRS=4.17
        ------------------------------------------------------------
        Signal length: 64
        Active points: 64/64
        FFT length: 64
        Breakpoint Lo/Hi: 9
        Breakpoint DC: 5
        Normalization: sqrt_N
        ============================================================
        """
        print(self.summary())

    def print_compact(self) -> None:
        """
        Print one-line compact summary to console.

        Useful for comparing multiple results or logging.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the required selection, mean or category data are invalid.

        Notes
        -----
        This view consumes means and categories, not VRS or optional caches.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV().fsv(np.ones(64), np.full(64, 2.0))
        >>> result.print_compact()
        ADM=0.4652 (Fair), FDM=0.0000 (Excellent), GDM=0.4652 (Fair), active=64/64
        """
        self._validate_summary_fields()
        print(
            f"ADM={self.adm_mean:.4f} ({self._display_category('adm')}), "
            f"FDM={self.fdm_mean:.4f} ({self._display_category('fdm')}), "
            f"GDM={self.gdm_mean:.4f} ({self._display_category('gdm')}), "
            f"active={self.active_count}/{len(self.adm_point)}"
        )

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a native Python summary mapping.

        Returns
        -------
        dict
            Scalar summary, category-scale and active-selection metadata.
            This is not a full round-trip serialization of point arrays
            or a complete IEEE report.

        Raises
        ------
        ValueError
            If required selection, means, ratings or category labels are invalid.

        Notes
        -----
        Mean/VRS correspondence is validated before returning the dictionary.
        Values are not repaired, rounded or converted to new storage types.
        This is not a deep cache-consistency audit or a JSON-type conversion.
        Existing category keys retain their configured interpretation.
        Standard-category keys use canonical limits. The effective scale's
        limits are copied with their native scalar values, including infinity;
        absent config is labelled current_defaults, not historical provenance.
        Length/breakpoint/normalization fields retain their source meaning after
        postprocessing. Operation arguments, complete optional-output state and
        history are not exported; keep the actual call settings separately.
        Point arrays, histograms, Grade/Spread, weighted-GDM fields, debug arrays
        and the full configuration object are omitted.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=False).fsv(np.ones(64), np.full(64, 2.0))
        >>> exported = result.to_dict()
        >>> exported["active_count"], exported["gdm_category"]
        (64, 'Fair')
        >>> "gdm_point" in exported, "gdm_weighted_mean" in exported
        (False, False)
        """
        self._validate_summary_fields(visuals=True)
        selection = self._active_index()
        return {
            **self._category_view(),
            "adm_mean": self.adm_mean,
            "adm_visual": self.adm_visual,
            "adm_category": self.adm_category,
            "adm_standard_category": self.adm_standard_category,
            "fdm_mean": self.fdm_mean,
            "fdm_visual": self.fdm_visual,
            "fdm_category": self.fdm_category,
            "fdm_standard_category": self.fdm_standard_category,
            "gdm_mean": self.gdm_mean,
            "gdm_visual": self.gdm_visual,
            "gdm_category": self.gdm_category,
            "gdm_standard_category": self.gdm_standard_category,
            "len_orig": self.len_orig,
            "len_fft": self.len_fft,
            "break_point_lo_hi": self.break_point_lo_hi,
            "break_point_dc": self.config.break_point_dc if self.config else None,
            "normalization": self.config.normalization if self.config else None,
            "active_count": self.active_count,
            "active_mask": None if isinstance(selection, slice) else selection.tolist(),
        }
