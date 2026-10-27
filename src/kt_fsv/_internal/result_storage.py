"""Private array storage and ownership support for result snapshots.

Notes
-----
These helpers retain the existing storage rules. Result classes and their
metadata, copying and pickle-restoration decisions remain in kt_fsv.analysis.
"""

from dataclasses import fields
from typing import Any, TYPE_CHECKING, Type, TypeVar

import numpy as np

if TYPE_CHECKING:
    from ..analysis import FSVAnalysis, FSVDebugData


_SnapshotT = TypeVar("_SnapshotT", "FSVAnalysis", "FSVDebugData")


def _check_snapshot_array(value: object, name: str) -> None:
    """Require primitive numerical storage without coercing values.

    Parameters
    ----------
    value : object
        Candidate ndarray.
    name : str
        Field identified in an error.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If storage is not a primitive numerical ndarray or carries masked data.
    """
    if not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value):
        raise ValueError(f"{name} must use primitive numerical ndarray storage.")
    if value.dtype.kind not in "biufc":
        raise ValueError(f"{name} must use primitive numerical ndarray storage.")


def _snapshot_array(value: np.ndarray, name: str, adopt: bool = False) -> np.ndarray:
    """Copy borrowed data or seal a trusted owned ndarray allocation chain.

    Parameters
    ----------
    value : ndarray
        Primitive numerical buffer, with dtype/shape preserved.
    name : str
        Field name for storage errors.
    adopt : bool, optional
        Private ownership-transfer promise from an audited producer.

    Returns
    -------
    ndarray
        Read-only view backed by read-only owned storage.

    Raises
    ------
    ValueError
        If storage is unsupported.

    Notes
    -----
    Unknown backing types and ndarray subclasses are copied. Trusted real
    views of complex IFFT arrays require sealing their full owned base chain.
    """
    _check_snapshot_array(value, name)
    buffer = value if adopt and type(value) is np.ndarray else np.array(value, copy=True, order="K", subok=False)
    chain = [buffer]
    while isinstance(chain[-1].base, np.ndarray):
        chain.append(chain[-1].base)
    if chain[-1].base is not None:
        buffer = np.array(buffer, copy=True, order="K", subok=False)
        chain = [buffer]
    for owner in reversed(chain):
        owner.setflags(write=False)
    view = buffer.view()
    view.setflags(write=False)
    return view


def _validate_snapshot_storage(record: _SnapshotT) -> None:
    """Validate storage immutability without auditing cached numerical values.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Newly constructed record.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a field retains unsupported mutable storage.
    """
    scalar_types = (int, float, complex, bool, np.integer, np.floating, np.complexfloating, np.bool_)
    for item in fields(record):
        if item.name in ("config", "debug", "active_mask"):
            continue
        value = getattr(record, item.name)
        if item.name in record._snapshot_array_names:
            if item.name == "gdm_weighted" and value is None:
                continue
            _check_snapshot_array(value, item.name)
        elif value is not None and not isinstance(value, scalar_types):
            raise ValueError(f"{item.name} must use immutable scalar storage.")


def _freeze_snapshot_arrays(record: _SnapshotT, adopt: bool) -> None:
    """Seal array fields after storage/selection checks.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Record under construction.
    adopt : bool
        Private producer ownership promise.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the private mode is not a Boolean.
    """
    if type(adopt) is not bool:
        raise ValueError("Snapshot ownership mode must be Boolean.")
    for name in record._snapshot_array_names:
        value = getattr(record, name)
        if value is not None:
            object.__setattr__(record, name, _snapshot_array(value, name, adopt))


def _readonly_snapshot_buffer(value: np.ndarray) -> bool:
    """Check that a stored view and its ndarray owners are read-only.

    Parameters
    ----------
    value : ndarray
        Snapshot buffer.

    Returns
    -------
    bool
        Whether ordinary array writes/direct view unlocking are blocked.
    """
    while isinstance(value, np.ndarray):
        if value.flags.writeable:
            return False
        value = value.base
    return value is None


def _snapshot_getattribute(record: _SnapshotT, name: str) -> Any:
    """Keep array header edits local to a retrieved read-only view.

    Parameters
    ----------
    record : FSVAnalysis or FSVDebugData
        Snapshot being accessed.
    name : str
        Requested field or attribute.

    Returns
    -------
    object
        A fresh numerical-array view after sealing, otherwise the stored value.

    Raises
    ------
    AttributeError
        If the requested attribute is absent.

    Notes
    -----
    NumPy's writeable flag protects data, not shape/dtype header assignment.
    A fresh header shares immutable data without copying the numerical buffer.
    """
    value = object.__getattribute__(record, name)
    array_names = object.__getattribute__(record, "_snapshot_array_names")
    if name in array_names or name == "active_mask":
        ready = object.__getattribute__(record, "__dict__").get("_snapshot_ready")
        if ready is True and isinstance(value, np.ndarray):
            return value.view()
    return value


def _from_owned_snapshot(snapshot_type: Type[_SnapshotT], **values: Any) -> _SnapshotT:
    """Transfer audited producer buffers through the normal constructor.

    Parameters
    ----------
    snapshot_type : type
        Concrete result/debug class.
    **values : object
        Fields whose arrays are newly owned or already stable parent buffers.

    Returns
    -------
    FSVAnalysis or FSVDebugData
        Sealed snapshot without redundant numerical-buffer copies.

    Notes
    -----
    This is a private transfer promise, never a caller-facing zero-copy option.
    """
    return snapshot_type(_adopt_owned_arrays=True, **values)
