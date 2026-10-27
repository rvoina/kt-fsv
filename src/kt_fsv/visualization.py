"""
Plot stored FSV results and optional intermediate data with Matplotlib.

Notes
-----
FSVVisualizer returns Figure objects. Individual methods accept suitable
caller-owned axes. Owned figures are tracked before subplot construction;
failed plotting operations close their newly tracked figures. Plotting does not
recalculate FSV statistics, establish physical coordinate units or produce a
complete report.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore
    Polygon = None  # type: ignore

from . import filters
from .analysis import FSVAnalysis, _get_standard_category, get_category
from .config import DEFAULT_CATEGORY_THRESHOLDS
from .constants import DEFAULT_CATEGORY_ITEMS
from .measures import _validate_reporting_scalar
from .styles import (
    PlotStyle,
    PRESET_STYLES,
    FIG_SIZE,
    FIG_SIZE_TALL,
    GRADE_SPREAD_REGION_EXCELLENT,
    GRADE_SPREAD_REGION_GOOD,
    GRADE_SPREAD_REGION_FAIR,
    GRADE_SPREAD_REGION_POOR,
)

_PlotReturn = TypeVar("_PlotReturn")


def _close_failed_plot(method: Callable[..., _PlotReturn]) -> Callable[..., _PlotReturn]:
    """Close newly tracked figures if a plotting operation does not complete.

    Parameters
    ----------
    method : callable
        Visualizer method whose figure ownership is recorded in _figures.

    Returns
    -------
    callable
        Wrapper preserving the method's signature metadata and return value.

    Notes
    -----
    Cleanup also runs when an interrupt unwinds the call. Existing tracked
    figures and untracked caller figures are outside this operation's cleanup.
    The original failure propagates; this is not an axes or filesystem rollback.
    """
    @wraps(method)
    def wrapped(self: FSVVisualizer, *args: object, **kwargs: object) -> _PlotReturn:
        """Invoke one operation with scoped ownership cleanup.

        Parameters
        ----------
        self : FSVVisualizer
            Owner of figures created by the operation.
        *args, **kwargs : object
            Original positional and keyword arguments.

        Returns
        -------
        object
            The original method's return value.
        """
        previous = {id(figure) for figure in self._figures}
        completed = False
        try:
            result = method(self, *args, **kwargs)
            completed = True
            return result
        finally:
            if not completed:
                self._close_new_figures(previous)
    return wrapped


def _validate_caption_threshold(value: object, name: str) -> None:
    """Require a finite probability threshold without changing its numeric type.

    Parameters
    ----------
    value : object
        Python/NumPy real scalar to validate.
    name : str
        Diagnostic field name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the value is not a finite scalar in (0,1].
    """
    _validate_reporting_scalar(value, name)
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1].")


def _format_caption_threshold(value: object) -> str:
    """Format a threshold scalar and percentage without discarding its precision.

    Parameters
    ----------
    value : object
        Finite real scalar in (0,1].

    Returns
    -------
    str
        Scalar value followed by its readable percentage.

    Raises
    ------
    ValueError
        If the threshold is invalid.

    Notes
    -----
    Format at least at float64 precision while retaining wider input types.
    The scalar text disambiguates any rounding in percentage multiplication.
    Small values use scientific notation rather than a long run of zeros.
    No analysis field is changed.

    Examples
    --------
    >>> _format_caption_threshold(0.855)
    '0.855 (85.5%)'
    >>> _format_caption_threshold(np.float32(0.85))
    '0.8500000238418579 (85.00000238418579%)'
    """
    _validate_caption_threshold(value, "threshold")
    array = np.asarray(value)
    promoted = array.astype(np.result_type(array.dtype, np.float64))[()]
    text = [
        np.format_float_scientific(number, unique=True, trim="-")
        if number < 1e-4 else np.format_float_positional(number, unique=True, trim="-")
        for number in (promoted, promoted * 100)
    ]
    return f"{text[0]} ({text[1]}%)"

# =============================================================================
# FSV Visualizer Class
# =============================================================================

class FSVVisualizer:
    """
    Plot a result snapshot and manage figures created by this helper.

    Parameters
    ----------
    analysis : FSVAnalysis
        Result to reference. Plot methods validate the fields they consume;
        construction does not perform a complete result-consistency audit.

    Returns
    -------
    FSVVisualizer
        Helper referencing the supplied analysis, with separate figure tracking.

    Raises
    ------
    ImportError
        If Matplotlib is unavailable.

    Notes
    -----
    Each access to analysis.visualizer creates a new helper. Retain one instance
    or use a context manager. Figures created without supplied axes are tracked
    before drawing; caller-owned figures are not tracked or automatically closed.
    All plot methods return a Figure, not a (figure, axes) tuple. Access its
    axes through figure.axes, or retain supplied axes.

    If a plot fails or is interrupted, it closes only the newly tracked figures
    from that operation. A failed batch also closes figures from its earlier
    successful views. Previously tracked and caller-owned figures remain open;
    context exit separately closes all figures still tracked by this helper.
    Supplied axes may retain partial drawing after a renderer error. Files
    already written, including a partial failed write, are not rolled back.
    A backend failure before its Figure factory returns lies outside this
    ownership cleanup. Closing releases the pyplot manager, not caller-held
    Figure/artist references.

    Measures use original sample indices and hide excluded positions with gaps.
    Raw/debug views retain full-analysis context. Neither sample nor FFT-bin
    indices are automatically converted to physical units. Plots display stored
    base measures and summaries; optional weighted GDM is not a separate view.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from kt_fsv import FSV
    >>> result = FSV(store_debug=False).fsv(np.ones(64), np.full(64, 2.0))
    >>> with result.visualizer as viz:
    ...     figure = viz.plot_measures()
    ...     print(len(figure.axes))
    3
    >>> plt.fignum_exists(figure.number)
    False
    """

    def __init__(self, analysis: FSVAnalysis) -> None:
        """
        Retain an analysis reference and initialize figure tracking.

        Parameters
        ----------
        analysis : FSVAnalysis
            Snapshot to visualize; its arrays are not copied.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If Matplotlib is unavailable.
        """
        self.analysis = analysis
        self._figures: List[plt.Figure] = []
        self._check_matplotlib()

    def _check_matplotlib(self) -> None:
        """
        Check if matplotlib is available.

        Raises
        ------
        ImportError
            If matplotlib is not installed
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

    def _track_figure(self, fig: plt.Figure) -> plt.Figure:
        """
        Track a created figure for later cleanup.

        Parameters
        ----------
        fig : plt.Figure
            Figure to track

        Returns
        -------
        plt.Figure
            Same figure (for chaining)
        """
        if not any(figure is fig for figure in self._figures):
            self._figures.append(fig)
        return fig

    def _close_new_figures(self, previous: Set[int]) -> None:
        """Close tracked figures created since an operation began.

        Parameters
        ----------
        previous : set of int
            Object identities tracked on entry.

        Returns
        -------
        None

        Notes
        -----
        Figure objects, rather than reused pyplot numbers, identify ownership.
        Caller-supplied figures are never added to this tracking list.
        """
        for figure in self._figures[:]:
            if id(figure) not in previous:
                plt.close(figure)
                self._figures.remove(figure)

    def close_all_figures(self) -> None:
        """
        Close this helper's tracked figures and clear its tracking list.

        Returns
        -------
        None

        Notes
        -----
        Caller-owned figures and figures from other visualizer instances remain
        open. A backend failure before a Figure is returned cannot be tracked.
        Caller-held Figure objects can still retain their artists and data.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=False).fsv(np.ones(64), np.full(64, 2.0))
        >>> viz = result.visualizer
        >>> figure = viz.plot_measures()
        >>> viz.close_all_figures()
        >>> plt.fignum_exists(figure.number)
        False
        """
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()

    def save_figure(
        self,
        fig: plt.Figure,
        output_dir: Union[str, Path],
        name: str,
        dpi: int = 150,
        close_after: bool = True,
    ) -> Path:
        """
        Save a Figure as PNG and optionally close it after successful saving.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to save
        output_dir : str or Path
            Destination directory; missing parents are created.
        name : str
            Base filename without extension; ".png" is appended.
        dpi : int, optional
            Resolution in dots per inch. Default is 150.
        close_after : bool, optional
            Close the figure after saving and remove it from this helper's
            tracking list when present. Default is True.

        Returns
        -------
        Path
            Destination path, retaining the supplied directory's relativity.

        Raises
        ------
        OSError
            If directory creation or file writing fails.
        ValueError
            If Matplotlib rejects a rendering or save parameter.

        Notes
        -----
        Uses bbox_inches="tight". The explicit figure may be caller-owned;
        close_after=True closes it too. Saving does not add a figure to tracking.
        A failed save does not reach the close step.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=False).fsv(np.ones(64), np.full(64, 2.0))
        >>> with TemporaryDirectory() as folder, result.visualizer as viz:
        ...     figure = viz.plot_measures()
        ...     saved = viz.save_figure(figure, folder, "measures")
        ...     print(saved.name, saved.is_file(), plt.fignum_exists(figure.number))
        measures.png True False
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{name}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

        if close_after:
            plt.close(fig)
            if fig in self._figures:
                self._figures.remove(fig)

        return filepath

    def __enter__(self) -> FSVVisualizer:
        """Return this helper for use in a with statement.

        Returns
        -------
        FSVVisualizer
            This instance, with its existing tracking state.
        """
        return self

    def __exit__(self, *args) -> None:
        """Close tracked figures when leaving a with statement.

        Parameters
        ----------
        *args : tuple
            Context-manager exception information; not otherwise consumed.

        Returns
        -------
        None
            Exceptions from the context are not suppressed.

        Notes
        -----
        Cleanup has the same ownership and failure limits as close_all_figures.
        """
        self.close_all_figures()

    # =========================================================================
    # Helper Methods for Plot Configuration
    # =========================================================================

    def _resolve_style(
        self, 
        style: Optional[Union[str, PlotStyle]]
    ) -> PlotStyle:
        """
        Resolve style parameter to PlotStyle instance.
        
        Parameters
        ----------
        style : str, PlotStyle, or None
            Style specification
            
        Returns
        -------
        PlotStyle
            Resolved style instance
            
        Raises
        ------
        ValueError
            If a preset name is unknown or a numeric style field is invalid.
        TypeError
            If the specification is not None, a known preset or PlotStyle, or
            a numeric/Boolean field has an unsupported type.

        Notes
        -----
        Shared presets and custom instances are revalidated without copying or
        repairing them. Checks run before figure allocation or caller-axis drawing.
        """
        if style is None:
            return PlotStyle()
        if isinstance(style, str):
            if style not in PRESET_STYLES:
                available = ", ".join(f"'{s}'" for s in PRESET_STYLES.keys())
                raise ValueError(
                    f"Unknown style '{style}'. "
                    f"Available preset styles: {available}"
                )
            style = PRESET_STYLES[style]
        if not isinstance(style, PlotStyle):
            raise TypeError("style must be a preset name, a PlotStyle instance, or None.")
        style._validate()
        return style

    @_close_failed_plot
    def _setup_axes(
        self,
        ax: Optional[Union[plt.Axes, np.ndarray]],
        figsize: Tuple[float, float],
        subplot_spec: Union[int, Tuple[int, int]],
        sharex: bool = False
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray], bool]:
        """
        Create or validate provided axes for plotting.
        
        Parameters
        ----------
        ax : matplotlib Axes or array of Axes, optional
            User-provided axes. If None, creates new figure.
        figsize : tuple
            Figure size (width, height) - used only if ax is None
        subplot_spec : int or tuple
            Subplot specification:
            - int: 1D array of N axes (e.g., 3 for (3, 1))
            - tuple: 2D array shape (nrows, ncols)
        sharex : bool
            Whether to share x-axis (for 1D arrays only)
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        axes : plt.Axes or np.ndarray
            Axes object(s)
        owns_figure : bool
            True if we created the figure, False if user-provided
            
        Raises
        ------
        ValueError
            If provided axes have an invalid layout, non-Axes/masked entries,
            duplicates or different owning figures.

        Notes
        -----
        Logical vectors accept (N,), (1,N) and (N,1) arrays and return (N,).
        Genuine multirow/multicolumn grids retain their two-dimensional shape.
        Supplied axes preserve identity, order, positions and caller ownership.
        Owned figures are tracked before subplot construction and closed if
        axes construction or normalization raises.
        """
        self.analysis._active_index()
        if ax is None:
            if not isinstance(subplot_spec, (int, tuple)):
                raise ValueError(f"Invalid subplot_spec: {subplot_spec}")
            # Create new figure
            fig = self._track_figure(plt.figure(figsize=figsize))
            if isinstance(subplot_spec, int):
                # 1D array: (N, 1) layout
                axes = fig.subplots(
                    subplot_spec, 1, sharex=sharex, squeeze=False,
                )
            elif isinstance(subplot_spec, tuple):
                if subplot_spec == (1, 1):
                    # Single axes
                    axes = fig.subplots()
                else:
                    # 2D array
                    axes = fig.subplots(*subplot_spec, sharex=sharex)
            else:
                raise ValueError(f"Invalid subplot_spec: {subplot_spec}")
            
            axes = self._canonical_axes(axes, subplot_spec)
            owns_figure = True
        else:
            # Validate user-provided axes
            axes = self._canonical_axes(ax, subplot_spec)
            fig = self._axes_figure(axes)
            if isinstance(axes, np.ndarray):
                axes = np.asarray(axes, dtype=object)
            owns_figure = False
        
        return fig, axes, owns_figure

    @staticmethod
    def _axis_vector(axes: np.ndarray, count: int) -> np.ndarray:
        """Normalize only a flat, row or column vector of the required length.

        Parameters
        ----------
        axes : np.ndarray
            Supplied container, not yet checked for valid members.
        count : int
            Required positive axes count.

        Returns
        -------
        np.ndarray
            One-dimensional view/container retaining element order.

        Raises
        ------
        ValueError
            If the shape or container does not represent the required vector.
        """
        if count < 1:
            raise ValueError("Axes count must be positive.")
        if not isinstance(axes, np.ndarray) or axes.shape not in ((count,), (1, count), (count, 1)):
            raise ValueError(f"Expected an axes array shaped ({count},), (1,{count}) or ({count},1).")
        return axes.reshape(-1)

    def _canonical_axes(
        self, axes: Union[plt.Axes, np.ndarray], subplot_spec: Union[int, Tuple[int, int]],
    ) -> Union[plt.Axes, np.ndarray]:
        """Return the same axes in the layout expected by the drawing methods.

        Parameters
        ----------
        axes : matplotlib Axes or np.ndarray
            Owned or supplied axes.
        subplot_spec : int or tuple[int, int]
            Logical vector length, scalar (1,1) or genuine grid shape.

        Returns
        -------
        matplotlib Axes or np.ndarray
            Scalar, flat vector or unchanged two-dimensional grid.

        Raises
        ------
        ValueError
            If the container layout does not match the requested panel layout.
        """
        if isinstance(subplot_spec, int):
            # Expect 1D array of N axes
            return self._axis_vector(axes, subplot_spec)
        if not isinstance(subplot_spec, tuple) or len(subplot_spec) != 2:
            raise ValueError(f"Invalid axes subplot specification: {subplot_spec}")
        if subplot_spec == (1, 1):
            # Expect single Axes
            if not isinstance(axes, plt.Axes):
                raise ValueError(f"Expected single Axes, got {type(axes).__name__}")
            return axes
        if 1 in subplot_spec:
            return self._axis_vector(axes, subplot_spec[0] * subplot_spec[1])
        # Expect 2D array with specific shape
        if not isinstance(axes, np.ndarray) or axes.shape != subplot_spec:
            raise ValueError(f"Expected axes array with grid shape {subplot_spec}.")
        return axes

    @staticmethod
    def _axes_figure(axes: Union[plt.Axes, np.ndarray]) -> plt.Figure:
        """Validate every supplied axes before returning their common owner.

        Parameters
        ----------
        axes : matplotlib Axes or np.ndarray
            Shape-normalized caller data.

        Returns
        -------
        matplotlib Figure
            Existing owning figure; ownership is not transferred.

        Raises
        ------
        ValueError
            If entries are masked, invalid, repeated, detached or span figures.

        Notes
        -----
        Unknown masks are checked before ordinary array conversion. No axes,
        figure or caller container is modified by these checks.
        """
        if np.ma.is_masked(axes):
            raise ValueError("Axes cannot contain masked/unknown entries.")
        flat = np.asarray(axes, dtype=object).reshape(-1)
        if not flat.size or not all(isinstance(axis, plt.Axes) for axis in flat):
            raise ValueError("Every axes entry must be a Matplotlib Axes.")
        if len({id(axis) for axis in flat}) != flat.size:
            raise ValueError("Supplied axes must be distinct.")
        figure = flat[0].get_figure()
        if figure is None or any(axis.get_figure() is not figure for axis in flat):
            raise ValueError("Supplied axes must belong to one owning figure.")
        return figure

    def _finalize_plot(
        self,
        fig: plt.Figure,
        owns_figure: bool,
        style: PlotStyle,
        title: Optional[str] = None,
        show: bool = False,
        tight_layout_rect: Optional[List[float]] = None
    ) -> plt.Figure:
        """
        Apply final styling and handle figure display/tracking.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure to finalize
        owns_figure : bool
            Whether we created the figure (vs user-provided)
        style : PlotStyle
            Style configuration
        title : str, optional
            Suptitle to add (if owns_figure is True)
        show : bool
            Whether to call plt.show()
        tight_layout_rect : list of float, optional
            Rectangle for tight_layout [left, bottom, right, top]
            Default: [0, 0, 1, 0.96]
            
        Returns
        -------
        plt.Figure
            The finalized figure (tracked if we own it)
        """
        if owns_figure:
            # Add suptitle if provided
            if title:
                plt.suptitle(
                    title,
                    fontsize=style.title_size + 2,
                    fontweight=style.title_weight,
                    y=0.98
                )
            
            # Apply tight layout if requested
            if style.tight_layout:
                rect = tight_layout_rect or [0, 0, 1, 0.96]
                plt.tight_layout(rect=rect)
        
        # Show if requested
        if show:
            plt.show()
        
        # Track and return
        return self._track_figure(fig) if owns_figure else fig

    def _label_selection(
        self, axes: Union[plt.Axes, np.ndarray], *, context: bool = False,
    ) -> None:
        """Identify selected statistics or original-analysis context on plot axes.

        Parameters
        ----------
        axes : matplotlib Axes or np.ndarray
            Only the axes belonging to the current plot.
        context : bool, optional
            Label raw/debug/frequency views as original full-analysis context.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the result has no valid recorded selection.

        Notes
        -----
        Existing title fonts/styles and caller-owned unrelated axes are preserved.
        Sample selections are never applied to frequency bins.
        """
        selection = self.analysis._active_index()
        if isinstance(selection, slice):
            return
        count = f"{self.analysis.active_count}/{len(self.analysis.adm_point)}"
        note = (
            f"Full analysis context; selected statistics: {count}" if context
            else f"Selected points: {count}"
        )
        for axis in np.asarray(axes, dtype=object).flat:
            axis.title.set_text(f"{axis.get_title()}\n{note}")

    def _grade_threshold_caption(self, supplied: Optional[float]) -> str:
        """Resolve recorded, caller-declared or unknown threshold provenance.

        Parameters
        ----------
        supplied : float or None
            Validated caller assertion, not a request to recalculate markers.

        Returns
        -------
        str
            Precise caption including caller provenance when configuration is absent.

        Raises
        ------
        ValueError
            If required recorded metadata is invalid or conflicts with the assertion.

        Notes
        -----
        Compare strong scalar-array numeric values, not weak NumPy scalar equality
        or a tolerance. Small threshold differences can change discrete counts.
        Recorded configuration is not proof that manually altered caches match it.
        """
        if self.analysis.config is None:
            if supplied is None:
                return "threshold unknown"
            return f"threshold={_format_caption_threshold(supplied)} (caller supplied)"
        try:
            recorded = self.analysis.config.grade_spread_threshold
        except AttributeError as error:
            raise ValueError("Recorded grade_spread_threshold is unavailable.") from error
        _validate_caption_threshold(recorded, "recorded grade_spread_threshold")
        if supplied is not None and not bool(np.equal(np.asarray(recorded), np.asarray(supplied))):
            raise ValueError(
                f"Supplied threshold {_format_caption_threshold(supplied)} does not match "
                f"recorded threshold {_format_caption_threshold(recorded)}. "
                "Recompute statistics explicitly before plotting at a different threshold."
            )
        return f"threshold={_format_caption_threshold(recorded)}"

    def _preflight_grade_chart(
        self, threshold: Optional[float], show_measures: bool,
    ) -> str:
        """Validate chart inputs and resolve only the caption the view will consume.

        Parameters
        ----------
        threshold : float or None
            Optional explicit threshold assertion.
        show_measures : bool
            Whether marker values will be consumed.

        Returns
        -------
        str
            Resolved title suffix, or an empty string when no markers are displayed.

        Raises
        ------
        ValueError
            If selection, threshold or requested marker fields are invalid.
        """
        self.analysis._validate_report_selection()
        if threshold is not None:
            _validate_caption_threshold(threshold, "threshold")
        if not show_measures:
            return ""
        present = [self.analysis._validate_grade_pair(name) for name in ("adm", "fdm", "gdm")]
        if not any(present):
            return ""
        return f" - {self._grade_threshold_caption(threshold)}"

    def _preflight_plot_all(self) -> None:
        """Check planned result views before creating or saving the first figure.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a required field or chart-availability pair is invalid.

        Notes
        -----
        Required DC configuration is checked when debug-dependent frequency
        views are planned. Plot selection/order remains unchanged. Raw/debug fields
        are not subjected to nonnegative measure validation. This preflight
        does not provide rollback for unrelated rendering or file-system errors.
        """
        self.analysis._validate_point_view()
        self.analysis._validate_histogram_view()
        if self.analysis._validate_grade_pair("gdm"):
            self._preflight_grade_chart(None, True)
        if self.analysis.debug is not None:
            self._require_frequency_config()

    def _require_frequency_config(self) -> None:
        """Require recorded DC metadata before drawing frequency-dependent views.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If configuration or its break_point_dc value is unavailable.

        Notes
        -----
        This checks availability, without supplying defaults or changing the
        numerical interpretation of existing breakpoint values. General metadata
        validation and deep debug-cache consistency remain separate contracts.
        """
        config = getattr(self.analysis, "config", None)
        if getattr(config, "break_point_dc", None) is None:
            raise ValueError(
                "Recorded config.break_point_dc is required for spectrum and filter plots."
            )

    def _require_debug_data(self, plot_name: str) -> None:
        """
        Verify that debug data is available for plotting.
        
        Parameters
        ----------
        plot_name : str
            Name of the plot (for error message)
            
        Raises
        ------
        ValueError
            If debug data is not available
        """
        if self.analysis.debug is None:
            raise ValueError(
                f"Debug data required for {plot_name} plot. "
                "Run FSV analysis with store_debug=True"
            )

    # =========================================================================
    # Core Visualization Methods
    # =========================================================================

    @_close_failed_plot
    def plot_measures(
        self,
        title: str = "FSV Point-by-Point Results",
        figsize: Tuple[float, float] = FIG_SIZE,
        ax: Optional[np.ndarray] = None,
        style: Optional[Union[str, PlotStyle]] = None,
        show: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Plot stored ADM, FDM and base GDM with their mean lines and labels.

        Parameters
        ----------
        title : str, optional
            Figure title, applied only when this method creates the figure.
        figsize : tuple, optional
            Figure size (width, height) - ignored if ax is provided
        ax : np.ndarray of 3 matplotlib Axes, optional
            Distinct axes in one owning figure, shaped (3,), (1,3) or (3,1).
            Their order, geometry and caller ownership are retained.
        style : str, PlotStyle, or None
            Visual style configuration. Can be:
            - PlotStyle instance for full customization
            - Preset name: "classic", "dark", "publication"
            - None to use default style
        show : bool
            If True, calls plt.show() after plotting
        **kwargs
            Extra arguments for the three point-data lines. Do not duplicate
            explicitly supplied style arguments such as color or linewidth;
            use PlotStyle for those settings.

        Returns
        -------
        plt.Figure
            Figure containing three axes; caller-owned when ax is supplied.

        Raises
        ------
        ValueError
            If required point layouts, active measures, means, labels or selection
            are invalid, or if style names, numeric style fields or supplied axes
            are invalid.
            Result/axes preflight occurs before drawing.
        TypeError
            If style objects or numeric/Boolean style field types are unsupported,
            or forwarded keyword arguments conflict with explicit line arguments.

        Notes
        -----
        Excluded point values are hidden with NaN gaps at their original sample
        indices. Source arrays are not changed. Titles identify the active count;
        mean lines and categories use the stored selected statistics.
        Mean-line labels identify standard/custom classification when applicable;
        category-boundary lines and weighted-GDM curves are not drawn.
        Stored means are validated but not recomputed from the points. Unused
        VRS, histograms and optional weighted fields are not audited by this view.
            
        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=False).fsv(np.ones(64), np.full(64, 2.0))
        >>> with result.visualizer as viz:
        ...     figure = viz.plot_measures(style="publication")
        ...     print(len(figure.axes))
        3
        """
        self.analysis._validate_point_view()
        # Use helper methods
        selection = self.analysis._active_index()
        style = self._resolve_style(style)
        fig, axes, owns_figure = self._setup_axes(ax, figsize, 3, sharex=True)
        
        x = np.arange(len(self.analysis.adm_point))
        adm = self.analysis.adm_point
        fdm = self.analysis.fdm_point
        gdm = self.analysis.gdm_point
        if not isinstance(selection, slice):
            adm = np.where(selection, adm, np.nan)
            fdm = np.where(selection, fdm, np.nan)
            gdm = np.where(selection, gdm, np.nan)
        
        # ADM subplot
        axes[0].plot(
            x, adm,
            color=style.adm_color, alpha=style.line_alpha,
            linewidth=style.line_width, linestyle=style.line_style,
            marker=style.marker_style, markersize=style.marker_size if style.marker_style else None,
            **kwargs
        )
        axes[0].axhline(
            y=self.analysis.adm_mean, color=style.mean_color,
            linestyle=style.mean_line_style, linewidth=style.mean_line_width,
            label=f"Mean: {self.analysis.adm_mean:.4f} ({self.analysis._display_category('adm')})",
        )
        axes[0].set_ylabel("ADM", fontsize=style.label_size)
        axes[0].set_title("Amplitude Difference Measure (ADM)", 
                         fontsize=style.title_size, fontweight=style.title_weight)
        axes[0].legend(loc="upper right", fontsize=style.legend_size)
        if style.show_grid:
            axes[0].grid(True, alpha=style.grid_alpha, color=style.grid_color,
                        linestyle=style.grid_style, linewidth=style.grid_width)

        # FDM subplot
        axes[1].plot(
            x, fdm,
            color=style.fdm_color, alpha=style.line_alpha,
            linewidth=style.line_width, linestyle=style.line_style,
            marker=style.marker_style, markersize=style.marker_size if style.marker_style else None,
            **kwargs
        )
        axes[1].axhline(
            y=self.analysis.fdm_mean, color=style.mean_color,
            linestyle=style.mean_line_style, linewidth=style.mean_line_width,
            label=f"Mean: {self.analysis.fdm_mean:.4f} ({self.analysis._display_category('fdm')})",
        )
        axes[1].set_ylabel("FDM", fontsize=style.label_size)
        axes[1].set_title("Feature Difference Measure (FDM)",
                         fontsize=style.title_size, fontweight=style.title_weight)
        axes[1].legend(loc="upper right", fontsize=style.legend_size)
        if style.show_grid:
            axes[1].grid(True, alpha=style.grid_alpha, color=style.grid_color,
                        linestyle=style.grid_style, linewidth=style.grid_width)

        # GDM subplot
        axes[2].plot(
            x, gdm,
            color=style.gdm_color, alpha=style.line_alpha,
            linewidth=style.line_width, linestyle=style.line_style,
            marker=style.marker_style, markersize=style.marker_size if style.marker_style else None,
            **kwargs
        )
        axes[2].axhline(
            y=self.analysis.gdm_mean, color=style.mean_color,
            linestyle=style.mean_line_style, linewidth=style.mean_line_width,
            label=f"Mean: {self.analysis.gdm_mean:.4f} ({self.analysis._display_category('gdm')})",
        )
        axes[2].set_xlabel("Sample Index", fontsize=style.label_size)
        axes[2].set_ylabel("GDM", fontsize=style.label_size)
        axes[2].set_title("Global Difference Measure (GDM)",
                         fontsize=style.title_size, fontweight=style.title_weight)
        axes[2].legend(loc="upper right", fontsize=style.legend_size)
        if style.show_grid:
            axes[2].grid(True, alpha=style.grid_alpha, color=style.grid_color,
                        linestyle=style.grid_style, linewidth=style.grid_width)

        self._label_selection(axes)
        # Finalize and return
        return self._finalize_plot(fig, owns_figure, style, title, show)

    @_close_failed_plot
    def plot_histograms(
        self,
        title: str = "FSV Confidence Histograms",
        figsize: Tuple[float, float] = FIG_SIZE,
        ax: Optional[np.ndarray] = None,
        style: Optional[Union[str, PlotStyle]] = None,
        show: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Plot stored ADM, FDM and base-GDM category proportions as percentages.

        Parameters
        ----------
        title : str, optional
            Figure title, applied only to a newly created figure.
        figsize : tuple
            Figure size (width, height) - ignored if ax is provided
        ax : np.ndarray of 3 matplotlib Axes, optional
            Flat, row or column array of distinct axes in one owning figure.
            Their order, positions and caller ownership are retained. If None,
            create a new figure.
        style : str, PlotStyle, or None
            Visual style configuration
        show : bool
            If True, calls plt.show() after plotting
        **kwargs
            Extra bar arguments, excluding the explicitly supplied color,
            edgecolor and alpha.

        Returns
        -------
        plt.Figure
            Figure with three standard-category bar charts.

        Raises
        ------
        ValueError
            If required cached probability arrays, means, labels or selection
            are invalid, or if style names, numeric style fields or supplied axes
            are invalid.
            Result/axes preflight occurs before drawing.
        TypeError
            If style objects or numeric/Boolean style field types are unsupported,
            or forwarded arguments conflict with explicit bar arguments.

        Notes
        -----
        Probability checks reuse the existing rounding allowance without
        renormalization. Cached histograms are not recomputed from point arrays.
        Bars show 100 times the stored category proportions. Bin names and
        mean labels use standard categories even when configured classifications
        differ. These proportions are not statistical confidence intervals.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from kt_fsv import FSV
        >>> from kt_fsv.styles import PlotStyle
        >>> result = FSV().fsv(np.ones(64), np.full(64, 2.0))
        >>> fig, axes = plt.subplots(1, 3)
        >>> with result.visualizer as viz:
        ...     returned = viz.plot_histograms(ax=axes, style=PlotStyle(tight_layout=False))
        >>> returned is fig
        True
        >>> plt.fignum_exists(fig.number)
        True
        >>> plt.close(fig)
        """
        self.analysis._validate_histogram_view()
        # Use helper methods
        style = self._resolve_style(style)
        fig, axes, owns_figure = self._setup_axes(ax, figsize, (1, 3))
        
        categories = [name for name, _ in DEFAULT_CATEGORY_ITEMS]
        colors = [style.category_colors[cat] for cat in categories]
        x = np.arange(len(categories))

        for ax_subplot, hist, mean, name in [
            (axes[0], self.analysis.adm_conf_hist, self.analysis.adm_mean, "ADM"),
            (axes[1], self.analysis.fdm_conf_hist, self.analysis.fdm_mean, "FDM"),
            (axes[2], self.analysis.gdm_conf_hist, self.analysis.gdm_mean, "GDM"),
        ]:
            ax_subplot.bar(x, hist * 100, color=colors, edgecolor="black", 
                          alpha=style.line_alpha, **kwargs)
            ax_subplot.set_xticks(x)
            ax_subplot.set_xticklabels(categories, rotation=45, ha="right", 
                                      fontsize=style.label_size)
            ax_subplot.set_ylabel("Percentage (%)", fontsize=style.label_size)
            ax_subplot.set_title(f"{name} Confidence (standard categories)\nMean: {mean:.4f} ({_get_standard_category(mean)})",
                               fontsize=style.title_size, fontweight=style.title_weight)
            ax_subplot.set_ylim(0, 105)
            if style.show_grid:
                ax_subplot.grid(True, alpha=style.grid_alpha, axis="y", 
                              color=style.grid_color, linestyle=style.grid_style)

        self._label_selection(axes)
        # Finalize and return
        return self._finalize_plot(fig, owns_figure, style, title, show, [0, 0, 1, 0.95])

    @_close_failed_plot
    def plot_spectrum(
        self,
        title: str = "Spectrum Comparison",
        figsize: Tuple[float, float] = FIG_SIZE,
        ax: Optional[np.ndarray] = None,
        style: Optional[Union[str, PlotStyle]] = None,
        show: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Plot retained one-sided FFT magnitudes on linear and logarithmic panels.

        Requires debug data and configuration supplying break_point_dc.

        Parameters
        ----------
        title : str, optional
            Figure title, applied only to a newly created figure.
        figsize : tuple
            Figure size (width, height) - ignored if ax is provided
        ax : np.ndarray of 2 matplotlib Axes, optional
            Distinct axes in one figure, shaped (2,), (1,2) or (2,1).
            Supplied axes retain their order, geometry and caller ownership.
        style : str, PlotStyle, or None
            Visual style configuration
        show : bool
            If True, calls plt.show() after plotting
        **kwargs
            Extra spectrum-line arguments, excluding explicit color, alpha,
            linewidth and label arguments.

        Returns
        -------
        plt.Figure
            Figure with linear and logarithmic magnitude panels.

        Raises
        ------
        ValueError
            If debug data, required config.break_point_dc, the selection, a style
            name, numeric style fields or supplied axes are unavailable/invalid.
        AttributeError
            If other required debug attributes are unavailable.
        TypeError
            If style objects or numeric/Boolean style field types are unsupported,
            or forwarded arguments conflict with explicit line arguments.

        Notes
        -----
        The horizontal coordinate is FFT-bin index, from zero through
        len_fft // 2. Magnitudes are not calibrated to physical frequency or
        normalized as a separate spectral estimate. The logarithmic panel adds
        a fixed 1e-10 for display only; this is not a change to normalization.
        Selection does not mask frequency bins: the view retains full-analysis
        debug context and labels the selected count. Availability of
        config.break_point_dc is checked before drawing; broader numeric metadata
        policy and deep cache consistency are not audited here.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=True).fsv(np.ones(64), np.full(64, 2.0))
        >>> with result.visualizer as viz:
        ...     figure = viz.plot_spectrum()
        ...     print(len(figure.axes))
        2
        """
        self._require_debug_data("spectrum")
        self._require_frequency_config()
        
        # Use helper methods
        style = self._resolve_style(style)
        fig, axes, owns_figure = self._setup_axes(ax, figsize, 2, sharex=True)

        n_single = self.analysis.len_fft // 2 + 1
        freq_idx = np.arange(n_single)

        # Linear scale
        axes[0].plot(
            freq_idx, self.analysis.debug.magnitude1[:n_single],
            color=style.adm_color, alpha=style.line_alpha, 
            linewidth=style.line_width, label="Signal 1", **kwargs
        )
        axes[0].plot(
            freq_idx, self.analysis.debug.magnitude2[:n_single],
            color=style.fdm_color, alpha=style.line_alpha,
            linewidth=style.line_width, label="Signal 2", **kwargs
        )
        axes[0].axvline(
            x=self.analysis.config.break_point_dc, color=style.mean_color,
            linestyle=":", linewidth=style.mean_line_width,
            label=f"DC end: {self.analysis.config.break_point_dc}",
        )
        axes[0].axvline(
            x=self.analysis.break_point_lo_hi, color=style.gdm_color,
            linestyle="--", linewidth=style.mean_line_width,
            label=f"Breakpoint: {self.analysis.break_point_lo_hi}",
        )
        axes[0].set_ylabel("Magnitude", fontsize=style.label_size)
        axes[0].set_title("Magnitude Spectrum (Linear Scale)", 
                         fontsize=style.title_size, fontweight=style.title_weight)
        axes[0].legend(loc="upper right", fontsize=style.legend_size)
        if style.show_grid:
            axes[0].grid(True, alpha=style.grid_alpha, color=style.grid_color,
                        linestyle=style.grid_style)

        # Log scale
        axes[1].semilogy(
            freq_idx, self.analysis.debug.magnitude1[:n_single] + 1e-10,
            color=style.adm_color, alpha=style.line_alpha,
            linewidth=style.line_width, label="Signal 1", **kwargs
        )
        axes[1].semilogy(
            freq_idx, self.analysis.debug.magnitude2[:n_single] + 1e-10,
            color=style.fdm_color, alpha=style.line_alpha,
            linewidth=style.line_width, label="Signal 2", **kwargs
        )
        axes[1].axvline(x=self.analysis.config.break_point_dc, color=style.mean_color, 
                       linestyle=":", linewidth=style.mean_line_width)
        axes[1].axvline(x=self.analysis.break_point_lo_hi, color=style.gdm_color, 
                       linestyle="--", linewidth=style.mean_line_width)
        axes[1].set_xlabel("Frequency Index", fontsize=style.label_size)
        axes[1].set_ylabel("Magnitude (log)", fontsize=style.label_size)
        axes[1].set_title("Magnitude Spectrum (Log Scale)",
                         fontsize=style.title_size, fontweight=style.title_weight)
        axes[1].legend(loc="upper right", fontsize=style.legend_size)
        if style.show_grid:
            axes[1].grid(True, alpha=style.grid_alpha, color=style.grid_color,
                        linestyle=style.grid_style)

        self._label_selection(axes, context=True)
        # Finalize and return
        return self._finalize_plot(fig, owns_figure, style, title, show)

    @_close_failed_plot
    def plot_filters(
        self,
        title: str = "Filter Debug View",
        figsize: Tuple[float, float] = FIG_SIZE,
        ax: Optional[np.ndarray] = None,
        style: Optional[Union[str, PlotStyle]] = None,
        show: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Display the DC, Lo, Hi and combined filter shapes used by this profile.

        Requires debug data and configuration supplying break_point_dc.

        Parameters
        ----------
        title : str, optional
            Figure title, applied only to a newly created figure.
        figsize : tuple
            Figure size (width, height) - ignored if ax is provided
        ax : np.ndarray of matplotlib Axes, optional
            Exact (2,2) grid of distinct axes in one figure. Supplied axes retain
            their order, geometry and caller ownership.
        style : str, PlotStyle, or None
            Preset/custom style for the owned figure title and layout.
            Panel colors, line/grid settings and panel text styles are fixed.
        show : bool
            If True, calls plt.show() after plotting
        **kwargs
            Accepted but currently unused; no artist arguments are forwarded.

        Returns
        -------
        plt.Figure
            Figure with a two-by-two diagnostic grid.

        Raises
        ------
        ValueError
            If debug data, required config.break_point_dc, selection, style name,
            numeric style fields, axes or filter parameters are unavailable/invalid.
        AttributeError
            If other required debug attributes are unavailable.
        TypeError
            If style objects or numeric/Boolean style field types are unsupported.

        Notes
        -----
        Filters are reconstructed from len_fft and recorded breakpoints using
        the current filter functions; they are not retrieved from cached filter
        arrays. The horizontal coordinate is FFT-bin index. Panels show the
        first 20 DC bins and regions around the Lo/Hi breakpoint. Selected results
        retain full-analysis context, without applying a sample mask to bins.
        Availability of config.break_point_dc is checked before drawing.
        Existing breakpoint interpretation is retained without a new numeric
        metadata policy or deep cache-consistency audit.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=True).fsv(np.ones(64), np.full(64, 2.0))
        >>> with result.visualizer as viz:
        ...     figure = viz.plot_filters()
        ...     print(len(figure.axes))
        4
        """
        self._require_debug_data("filter")
        self._require_frequency_config()
        
        # Use helper methods
        style = self._resolve_style(style)
        fig, axes, owns_figure = self._setup_axes(ax, figsize, (2, 2))
        n_single = self.analysis.len_fft // 2 + 1
        freq_idx = np.arange(n_single)

        # Create filters
        dc_filter = filters.create_dc_filter(
            self.analysis.len_fft, self.analysis.config.break_point_dc
        )[:n_single]
        lo_filter = filters.create_lo_filter(
            self.analysis.len_fft,
            self.analysis.break_point_lo_hi,
            self.analysis.config.break_point_dc,
        )[:n_single]
        hi_filter = filters.create_hi_filter(
            self.analysis.len_fft, self.analysis.break_point_lo_hi
        )[:n_single]

        # DC filter
        axes[0, 0].stem(
            freq_idx[:20], dc_filter[:20], linefmt="b-", markerfmt="bo", basefmt="k-"
        )
        axes[0, 0].axvline(
            x=self.analysis.config.break_point_dc,
            color="r",
            linestyle="--",
            label=f"DC end: {self.analysis.config.break_point_dc}",
        )
        axes[0, 0].set_xlabel("Frequency Index")
        axes[0, 0].set_ylabel("Filter Value")
        axes[0, 0].set_title(f"DC Filter (first 20 bins)")
        axes[0, 0].legend(loc="upper right")
        axes[0, 0].grid(True, alpha=0.3)

        # Lo filter - 50 points centered around breakpoint
        lo_half_window = 25
        lo_xmin = max(self.analysis.config.break_point_dc, self.analysis.break_point_lo_hi - lo_half_window)
        lo_xmax = min(n_single, self.analysis.break_point_lo_hi + lo_half_window)
        axes[0, 1].plot(freq_idx, lo_filter, "g-", linewidth=2, label="Lo Filter")
        axes[0, 1].axvline(
            x=self.analysis.break_point_lo_hi,
            color="r",
            linestyle="--",
            label=f"Breakpoint: {self.analysis.break_point_lo_hi}",
        )
        axes[0, 1].set_xlabel("Frequency Index")
        axes[0, 1].set_ylabel("Filter Value")
        axes[0, 1].set_title("Lo (Low-Frequency) Filter")
        axes[0, 1].legend(loc="upper right")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(lo_xmin, lo_xmax)

        # Hi filter - 50 points centered around breakpoint
        hi_half_window = 25
        hi_xmin = max(0, self.analysis.break_point_lo_hi - hi_half_window)
        hi_xmax = min(n_single, self.analysis.break_point_lo_hi + hi_half_window)
        axes[1, 0].plot(freq_idx, hi_filter, "m-", linewidth=2, label="Hi Filter")
        axes[1, 0].axvline(
            x=self.analysis.break_point_lo_hi,
            color="r",
            linestyle="--",
            label=f"Breakpoint: {self.analysis.break_point_lo_hi}",
        )
        axes[1, 0].set_xlabel("Frequency Index")
        axes[1, 0].set_ylabel("Filter Value")
        axes[1, 0].set_title("Hi (High-Frequency) Filter")
        axes[1, 0].legend(loc="upper right")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(hi_xmin, hi_xmax)

        # Combined - 50 points centered around breakpoint
        combined = lo_filter + hi_filter
        combined_half_window = 25
        combined_xmin = max(0, self.analysis.break_point_lo_hi - combined_half_window)
        combined_xmax = min(n_single, self.analysis.break_point_lo_hi + combined_half_window)
        axes[1, 1].plot(freq_idx, lo_filter, "g-", linewidth=1.5, alpha=0.7, label="Lo")
        axes[1, 1].plot(freq_idx, hi_filter, "m-", linewidth=1.5, alpha=0.7, label="Hi")
        axes[1, 1].plot(
            freq_idx, combined, "k--", linewidth=1, alpha=0.5, label="Lo + Hi"
        )
        axes[1, 1].axvline(x=self.analysis.break_point_lo_hi, color="r", linestyle="--")
        axes[1, 1].set_xlabel("Frequency Index")
        axes[1, 1].set_ylabel("Filter Value")
        axes[1, 1].set_title("Combined Filter View")
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(combined_xmin, combined_xmax)

        self._label_selection(axes, context=True)
        # Finalize and return
        return self._finalize_plot(fig, owns_figure, style, title, show)

    @_close_failed_plot
    def plot_filtered_signals(
        self,
        title: str = "Filtered Signal Components",
        figsize: Tuple[float, float] = FIG_SIZE_TALL,
        ax: Optional[np.ndarray] = None,
        style: Optional[Union[str, PlotStyle]] = None,
        show: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Plot retained DC, Lo and Hi components against working-sample index.

        Requires debug data (store_debug=True during analysis).

        Parameters
        ----------
        title : str, optional
            Figure title, applied only to a newly created figure.
        figsize : tuple
            Figure size (width, height) - ignored if ax is provided
        ax : np.ndarray of matplotlib Axes, optional
            Exact (3,2) grid of distinct axes in one figure. Supplied axes retain
            their order, geometry and caller ownership.
        style : str, PlotStyle, or None
            Preset/custom style for the owned figure title and layout.
            Panel colors, line/grid settings and panel text styles are fixed.
        show : bool
            If True, calls plt.show() after plotting
        **kwargs
            Accepted but currently unused; no artist arguments are forwarded.

        Returns
        -------
        plt.Figure
            Figure with rows for DC, Lo and Hi and columns for responses 1 and 2.

        Raises
        ------
        ValueError
            If debug data, the selection, a style name, numeric style fields or
            supplied axes are invalid.
        TypeError
            If style objects or numeric/Boolean style field types are unsupported.

        Notes
        -----
        Components belong to the retained working-response domain; it is not
        necessarily time. Coordinates are sample indices, not physical units.
        Selected results show the original full-analysis debug context and a
        selected-count caption. No filtering or FSV computation is rerun.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=True).fsv(np.ones(64), np.full(64, 2.0))
        >>> with result.visualizer as viz:
        ...     figure = viz.plot_filtered_signals()
        ...     print(len(figure.axes))
        6
        """
        self._require_debug_data("filtered signals")
        
        # Use helper methods
        style = self._resolve_style(style)
        fig, axes, owns_figure = self._setup_axes(ax, figsize, (3, 2))
        x = np.arange(self.analysis.len_orig)

        # DC data
        axes[0, 0].plot(
            x, self.analysis.debug.dc_data1, "b-", alpha=0.7, label="Data 1 DC"
        )
        axes[0, 0].set_title("Signal 1 - DC Component")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].legend(loc="upper right")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(
            x, self.analysis.debug.dc_data2, "r-", alpha=0.7, label="Data 2 DC"
        )
        axes[0, 1].set_title("Signal 2 - DC Component")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].legend(loc="upper right")
        axes[0, 1].grid(True, alpha=0.3)

        # Lo data
        axes[1, 0].plot(
            x, self.analysis.debug.lo_data1, "b-", alpha=0.7, label="Data 1 Lo"
        )
        axes[1, 0].set_title("Signal 1 - Lo (Low-Frequency)")
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].legend(loc="upper right")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(
            x, self.analysis.debug.lo_data2, "r-", alpha=0.7, label="Data 2 Lo"
        )
        axes[1, 1].set_title("Signal 2 - Lo (Low-Frequency)")
        axes[1, 1].set_ylabel("Amplitude")
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].grid(True, alpha=0.3)

        # Hi data
        axes[2, 0].plot(
            x, self.analysis.debug.hi_data1, "b-", alpha=0.7, label="Data 1 Hi"
        )
        axes[2, 0].set_title("Signal 1 - Hi (High-Frequency)")
        axes[2, 0].set_xlabel("Sample Index")
        axes[2, 0].set_ylabel("Amplitude")
        axes[2, 0].legend(loc="upper right")
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(
            x, self.analysis.debug.hi_data2, "r-", alpha=0.7, label="Data 2 Hi"
        )
        axes[2, 1].set_title("Signal 2 - Hi (High-Frequency)")
        axes[2, 1].set_xlabel("Sample Index")
        axes[2, 1].set_ylabel("Amplitude")
        axes[2, 1].legend(loc="upper right")
        axes[2, 1].grid(True, alpha=0.3)

        self._label_selection(axes, context=True)
        # Finalize and return
        return self._finalize_plot(fig, owns_figure, style, title, show)

    @_close_failed_plot
    def plot_signals(
        self,
        title: str = "Input Signals",
        figsize: Tuple[float, float] = FIG_SIZE,
        ax: Optional[np.ndarray] = None,
        style: Optional[Union[str, PlotStyle]] = None,
        show: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Plot the two working responses retained in the debug snapshot.

        Requires debug data (store_debug=True during analysis).

        Parameters
        ----------
        title : str, optional
            Accepted but currently unused. Panel titles identify Data 1
            (Reference) and Data 2 (Test); no figure title is added.
        figsize : tuple
            Figure size (width, height) - ignored if ax is provided
        ax : np.ndarray of 2 matplotlib Axes, optional
            Distinct axes in one figure, shaped (2,), (1,2) or (2,1).
            Supplied axes retain their order, geometry and caller ownership.
        style : str, PlotStyle, or None
            Visual style configuration
        show : bool
            If True, calls plt.show() after plotting
        **kwargs
            Extra response-line arguments, excluding explicit color, alpha,
            linewidth and label arguments.

        Returns
        -------
        plt.Figure
            Figure with reference and test response panels.

        Raises
        ------
        ValueError
            If debug data, the selection, a style name, numeric style fields or
            supplied axes are invalid.
        TypeError
            If style objects or numeric/Boolean style field types are unsupported,
            or forwarded arguments conflict with explicit line arguments.

        Notes
        -----
        Coordinates are sample indices. After fsv_from_xy these are the prepared
        working responses, not the original unaligned raw records. Selected
        results retain the full debug inputs and identify the selected count.
        The method does not apply the sample mask or infer physical coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=True).fsv(np.ones(64), np.full(64, 2.0))
        >>> with result.visualizer as viz:
        ...     figure = viz.plot_signals()
        ...     print(len(figure.axes))
        2
        """
        self._require_debug_data("input signals")
        
        # Use helper methods
        style = self._resolve_style(style)
        fig, axes, owns_figure = self._setup_axes(ax, figsize, 2, sharex=True)

        x = np.arange(self.analysis.len_orig)

        # Data 1
        axes[0].plot(x, self.analysis.debug.data1, color=style.adm_color,
                    alpha=style.line_alpha, linewidth=style.line_width,
                    label="Data 1", **kwargs)
        axes[0].set_ylabel("Amplitude", fontsize=style.label_size)
        axes[0].set_title("Data 1 (Reference)", fontsize=style.title_size,
                         fontweight=style.title_weight)
        axes[0].legend(loc="upper right", fontsize=style.legend_size)
        if style.show_grid:
            axes[0].grid(True, alpha=style.grid_alpha, color=style.grid_color,
                        linestyle=style.grid_style)

        # Data 2
        axes[1].plot(x, self.analysis.debug.data2, color=style.fdm_color,
                    alpha=style.line_alpha, linewidth=style.line_width,
                    label="Data 2", **kwargs)
        axes[1].set_xlabel("Sample Index", fontsize=style.label_size)
        axes[1].set_ylabel("Amplitude", fontsize=style.label_size)
        axes[1].set_title("Data 2 (Test)", fontsize=style.title_size,
                         fontweight=style.title_weight)
        axes[1].legend(loc="upper right", fontsize=style.legend_size)
        if style.show_grid:
            axes[1].grid(True, alpha=style.grid_alpha, color=style.grid_color,
                        linestyle=style.grid_style)

        self._label_selection(axes, context=True)
        # Finalize and return (no suptitle for signals)
        return self._finalize_plot(fig, owns_figure, style, None, show, [0, 0, 1, 1])

    @_close_failed_plot
    def plot_grade_spread_chart(
        self,
        threshold: Optional[float] = None,
        title: str = "FSV Grade/Spread Color Code Chart",
        figsize: Tuple[float, float] = FIG_SIZE,
        show_measures: bool = True,
        ax: Optional[plt.Axes] = None,
        style: Optional[Union[str, PlotStyle]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Display stored Grade/Spread counts on the package's color chart.
        
        Parameters
        ----------
        threshold : float or None, optional
            Finite assertion in (0,1], exactly matching the recorded configuration
            when available. None uses the recorded value, or labels it unknown
            if configuration is absent. An explicit value without configuration
            is labelled caller supplied. Stored markers are never recomputed.
        title : str
            Plot title
        figsize : tuple
            Figure size - ignored if ax provided
        show_measures : bool
            Show ADM/FDM/GDM markers
        ax : matplotlib Axes, optional
            Custom axes
        style : str, PlotStyle, or None, optional
            Preset/custom style for fonts and layout. Region colors, markers
            and grid styling are fixed rather than taken from category_colors.
        show : bool
            Auto-display with plt.show()
        
        Returns
        -------
        plt.Figure
            Figure containing a single Grade/Spread axes.

        Raises
        ------
        ValueError
            If the selection, threshold argument or requested marker pairs are
            invalid. Both absent fields are allowed; partial pairs are rejected.
            A conflicting threshold assertion fails before figure/axes changes.
            Invalid style names, numeric style fields or supplied axes are rejected.
        TypeError
            If style objects or numeric/Boolean style field types are unsupported.

        Notes
        -----
        Grade/Spread is discussed in IEEE Std 1597.1-2022 §8.5(b). This view uses
        the package's fixed regions and stored counts; it does not establish a
        statistical confidence level or verify the source of cached results.
        With no displayed markers, no computation-threshold claim is made.
        Explicit arguments are still validated; unused marker/configuration
        fields are not inspected for a backdrop-only view. Scalar and percentage
        text preserve available numeric precision. Configuration identifies the
        recorded setting, not independently verified cache provenance.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV, FSVConfig
        >>> from kt_fsv.styles import PlotStyle
        >>> result = FSV(FSVConfig(grade_spread_threshold=0.95)).fsv(
        ...     np.ones(64), np.full(64, 2.0))
        >>> with result.visualizer as viz:
        ...     fig = viz.plot_grade_spread_chart(style=PlotStyle(tight_layout=False))
        ...     print(fig.axes[0].get_title())
        FSV Grade/Spread Color Code Chart - threshold=0.95 (95%)
        """
        caption = self._preflight_grade_chart(threshold, show_measures)
        # Use helper methods
        style = self._resolve_style(style)
        fig, ax_plot, owns_figure = self._setup_axes(ax, figsize, (1, 1))
        
        # Add color regions (from outermost to innermost)
        # Fixed chart regions; Grade/Spread reference: IEEE Std 1597.1-2022 §8.5(b)
        ax_plot.add_patch(Polygon(
            GRADE_SPREAD_REGION_POOR, closed=True,
            facecolor='red', edgecolor='none', alpha=1.0, zorder=1
        ))
        ax_plot.add_patch(Polygon(
            GRADE_SPREAD_REGION_FAIR, closed=True,
            facecolor='yellow', edgecolor='none', alpha=1.0, zorder=2
        ))
        ax_plot.add_patch(Polygon(
            GRADE_SPREAD_REGION_GOOD, closed=True,
            facecolor='white', edgecolor='none', alpha=1.0, zorder=3
        ))
        ax_plot.add_patch(Polygon(
            GRADE_SPREAD_REGION_EXCELLENT, closed=True,
            facecolor='blue', edgecolor='none', alpha=1.0, zorder=4
        ))
        
        # Add markers
        if show_measures:
            if self.analysis.adm_grade is not None and self.analysis.adm_spread is not None:
                ax_plot.scatter(
                    self.analysis.adm_spread,
                    self.analysis.adm_grade,
                    s=400,
                    c='black',
                    marker='+',
                    linewidths=3,
                    label=f'ADM (Grade={self.analysis.adm_grade}, Spread={self.analysis.adm_spread})',
                    zorder=10
                )
            
            if self.analysis.fdm_grade is not None and self.analysis.fdm_spread is not None:
                ax_plot.scatter(
                    self.analysis.fdm_spread,
                    self.analysis.fdm_grade,
                    s=400,
                    c='black',
                    marker='x',
                    linewidths=3,
                    label=f'FDM (Grade={self.analysis.fdm_grade}, Spread={self.analysis.fdm_spread})',
                    zorder=10
                )
            
            if self.analysis.gdm_grade is not None and self.analysis.gdm_spread is not None:
                ax_plot.scatter(
                    self.analysis.gdm_spread,
                    self.analysis.gdm_grade,
                    s=400,
                    facecolors='none',
                    edgecolors='black',
                    linewidths=2,
                    marker='o',
                    label=f'GDM (Grade={self.analysis.gdm_grade}, Spread={self.analysis.gdm_spread})',
                    zorder=10
                )
        
        # Configure axes
        ax_plot.set_xlabel('SPREAD (1=High Confidence → 6=Low Confidence)', fontsize=style.label_size, fontweight='bold')
        ax_plot.set_ylabel('GRADE (1=Excellent → 6=Very Poor)', fontsize=style.label_size, fontweight='bold')
        ax_plot.set_title(f'{title}{caption}', fontsize=style.title_size, fontweight=style.title_weight)
        
        # Set ticks and limits
        ax_plot.set_xticks(np.arange(0, 7))
        ax_plot.set_yticks(np.arange(0, 7))
        ax_plot.set_xlim(0, 6.5)
        ax_plot.set_ylim(0, 6.5)
        
        # Add grid
        ax_plot.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        self._label_selection(ax_plot)
        # Add legend if markers shown (special handling for grade/spread chart)
        if show_measures and (self.analysis.adm_grade is not None or 
                             self.analysis.fdm_grade is not None or 
                             self.analysis.gdm_grade is not None):
            ax_plot.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=style.legend_size, 
                          framealpha=0.9, markerscale=0.6, borderaxespad=0)
            # Custom tight_layout for legend space
            return self._finalize_plot(fig, owns_figure, style, None, show, [0, 0, 0.85, 1])
        else:
            # Standard finalization
            return self._finalize_plot(fig, owns_figure, style, None, show, [0, 0, 1, 1])

    @_close_failed_plot
    def plot_all(
        self,
        output_dir: Optional[Path] = None,
        prefix: str = "",
        dpi: int = 150,
        close_after_save: bool = True,
    ) -> List[plt.Figure]:
        """
        Generate available result views, optionally saving PNGs and closing them.

        Parameters
        ----------
        output_dir : str, Path, or None, optional
            Directory for PNG output; missing parents are created.
            None (default) creates figures without saving.
        prefix : str, optional
            Filename prefix, default empty.
        dpi : int, optional
            Saved-image resolution in dots per inch, default 150.
        close_after_save : bool, optional
            Close each successfully saved figure, default True. Has no effect
            when output_dir is None.

        Returns
        -------
        list of plt.Figure
            Figures in generation order. Entries may already be closed after
            saving or later context cleanup; the returned objects still exist.

        Raises
        ------
        ValueError
            If required result-view fields, DC configuration or chart-availability
            pairs are invalid, before creating figures or saving any batch output.
            Plot/render errors can also occur later.
        OSError
            If output-directory creation or file writing fails.
        AttributeError
            If required debug/configuration attributes are unavailable.

        Notes
        -----
        Order: signals (with debug), measures, histograms, grade_spread (with a
        complete GDM Grade/Spread pair), then spectrum, filters and filtered_signals
        (with debug). These names, prefix and ".png" form the saved filenames.
        The complete batch contains seven figures; both optional groups disabled
        leaves two. The batch uses each plot's default style and does not include
        a separate weighted-GDM view.

        Preflight checks planned measure/histogram/Grade-Spread views and required
        DC-configuration availability, not every numeric/debug/render/file-system
        condition or deep cache consistency. On failure, newly tracked batch
        figures are closed, including earlier views left open after saving.
        Existing helper/caller figures are retained. Completed and partially
        written files remain; see the class cleanup boundary. No result arrays
        or statistics are recalculated.

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from kt_fsv import FSV
        >>> result = FSV(store_debug=True).fsv(np.ones(64), np.full(64, 2.0))
        >>> with TemporaryDirectory() as folder, result.visualizer as viz:
        ...     figures = viz.plot_all(output_dir=folder, prefix="fsv_")
        ...     print(len(figures), len(list(Path(folder).glob("fsv_*.png"))))
        ...     print(any(plt.fignum_exists(fig.number) for fig in figures))
        7 7
        False
        """
        self._preflight_plot_all()
        figures = []
        plot_info = [
            ("measures", self.plot_measures, "Point-by-Point Results"),
            ("histograms", self.plot_histograms, "Confidence Histograms"),
        ]
        
        # Add grade/spread chart if available
        if self.analysis.gdm_grade is not None and self.analysis.gdm_spread is not None:
            plot_info.append(("grade_spread", self.plot_grade_spread_chart, "Grade/Spread Chart"))

        # Add debug-dependent plots if available
        if self.analysis.debug is not None:
            plot_info.insert(0, ("signals", self.plot_signals, "Input Signals"))
            plot_info.extend(
                [
                    ("spectrum", self.plot_spectrum, "Spectrum Comparison"),
                    ("filters", self.plot_filters, "Filter Debug View"),
                    (
                        "filtered_signals",
                        self.plot_filtered_signals,
                        "Filtered Signals",
                    ),
                ]
            )

        for name, plot_func, title in plot_info:
            fig = plot_func(title=title)
            figures.append(fig)

            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{prefix}{name}.png" if prefix else f"{name}.png"
                fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
                if close_after_save:
                    plt.close(fig)
                    # Remove from tracked figures since we closed it
                    if fig in self._figures:
                        self._figures.remove(fig)

        return figures
