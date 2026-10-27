"""
FSV (Feature Selective Validation) main class.

Compare paired real, one-dimensional responses using the package's documented
FSV implementation profile based on IEEE Std 1597.1-2022.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .analysis import FSVAnalysis, FSVDebugData
from .config import FSVConfig
from . import spectrum, filters, measures, constants
from .preprocessing import _as_real_float_array, build_working_data_sets


# =============================================================================
# Constants
# =============================================================================

# Signal length requirements per IEEE Std 1597.1-2022
MIN_SIGNAL_LENGTH: int = constants.MIN_SIGNAL_LENGTH  # Absolute minimum (IEEE requirement)
RECOMMENDED_MIN_LENGTH: int = constants.RECOMMENDED_MIN_LENGTH  # Recommended minimum for reliable results

# Practical upper limit to prevent memory exhaustion
# ~1.2 GB RAM per 10M points in typical FSV analysis
MAX_SIGNAL_LENGTH: int = constants.MAX_SIGNAL_LENGTH  # 100 million points (~12 GB RAM limit)

# Note: FSV algorithm thresholds (break_point_dc, threshold_lo_hi, etc.)
# are now configured via FSVConfig instead of constants


class FSV:
    """
    Compare paired real, one-dimensional responses using FSV.

    Each call returns a new FSVAnalysis snapshot. The engine retains its
    configuration and debug-storage setting, but not the previous result.
    The supported implementation profile is based on IEEE Std 1597.1-2022;
    using it does not establish conformance of a complete validation study.

    Parameters
    ----------
    config : FSVConfig, optional
        Configuration for FSV algorithm parameters (thresholds, breakpoints,
        normalization, zero_pad, etc.).
        If None, constructs the package reference configuration with FSVConfig().
        See FSVConfig documentation for customization options.
    store_debug : bool, optional
        Whether to store intermediate computation data for debugging
        and visualization. Default True. Access via analysis.debug. False
        omits retained debug data; measure and histogram plots still work.

    Returns
    -------
    FSV
        Engine configured to produce independent result snapshots.

    Notes
    -----
    Algorithm options belong in FSVConfig; fsv() accepts only the two working
    data sets. Use fsv_from_xy() when coordinates need overlap selection and
    alignment first. Plots are available through analysis.visualizer.

    Concurrent calls require fixed engine settings and inputs that callers do
    not modify during analysis. FSVConfig owns read-only category metadata and
    immutable scalar settings; returned result/debug records are snapshots.
    The engine itself remains mutable. Prefer a new engine and configuration
    for new settings. Visualization follows Matplotlib's threading limitations;
    serialize access to visualization methods.

    Examples
    --------
    >>> import numpy as np
    >>> from kt_fsv import FSV, FSVConfig
    >>> t = np.linspace(0.0, 1.0, 200)
    >>> signal1 = np.sin(2 * np.pi * 5 * t)
    >>> signal2 = signal1.copy()
    >>> fsv = FSV()
    >>> analysis = fsv.fsv(signal1, signal2)
    >>> print(analysis.gdm_category)
    Excellent
    >>> 
    >>> # Change the cumulative spectral-magnitude target
    >>> config = FSVConfig(threshold_lo_hi=0.5)
    >>> fsv = FSV(config=config)
    >>> analysis = fsv.fsv(signal1, signal2)
    >>>
    >>> # Custom normalization
    >>> config = FSVConfig(normalization='N')
    >>> fsv = FSV(config=config)

    """

    def __init__(self, config: Optional[FSVConfig] = None, store_debug: bool = True) -> None:
        """
        Initialize the FSV instance.

        Parameters
        ----------
        config : FSVConfig, optional
            Configuration for FSV algorithm parameters.
            If None, constructs FSVConfig() using the package reference settings.
        store_debug : bool, optional
            Whether to store intermediate computation data for debugging
            and visualization. Default True. Access via analysis.debug.
            Retained working inputs are copied into read-only debug snapshots,
            adding 16 bytes per sample for two float64 inputs. False retains
            no debug inputs. Computed result buffers are read-only.

        Returns
        -------
        None
        """
        self.config = config or FSVConfig()
        self.store_debug = store_debug

    def __str__(self) -> str:
        """Describe whether the engine uses the canonical package configuration.

        Returns
        -------
        str
            Default only when settings and scalar types match FSVConfig().
            Custom includes representation-only differences and is not a
            numerical-deviation or IEEE-conformance verdict.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV, FSVConfig
        >>> str(FSV())
        'FSV(config=default)'
        >>> str(FSV(FSVConfig(grade_spread_threshold=np.float64(0.85))))
        'FSV(config=custom)'
        """
        config_str = "default" if self.config == FSVConfig() else "custom"
        return f"FSV(config={config_str})"

    def __repr__(self) -> str:
        """Return the engine's configuration representation.

        Returns
        -------
        str
            Engine class name and configuration representation.
        """
        return f"{self.__class__.__name__}(config={self.config})"

    def _validate_inputs(
        self,
        data_set_1: ArrayLike,
        data_set_2: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate input datasets before FSV analysis.

        Parameters
        ----------
        data_set_1 : array-like
            First dataset for comparison
        data_set_2 : array-like
            Second dataset for comparison

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Validated datasets as numpy arrays

        Raises
        ------
        ValueError
            If datasets contain unknown masked entries, complex-typed values
            (including zero imaginary parts), or fail the existing shape, length
            or finite-value checks.
        """
        data_set_1 = _as_real_float_array(data_set_1, "data_set_1")
        data_set_2 = _as_real_float_array(data_set_2, "data_set_2")

        if data_set_1.ndim != 1 or data_set_2.ndim != 1:
            raise ValueError("Datasets must be 1D arrays")

        if len(data_set_1) != len(data_set_2):
            raise ValueError(
                f"Datasets must have equal length. "
                f"Got {len(data_set_1)} and {len(data_set_2)}"
            )

        if len(data_set_1) < MIN_SIGNAL_LENGTH:
            raise ValueError(
                f"Minimum {MIN_SIGNAL_LENGTH} data points required per IEEE Std 1597.1-2022. "
                f"Got {len(data_set_1)}"
            )

        if len(data_set_1) > MAX_SIGNAL_LENGTH:
            warnings.warn(
                f"Signal length {len(data_set_1):,} exceeds recommended maximum "
                f"of {MAX_SIGNAL_LENGTH:,} points. This may require significant memory "
                f"(~{len(data_set_1) * 12 / 1e9:.1f} GB RAM). "
                f"Consider downsampling or processing in chunks.",
                UserWarning,
                stacklevel=2
            )

        if len(data_set_1) < RECOMMENDED_MIN_LENGTH:
            warnings.warn(
                f"Dataset length {len(data_set_1)} is below recommended minimum "
                "of 50-100 points. Results may be less reliable.",
                UserWarning,
            )

        if np.any(~np.isfinite(data_set_1)) or np.any(~np.isfinite(data_set_2)):
            raise ValueError("Datasets contain NaN or Inf values")

        return data_set_1, data_set_2

    def fsv(
        self,
        data_set_1: ArrayLike,
        data_set_2: ArrayLike,
    ) -> FSVAnalysis:
        """
        Compute FSV (Feature Selective Validation) for two datasets.

        Calculate ADM, FDM and GDM under the documented implementation profile,
        with configured category-distribution and weighted-GDM outputs.

        Step a) (overlap detection and interpolation onto a common
        independent-variable grid) is NOT performed here: ``data_set_1`` and
        ``data_set_2`` must already be *working data sets* -- dependent (y)
        values sampled on the same coincident grid, hence of equal length.
        Use ``fsv_from_xy`` or ``kt_fsv.build_working_data_sets`` to prepare raw
        ``(x, y)`` data. The direct core receives no coordinates and cannot verify
        that the input sample locations coincide.

        Parameters
        ----------
        data_set_1 : array_like
            First real numeric working data set: dependent (y) values on the
            common grid, supplied as a one-dimensional non-complex array/sequence
            of at least 32 finite values. Unknown masked entries are rejected;
            fully unmasked wrappers follow the ordinary input rules.
        data_set_2 : array_like
            Second real numeric working data set on the same grid, equal in
            length to ``data_set_1``. Real integer/float inputs are converted
            to float64; complex storage is rejected even for zero imaginary parts.

        Returns
        -------
        FSVAnalysis
            Snapshot containing pointwise ADM/FDM/GDM, their means, standard
            category-proportion histograms, VRS values and configuration metadata.
            Point arrays have the original working-data length and are read-only.
            Grade/Spread fields are None when compute_grade_spread=False.
            Weighted outputs and coefficients are None unless both
            compute_grade_spread and use_weighted_gdm are True. Debug data are
            None when store_debug=False. Use result.visualizer for plotting;
            plot methods are not methods of FSVAnalysis itself.

        Raises
        ------
        ValueError
            If an input contains unknown masked entries, has complex dtype or
            complex values in object storage (including zero-imaginary values),
            fails existing input validation, a required reduction/norm operand
            is nonfinite or non-real, or an internal transform receives
            nonfinite components.
        FloatingPointError
            If transform, norm or reduction recovery cannot safely represent a required result.
            A required denominator may be unrepresentable even if its ratio would fit.

        Warns
        -----
        UserWarning
            When zero_pad=True, when the working length is 32 through 49,
            or when it exceeds the resource-warning threshold of 100 million
            points. That threshold is not a hard limit or a memory guarantee.
        RuntimeWarning
            After successful FFT/IFFT range recovery or scaled recovery of a denominator, scalar mean,
            breakpoint calculation or GDM norm. One analysis can recover at
            multiple stages; warning filters, including escalation, still apply.
            Separately, up to one ADM and one FDM epsilon-regularization warning
            identify floored denominator families with nonzero numerators. These
            can occur for ordinary signals with roundoff-level spectral residues;
            they do not claim that results are inaccurate or numerically recovered.

        Notes
        -----
        NumPy masked entries are rejected before conversion, including entries
        nested in Python lists/tuples or object arrays. Nomask/all-False wrappers
        remain subject to real/finite/shape/length checks. Missing samples require
        explicit caller resolution; no filling, dropping or compression is applied.

        Complex-valued comparisons use the explicit IEEE section 8.5(c) workflow:
        compare real/imaginary or magnitude/phase components separately, then
        pass both FSVAnalysis results to combine_multidof(). This method does not
        select a representation or discard imaginary parts automatically. A caller
        choosing a real component must do so explicitly; internal FFT spectra
        still retain their complex coefficients.

        The absolute epsilon floor is retained as an explicit implementation
        limitation. It can introduce unit/scale dependence. Equivalent common
        unit conversion by a factor k requires epsilon to scale by abs(k), within
        numerical range/precision limits. Record epsilon, units and preprocessing
        in the validation report; warning messages are not stored in FSVAnalysis.
        Do not normalize datasets independently merely to silence a diagnostic.

        IEEE Std 1597.1-2022 section 8.4(b2) says next-power-of-two working-data
        enlargement shall be avoided. The package keeps zero_pad=False by default
        and exposes zero_pad=True as an explicit deviation that warns at analysis
        time. It is separate from inverse spectral zero-filling and derivative
        endpoint padding.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> from kt_fsv.postprocessing import combine_multidof
        >>> engine = FSV(store_debug=False)
        >>> first = np.full(64, 1 + 1j)
        >>> second = np.full(64, 1 + 2j)
        >>> real_result = engine.fsv(first.real, second.real)
        >>> imag_result = engine.fsv(first.imag, second.imag)
        >>> combined = combine_multidof(
        ...     real_result, imag_result, k=0.5,
        ...     component1_name="Real", component2_name="Imaginary",
        ... )
        >>> combined.gdm_mean > 0
        True
        """
        if self.config.zero_pad:
            warnings.warn(
                "Zero padding is enabled. "
                "IEEE Std 1597.1-2022 recommends avoiding zero padding as it "
                "can introduce significant variability in results.",
                UserWarning,
            )

        # Validate the (pre-aligned) working data sets. IEEE step a)
        # -- overlap detection + interpolation -- is the caller's
        # responsibility; see kt_fsv.build_working_data_sets.
        data1, data2 = self._validate_inputs(data_set_1, data_set_2)
        len_orig = len(data1)

        # Step b: Compute FFT (magnitude spectrums) for both datasets
        _, magnitude1, fft_result1, _ = spectrum.compute_magnitude_spectrum(
            data1, self.config.normalization, self.config.zero_pad
        )
        _, magnitude2, fft_result2, _ = spectrum.compute_magnitude_spectrum(
            data2, self.config.normalization, self.config.zero_pad
        )
        len_fft = len(fft_result1)

        # Step c.1: Compute DC components and time domain signals
        dc_filter = filters.create_dc_filter(len_fft, self.config.break_point_dc)
        dc_fft1 = fft_result1 * dc_filter
        dc_fft2 = fft_result2 * dc_filter
        dc_data1 = spectrum.inverse_spectrum(dc_fft1, len_orig, self.config.normalization)
        dc_data2 = spectrum.inverse_spectrum(dc_fft2, len_orig, self.config.normalization)

        # Step c.2-4: Compute breakpoint
        break_point_lo_hi, _, _ = filters.compute_breakpoint(
            magnitude1, magnitude2, len_fft, self.config.break_point_dc, self.config.threshold_lo_hi
        )

        # Step c.5-6, d: Compute Lo and Hi filtered signals
        lo_filter = filters.create_lo_filter(
            len_fft, break_point_lo_hi, self.config.break_point_dc
        )
        hi_filter = filters.create_hi_filter(len_fft, break_point_lo_hi)

        lo_data1 = spectrum.inverse_spectrum(
            fft_result1 * lo_filter, len_orig, self.config.normalization
        )
        lo_data2 = spectrum.inverse_spectrum(
            fft_result2 * lo_filter, len_orig, self.config.normalization
        )
        hi_data1 = spectrum.inverse_spectrum(
            fft_result1 * hi_filter, len_orig, self.config.normalization
        )
        hi_data2 = spectrum.inverse_spectrum(
            fft_result2 * hi_filter, len_orig, self.config.normalization
        )

        # Step e: Compute ADM
        adm_point = measures.compute_adm(
            lo_data1, lo_data2, dc_data1, dc_data2, epsilon=self.config.epsilon_floor
        )
        adm_mean = measures._guarded_mean(adm_point, "ADM mean")
        adm_counts = measures.compute_confidence_counts(adm_point)
        adm_conf_hist = measures._histogram_from_counts(adm_counts)
        adm_visual = float(measures.convert_to_vrs(np.array([adm_mean]))[0])

        # Step h-i: Compute FDM
        fdm_point = measures.compute_fdm(
            lo_data1, lo_data2, hi_data1, hi_data2, len_orig, epsilon=self.config.epsilon_floor
        )
        fdm_mean = measures._guarded_mean(fdm_point, "FDM mean")
        fdm_counts = measures.compute_confidence_counts(fdm_point)
        fdm_conf_hist = measures._histogram_from_counts(fdm_counts)
        fdm_visual = float(measures.convert_to_vrs(np.array([fdm_mean]))[0])

        # Step l: Compute GDM
        gdm_point = measures.compute_gdm(adm_point, fdm_point)
        gdm_mean = measures._guarded_mean(gdm_point, "GDM mean")
        gdm_counts = measures.compute_confidence_counts(gdm_point)
        gdm_conf_hist = measures._histogram_from_counts(gdm_counts)
        gdm_visual = float(measures.convert_to_vrs(np.array([gdm_mean]))[0])

        # Optional: Compute Grade and Spread (IEEE §8.4b)
        adm_grade = adm_spread = None
        fdm_grade = fdm_spread = None
        gdm_grade = gdm_spread = None
        if self.config.compute_grade_spread:
            adm_grade, adm_spread = measures.compute_grade_spread(
                threshold=self.config.grade_spread_threshold, counts=adm_counts,
            )
            fdm_grade, fdm_spread = measures.compute_grade_spread(
                threshold=self.config.grade_spread_threshold, counts=fdm_counts,
            )
            gdm_grade, gdm_spread = measures.compute_grade_spread(
                threshold=self.config.grade_spread_threshold, counts=gdm_counts,
            )

        # Optional: Compute Weighted GDM (IEEE §8.4b)
        gdm_weighted = gdm_weighted_mean = K_ADM = K_FDM = None
        if self.config.use_weighted_gdm and self.config.compute_grade_spread:
            gdm_weighted, K_ADM, K_FDM = measures.compute_weighted_gdm(
                adm_point, fdm_point, adm_spread, fdm_spread
            )
            gdm_weighted_mean = measures._guarded_mean(gdm_weighted, "weighted GDM mean")

        # Build debug data if requested
        debug = None
        if self.store_debug:
            debug = FSVDebugData._from_owned_arrays(
                data1=data1.copy(),
                data2=data2.copy(),
                magnitude1=magnitude1,
                magnitude2=magnitude2,
                dc_data1=dc_data1,
                dc_data2=dc_data2,
                lo_data1=lo_data1,
                lo_data2=lo_data2,
                hi_data1=hi_data1,
                hi_data2=hi_data2,
            )

        # Create and return FSVAnalysis
        return FSVAnalysis._from_owned_arrays(
            adm_point=adm_point,
            adm_mean=adm_mean,
            adm_conf_hist=adm_conf_hist,
            adm_visual=adm_visual,
            fdm_point=fdm_point,
            fdm_mean=fdm_mean,
            fdm_conf_hist=fdm_conf_hist,
            fdm_visual=fdm_visual,
            gdm_point=gdm_point,
            gdm_mean=gdm_mean,
            gdm_conf_hist=gdm_conf_hist,
            gdm_visual=gdm_visual,
            len_orig=len_orig,
            len_fft=len_fft,
            break_point_lo_hi=break_point_lo_hi,
            config=self.config,
            adm_grade=adm_grade,
            adm_spread=adm_spread,
            fdm_grade=fdm_grade,
            fdm_spread=fdm_spread,
            gdm_grade=gdm_grade,
            gdm_spread=gdm_spread,
            gdm_weighted=gdm_weighted,
            gdm_weighted_mean=gdm_weighted_mean,
            K_ADM=K_ADM,
            K_FDM=K_FDM,
            debug=debug,
        )

    def fsv_from_xy(
        self,
        x1: ArrayLike,
        y1: ArrayLike,
        x2: ArrayLike,
        y2: ArrayLike,
        *,
        kind: str = "linear",
    ) -> FSVAnalysis:
        """
        Align two raw ``(x, y)`` data sets and compare their working responses.

        Convenience wrapper that first performs step a) -- overlap detection and
        interpolation onto a common independent-variable grid via
        :func:`kt_fsv.build_working_data_sets` -- and then runs the FSV core
        (:meth:`fsv`) on the resulting working data sets. Inputs may span
        different ranges, use different sampling, and be unsorted.

        Parameters
        ----------
        x1, y1 : array_like
            Real, finite, one-dimensional coordinate and response arrays of
            equal length for the first input. Coordinates must be unique;
            unsorted inputs are sorted together. Unknown masked entries and
            complex storage are rejected. Fully unmasked wrappers follow the
            ordinary input rules.
        x2, y2 : array_like
            Coordinate and response arrays for the second input, subject to the
            same rules. The original grids and lengths may differ. Each input
            must contribute at least two points within a nonzero overlap;
            the resulting working grid must contain at least 32 points for FSV.
        kind : str, optional
            Interpolation scheme passed to step a). IEEE §8.4.a6 mandates
            ``"linear"`` (the default and only supported value).

        Returns
        -------
        FSVAnalysis
            FSV results for the aligned working data sets (see :meth:`fsv`).
            The common coordinates and alignment metadata are not stored in this
            result; use build_working_data_sets separately when they are needed.

        Raises
        ------
        ValueError
            If any x/y input contains unknown masked or complex-typed values,
            step a) fails (e.g. no overlap, non-unique or non-finite inputs), or
            the aligned working data sets are invalid for :meth:`fsv`.
        TypeError
            If preprocessing cannot convert an input representation to float64.
        OverflowError
            If a value overflows during preprocessing conversion.
        FloatingPointError
            If the core cannot recover a required numerical result; see :meth:`fsv`.

        Warns
        -----
        UserWarning
            For the core's padding or working-length advisories; see :meth:`fsv`.
        RuntimeWarning
            On successful FFT/IFFT or scaled arithmetic/norm recovery, or when the core
            reports epsilon regularization with nonzero numerators. See :meth:`fsv`.

        Notes
        -----
        Preparation converts coordinates/responses to float64, sorts each pair
        together and checks uniqueness in that representation. Unknown masked
        entries are rejected before conversion, even outside the eventual overlap.
        Fully unmasked wrappers retain ordinary validation. Resolve missing samples
        explicitly before use; no automatic missing-data cleaning is performed.

        The selected common grid can remain nonuniform and can miss features
        present only in the denser input. The core receives sample-indexed
        responses, not their coordinates. Assess spacing and graphical fidelity
        and retain any modifications in the study record. Use the separate
        builder when the common grid and alignment diagnostics must be retained.

        For complex dependent data, choose real/imaginary or magnitude/phase
        explicitly, call this method separately for the component pairs, then
        combine the FSVAnalysis results with combine_multidof(). Independent
        coordinates must remain real, and combined results must share aligned
        point locations. No automatic component selection is performed.

        See Also
        --------
        fsv : FSV core for data already on a common grid.
        kt_fsv.build_working_data_sets : Step a) used internally.

        Examples
        --------
        >>> import numpy as np
        >>> from kt_fsv import FSV
        >>> x1 = np.linspace(0.0, 1.0, 300)
        >>> x2 = np.linspace(0.05, 1.05, 120)
        >>> analysis = FSV().fsv_from_xy(x1, np.sin(6 * x1), x2, np.sin(6 * x2))
        >>> analysis.len_orig
        114

        References
        ----------
        IEEE Std 1597.1-2022, Section 8.4 step a): "Generation of the working
        data sets".
        """
        working = build_working_data_sets(x1, y1, x2, y2, kind=kind)
        return self.fsv(working.y1, working.y2)
