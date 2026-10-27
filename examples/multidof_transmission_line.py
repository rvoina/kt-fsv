"""Compare loaded coaxial lines using S11 magnitude/phase and multi-DOF FSV.

Notes
-----
These are synthetic 0.30 m lines with nominal characteristic impedances of
50 and 45 ohms, both terminated in 75 ohms. They share a frequency grid and
50-ohm port references. Magnitude is 20*log10(abs(S11)); phase is independently
unwrapped in radians and converted to degrees. The chosen sweep starts both
phases on the same branch. No phase-offset correction or dB floor is applied.

The equal-weight combination averages component ADM/FDM arrays and derives
GDM from them. It does not average the component GDM means. This demonstrates
related-component combination, not native complex or multidimensional FSV.

Examples
--------
Run after installing the optional examples dependency::

    python -m pip install ".[examples]"
    python examples/multidof_transmission_line.py --output-dir outputs/multidof
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from skrf.media import Coaxial

from kt_fsv import FSV, FSVAnalysis
from kt_fsv.postprocessing import combine_multidof

PORT_IMPEDANCE_OHM = 50.0
MAGNITUDE_WEIGHT = 0.5
LINE_CASES = {
    "Reference": {"length_m": 0.30, "load_ohm": 75.0, "nominal_z0_ohm": 50.0},
    "Modified": {"length_m": 0.30, "load_ohm": 75.0, "nominal_z0_ohm": 45.0},
}
COAX_PARAMETERS = {
    "Dout": 0.003,
    "epsilon_r": 2.1,
    "tan_delta": 0.0002,
    "sigma": 5.8e7,
}


def make_networks() -> dict[str, rf.Network]:
    """Create two loaded coaxial-line networks using scikit-rf.

    Returns
    -------
    dict of str to skrf.Network
        Reference and modified one-port networks on 501 points from 0.1 to
        3 GHz. Both are 0.30 m long, terminated in 75 ohms and referenced to
        50 ohms. Their nominal characteristic impedances are 50 and 45 ohms.

    Notes
    -----
    Both coaxial media have a 3 mm outer-conductor inner diameter, relative
    permittivity 2.1, loss tangent 0.0002 and copper conductivity 5.8e7 S/m.
    Each nominal impedance determines a different inner-conductor diameter.
    The lossy media calculate their frequency-dependent complex characteristic
    impedances and propagation constants; port references remain 50 ohms.
    """
    frequency = rf.Frequency(0.1, 3.0, 501, unit="ghz")
    networks = {}
    for name, case in LINE_CASES.items():
        coax = Coaxial.from_Z0_Dout(
            frequency=frequency,
            z0=case["nominal_z0_ohm"],
            z0_port=PORT_IMPEDANCE_OHM,
            **COAX_PARAMETERS,
        )
        gamma_load = (
            (case["load_ohm"] - PORT_IMPEDANCE_OHM)
            / (case["load_ohm"] + PORT_IMPEDANCE_OHM)
        )
        line = coax.line(case["length_m"], unit="m")
        load = coax.load(gamma_load, z0=PORT_IMPEDANCE_OHM)
        networks[name] = line ** load
        networks[name].name = (
            f"{name} ({case['nominal_z0_ohm']:g} ohm line, "
            f"{case['length_m']:.2f} m, {case['load_ohm']:g} ohm load)"
        )
    return networks


def run_demo(output_dir: Path) -> dict[str, FSVAnalysis]:
    """Compare S11 components, combine FSV and save the plot and processing recipe.

    Parameters
    ----------
    output_dir : pathlib.Path
        Destination directory, created if necessary.

    Returns
    -------
    dict of str to FSVAnalysis
        ``magnitude_db``, ``phase_deg`` and ``combined`` comparisons.

    Raises
    ------
    ValueError
        If a generated S11 contains nonfinite or zero values. A zero reflection
        has no finite dB magnitude or defined phase.
    OSError
        If the output directory or files cannot be written.
    """
    networks = make_networks()
    components = {}
    for name, network in networks.items():
        s11 = network.s[:, 0, 0]
        magnitude = np.abs(s11)
        if not np.all(np.isfinite(s11)) or np.any(magnitude == 0):
            raise ValueError("S11 must be finite and nonzero for dB/phase FSV.")
        components[name] = (
            20.0 * np.log10(magnitude),
            np.rad2deg(np.unwrap(np.angle(s11))),
        )

    calculator = FSV()
    magnitude = calculator.fsv(components["Reference"][0], components["Modified"][0])
    phase = calculator.fsv(components["Reference"][1], components["Modified"][1])
    combined = combine_multidof(magnitude, phase, k=MAGNITUDE_WEIGHT)
    results = {"magnitude_db": magnitude, "phase_deg": phase, "combined": combined}
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, layout="constrained"
    )
    try:
        for name, (magnitude_db, phase_deg) in components.items():
            network = networks[name]
            axes[0].plot(network.f / 1e9, magnitude_db, label=network.name)
            axes[1].plot(network.f / 1e9, phase_deg, label=network.name)
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        axes[0].set_ylabel("S11 magnitude (dB)")
        axes[1].set_ylabel("S11 phase (degrees, unwrapped)")
        axes[1].set_xlabel("Frequency (GHz)")
        axes[0].set_title(f"Magnitude GDM: {magnitude.gdm_mean:.4f}")
        axes[1].set_title(f"Phase GDM: {phase.gdm_mean:.4f}")
        figure.suptitle(
            "Multi-DOF FSV: loaded coaxial transmission lines\n"
            f"Combined GDM: {combined.gdm_mean:.4f} (k={MAGNITUDE_WEIGHT:g})"
        )
        figure.savefig(output_dir / "multidof_s11.png", dpi=150)
    finally:
        plt.close(figure)

    summary = {
        "data": "synthetic scikit-rf Coaxial networks",
        "line_cases": LINE_CASES,
        "coax_parameters_SI": COAX_PARAMETERS,
        "port_impedance_ohm": PORT_IMPEDANCE_OHM,
        "frequency_ghz": [0.1, 3.0],
        "points": 501,
        "representations": {
            "magnitude": "20*log10(abs(S11)), dB; no clipping/floor",
            "phase": "independent unwrap(angle(S11)), degrees; no offset correction",
        },
        "magnitude_weight_k": MAGNITUDE_WEIGHT,
        "phase_weight": 1.0 - MAGNITUDE_WEIGHT,
        "gdm_mean": {name: float(result.gdm_mean) for name, result in results.items()},
    }
    (output_dir / "multidof_s11_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, result in results.items():
        print(f"{name}: GDM={result.gdm_mean:.6f} ({result.gdm_category})")
    print(f"Saved plot and recipe to {output_dir.resolve()}")
    return results


def main() -> None:
    """Parse the output directory and run the example.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multidof"))
    run_demo(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
