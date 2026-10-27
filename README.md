# kt-fsv

Feature Selective Validation (FSV) for paired, real-valued one-dimensional
responses, with multi-DOF component combination and plotting tools.

Implements selected methods from IEEE Std 1597.1-2022; full IEEE conformance
is not claimed.

Requires Python 3.13.5+, NumPy 2.5.2+ and Matplotlib 3.10.1+.

Install from this directory:

```bash
python -m pip install .
```

Transmission-line example using scikit-rf: equal-length 50 Ω and 45 Ω lines,
comparing S11 magnitude in dB and unwrapped phase in degrees:

```bash
python -m pip install ".[examples]"
python examples/multidof_transmission_line.py --output-dir outputs/multidof
```

MIT License. See [LICENSE](LICENSE).
