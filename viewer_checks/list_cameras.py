#!/usr/bin/env python3
"""
Load MATLAB .mat files — works with both classic (v7 and earlier)
and v7.3 (HDF5-based) formats.

Usage:
    python load_mat.py data.mat              # list all variables
    python load_mat.py data.mat varname      # print one variable

Or import it:
    from load_mat import load_mat
    data = load_mat("data.mat")   # dict of {name: numpy array}

Requires: scipy, numpy. h5py only needed for v7.3 files.
"""

import sys
from pathlib import Path

import numpy as np


def load_mat(path):
    """Return the contents of a .mat file as a dict."""
    path = Path(path)
    try:
        from scipy.io import loadmat
        data = loadmat(path, squeeze_me=True, struct_as_record=False)

        def to_dict(v):  # turn MATLAB structs into plain dicts
            if hasattr(v, "_fieldnames"):
                return {f: to_dict(getattr(v, f)) for f in v._fieldnames}
            return v

        return {k: to_dict(v) for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        # scipy can't read v7.3 (HDF5) files -> fall back to h5py
        import h5py

        def unpack(obj):
            if isinstance(obj, h5py.Group):
                return {k: unpack(v) for k, v in obj.items()}
            arr = obj[()]
            # MATLAB char arrays come through as uint16 codepoints
            if obj.attrs.get("MATLAB_class") == b"char":
                return "".join(map(chr, np.atleast_1d(arr).ravel()))
            # MATLAB stores arrays column-major, so transpose back
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                arr = arr.T
            return arr

        with h5py.File(path, "r") as f:
            return {k: unpack(f[k]) for k in f if not k.startswith("#")}


def describe(data, indent=0):
    pad = "  " * indent
    for name, val in data.items():
        if isinstance(val, dict):
            print(f"{pad}{name}/ (struct)")
            describe(val, indent + 1)
        elif isinstance(val, np.ndarray):
            print(f"{pad}{name}: array {val.shape} {val.dtype}")
        else:
            print(f"{pad}{name}: {type(val).__name__} = {val!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data = load_mat(sys.argv[1])

    if len(sys.argv) > 2:
        print(data[sys.argv[2]])
    else:
        print(f"Variables in {sys.argv[1]}:")
        describe(data)