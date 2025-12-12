"""
Data loading utilities for LHC Olympics R&D dataset.

This module handles downloading and loading the background and signal datasets
from Zenodo, along with feature extraction and preprocessing.
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


def download_file(url: str, output_path: str, force: bool = False) -> None:
    """
    Download a file from URL to local path.

    Args:
        url: URL to download from
        output_path: Local path to save file
        force: If True, re-download even if file exists
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"File already exists: {output_path}")
        return

    print(f"Downloading {url} to {output_path}...")
    urllib.request.urlretrieve(url, str(output_path))
    print(f"Download complete: {output_path}")


def load_background_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load background dataset from HDF5 file and compute high-level features.

    Args:
        file_path: Path to background HDF5 file

    Returns:
        Dictionary with feature arrays
    """
    print(f"Loading background data from {file_path}...")
    # Load pandas DataFrame from HDF5
    df = pd.read_hdf(file_path, key="df")

    # Extract 4-vectors for both jets
    pxj1, pyj1, pzj1, mj1 = (
        df["pxj1"].values,
        df["pyj1"].values,
        df["pzj1"].values,
        df["mj1"].values,
    )
    pxj2, pyj2, pzj2, mj2 = (
        df["pxj2"].values,
        df["pyj2"].values,
        df["pzj2"].values,
        df["mj2"].values,
    )

    # Compute energies
    ej1 = np.sqrt(pxj1**2 + pyj1**2 + pzj1**2 + mj1**2)
    ej2 = np.sqrt(pxj2**2 + pyj2**2 + pzj2**2 + mj2**2)

    # Compute dijet 4-vector
    px_jj = pxj1 + pxj2
    py_jj = pyj1 + pyj2
    pz_jj = pzj1 + pzj2
    e_jj = ej1 + ej2

    # Compute dijet invariant mass in TeV
    mJJ = np.sqrt(e_jj**2 - px_jj**2 - py_jj**2 - pz_jj**2) / 1000.0  # Convert GeV to TeV

    # Jet masses in GeV (keep original units for consistency with paper)
    mJ1 = mj1
    mJ2 = mj2

    # N-subjettiness ratios
    tau21_J1 = df["tau2j1"].values / (df["tau1j1"].values + 1e-10)  # Avoid division by zero
    tau21_J2 = df["tau2j2"].values / (df["tau1j2"].values + 1e-10)

    data = {
        "mJJ": mJJ,  # In TeV
        "mJ1": mJ1,  # In GeV
        "mJ2": mJ2,  # In GeV
        "tau21_J1": tau21_J1,
        "tau21_J2": tau21_J2,
    }

    # Compute derived feature: jet mass difference
    data["delta_mJ"] = mJ2 - mJ1

    print(f"Loaded {len(data['mJJ'])} background events")
    print(f"  mJJ range: [{mJJ.min():.2f}, {mJJ.max():.2f}] TeV")
    return data


def load_signal_data(file_path: str, mX: float = 500.0, mY: float = 100.0) -> Dict[str, np.ndarray]:
    """
    Load signal dataset from parametric HDF5 file and compute high-level features.

    Args:
        file_path: Path to signal HDF5 file
        mX: X particle mass in GeV (default 500)
        mY: Y particle mass in GeV (default 100)

    Returns:
        Dictionary with feature arrays for selected mass point
    """
    print(f"Loading signal data from {file_path} (mX={mX} GeV, mY={mY} GeV)...")

    # Load pandas DataFrame from HDF5 (signal file uses 'output' key, not 'df')
    df = pd.read_hdf(file_path, key="output")

    # Check if parametric (has mx and my columns - note lowercase in signal file)
    if "mx" in df.columns and "my" in df.columns:
        available_mX = np.unique(df["mx"].values)
        available_mY = np.unique(df["my"].values)
        print(f"Available mX values: {available_mX}")
        print(f"Available mY values: {available_mY}")

        # Select events matching the requested mass point
        mask = (np.abs(df["mx"] - mX) < 1.0) & (np.abs(df["my"] - mY) < 1.0)
        df = df[mask]
        print(f"Selected {len(df)} events with mX={mX}, mY={mY}")

    # Extract 4-vectors for both jets
    pxj1, pyj1, pzj1, mj1 = (
        df["pxj1"].values,
        df["pyj1"].values,
        df["pzj1"].values,
        df["mj1"].values,
    )
    pxj2, pyj2, pzj2, mj2 = (
        df["pxj2"].values,
        df["pyj2"].values,
        df["pzj2"].values,
        df["mj2"].values,
    )

    # Compute energies
    ej1 = np.sqrt(pxj1**2 + pyj1**2 + pzj1**2 + mj1**2)
    ej2 = np.sqrt(pxj2**2 + pyj2**2 + pzj2**2 + mj2**2)

    # Compute dijet 4-vector
    px_jj = pxj1 + pxj2
    py_jj = pyj1 + pyj2
    pz_jj = pzj1 + pzj2
    e_jj = ej1 + ej2

    # Compute dijet invariant mass in TeV
    mJJ = np.sqrt(e_jj**2 - px_jj**2 - py_jj**2 - pz_jj**2) / 1000.0  # Convert GeV to TeV

    # Jet masses in GeV
    mJ1 = mj1
    mJ2 = mj2

    # N-subjettiness ratios
    tau21_J1 = df["tau2j1"].values / (df["tau1j1"].values + 1e-10)
    tau21_J2 = df["tau2j2"].values / (df["tau1j2"].values + 1e-10)

    data = {
        "mJJ": mJJ,  # In TeV
        "mJ1": mJ1,  # In GeV
        "mJ2": mJ2,  # In GeV
        "tau21_J1": tau21_J1,
        "tau21_J2": tau21_J2,
    }

    # Store mass point info if available (using lowercase column names)
    if "mx" in df.columns:
        data["mX"] = df["mx"].values
        data["mY"] = df["my"].values

    # Compute derived feature: jet mass difference
    data["delta_mJ"] = mJ2 - mJ1

    print(f"Loaded {len(data['mJJ'])} signal events")
    print(f"  mJJ range: [{mJJ.min():.2f}, {mJJ.max():.2f}] TeV")
    return data


def get_features(data: Dict[str, np.ndarray], include_mass: bool = False) -> np.ndarray:
    """
    Extract feature matrix from data dictionary.

    The CATHODE paper uses 5 features:
    - mJ1: Leading jet mass (lighter jet)
    - delta_mJ: Jet mass difference (mJ2 - mJ1)
    - tau21_J1: N-subjettiness ratio for jet 1
    - tau21_J2: N-subjettiness ratio for jet 2
    - mJJ: Dijet invariant mass (only if include_mass=True)

    Args:
        data: Dictionary with feature arrays
        include_mass: If True, include mJJ as a feature

    Returns:
        Feature matrix of shape (n_events, n_features)
    """
    features = [
        data["mJ1"],
        data["delta_mJ"],
        data["tau21_J1"],
        data["tau21_J2"],
    ]

    if include_mass:
        features.append(data["mJJ"])

    return np.column_stack(features)


def apply_region_cut(
    data: Dict[str, np.ndarray],
    mJJ_low: float = 1.5,
    mJJ_high: float = 5.5,
    sr_low: Optional[float] = None,
    sr_high: Optional[float] = None,
    signal_region: bool = True,
) -> np.ndarray:
    """
    Create boolean mask for signal region (SR) or sideband (SB).

    Args:
        data: Dictionary with feature arrays (must contain 'mJJ' in TeV)
        mJJ_low: Lower bound for full mJJ range in TeV
        mJJ_high: Upper bound for full mJJ range in TeV
        sr_low: Lower bound for SR in TeV (default 3.3)
        sr_high: Upper bound for SR in TeV (default 3.7)
        signal_region: If True, select SR; if False, select SB

    Returns:
        Boolean mask for selected region
    """
    if sr_low is None:
        sr_low = 3.3
    if sr_high is None:
        sr_high = 3.7

    mJJ = data["mJJ"]

    # Full range cut
    in_range = (mJJ >= mJJ_low) & (mJJ <= mJJ_high)

    # Signal region cut
    in_sr = (mJJ >= sr_low) & (mJJ <= sr_high)

    if signal_region:
        return in_range & in_sr
    else:
        # Sideband: in full range but NOT in SR
        return in_range & ~in_sr
