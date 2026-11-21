"""Darcy samples on an interior vertex grid with zero Dirichlet boundary data."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import expit

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_VERSION = 1


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _coordinates(N: int) -> tuple[Float64Array, Float64Array]:
    """Return x[j, i] and y[j, i] on the N by N interior vertex grid."""
    axis = (np.arange(N, dtype=np.float64) + 1.0) / (N + 1)
    return np.meshgrid(axis, axis, indexing="xy")


def _harmonic(left: Float64Array, right: Float64Array) -> Float64Array:
    """Harmonic mean without the potentially overflowing product left*right."""
    low, high = np.minimum(left, right), np.maximum(left, right)
    return low * (2.0 / (1.0 + low / high))


def solve_darcy(a: NDArray[np.floating]) -> Float64Array:
    """Solve -div(a grad u)=1 with u=0 on the boundary.

    Permeability and the float64 solution use [j, i] indexing on an N by N
    interior grid with h=1/(N+1).
    """
    if np.iscomplexobj(a):
        raise ValueError("Permeability must be real.")
    coefficient = np.asarray(a, dtype=np.float64)
    if (coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]
            or coefficient.shape[0] < 1):
        raise ValueError("Permeability must have nonempty square shape (N, N).")
    if not np.isfinite(coefficient).all() or np.any(coefficient <= 0):
        raise ValueError("Permeability must be finite and strictly positive.")
    N = coefficient.shape[0]
    h = 1.0 / (N + 1)

    east = _harmonic(coefficient[:, :-1], coefficient[:, 1:])
    north = _harmonic(coefficient[:-1, :], coefficient[1:, :])
    diagonal = np.zeros_like(coefficient)
    diagonal[:, :-1] += east
    diagonal[:, 1:] += east
    diagonal[:-1, :] += north
    diagonal[1:, :] += north
    # Boundary vertices are one grid spacing from the outer unknowns.
    diagonal[:, 0] += coefficient[:, 0]
    diagonal[:, -1] += coefficient[:, -1]
    diagonal[0, :] += coefficient[0, :]
    diagonal[-1, :] += coefficient[-1, :]

    index = np.arange(N*N).reshape(N, N)
    west_index, east_index = index[:, :-1].ravel(), index[:, 1:].ravel()
    south_index, north_index = index[:-1, :].ravel(), index[1:, :].ravel()
    rows = np.concatenate((index.ravel(), west_index, east_index,
                           south_index, north_index))
    columns = np.concatenate((index.ravel(), east_index, west_index,
                              north_index, south_index))
    entries = np.concatenate((diagonal.ravel(), -east.ravel(), -east.ravel(),
                              -north.ravel(), -north.ravel())) / h**2
    A: sparse.csc_matrix = sparse.coo_matrix(
        (entries, (rows, columns)), shape=(N*N, N*N), dtype=np.float64,
    ).tocsc()
    if not np.isfinite(A.data).all():
        raise ValueError("Permeability scale exceeds float64 matrix range.")
    solution = np.asarray(spsolve(A, np.ones(N*N, dtype=np.float64)))
    if not np.isfinite(solution).all():
        raise FloatingPointError("Darcy solve produced non-finite values.")
    return solution.reshape(N, N)


def sample_permeability(N: int, rng: np.random.Generator) -> Float64Array:
    """Filter Gaussian noise, standardize each field, and map it into (3, 12)."""
    N = _integer(N, "N", 2)
    h = 1.0 / (N + 1)
    wavevectors = 2.0 * np.pi * np.fft.fftfreq(N, d=h)
    kx, ky = np.meshgrid(wavevectors, wavevectors, indexing="xy")
    k0 = 6.0 * np.pi
    spectral_filter = np.exp(-(kx**2 + ky**2) / (2.0 * k0**2))
    noise = rng.standard_normal((N, N))
    field = np.fft.ifft2(np.fft.fft2(noise) * spectral_filter).real
    field -= field.mean()
    deviation = float(field.std())
    if not np.isfinite(deviation) or deviation <= 0:
        raise FloatingPointError("Gaussian field has zero or non-finite variance.")
    field /= deviation
    permeability = 3.0 + 9.0 * expit(1.5 * field)
    # Retain open-interval bounds if the sigmoid saturates in finite precision.
    return np.clip(permeability, np.nextafter(3.0, 12.0),
                   np.nextafter(12.0, 3.0))


def _validate_dataset(
    inputs: Float32Array, targets: Float32Array, n_samples: int, N: int,
) -> None:
    if (inputs.shape != (n_samples, 3, N, N)
            or targets.shape != (n_samples, 1, N, N)
            or inputs.dtype != np.float32 or targets.dtype != np.float32):
        raise ValueError("Dataset shapes or storage dtypes are invalid.")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("Dataset contains non-finite values.")
    if np.any(inputs[:, 0] < 3.0) or np.any(inputs[:, 0] > 12.0):
        raise ValueError("Cached permeability is outside its specified range.")
    x, y = _coordinates(N)
    if (not np.all(inputs[:, 1] == x.astype(np.float32))
            or not np.all(inputs[:, 2] == y.astype(np.float32))):
        raise ValueError("Cached coordinate channels do not match the grid.")


def get_dataset(
    n_samples: int, N: int, seed: int,
) -> tuple[Float32Array, Float32Array]:
    """Cache float32 inputs [sample, (a,x,y), j, i] and targets [sample, 1, j, i]."""
    n_samples = _integer(n_samples, "n_samples", 1)
    N = _integer(N, "N", 2)
    seed = _integer(seed, "seed", 0)
    path = DATA_DIR / f"darcy_N{N}_n{n_samples}_s{seed}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            if (int(cached["cache_version"]) != CACHE_VERSION
                    or int(cached["N"]) != N
                    or int(cached["n_samples"]) != n_samples
                    or int(cached["seed"]) != seed):
                raise ValueError(f"Cache metadata mismatch: {path}")
            inputs, targets = cached["inputs"], cached["targets"]
        _validate_dataset(inputs, targets, n_samples, N)
        return inputs, targets

    rng = np.random.default_rng(seed)
    x, y = _coordinates(N)
    inputs = np.empty((n_samples, 3, N, N), dtype=np.float32)
    targets = np.empty((n_samples, 1, N, N), dtype=np.float32)
    inputs[:, 1], inputs[:, 2] = x, y
    for sample in range(n_samples):
        permeability = sample_permeability(N, rng)
        inputs[sample, 0] = permeability
        targets[sample, 0] = solve_darcy(permeability)
    _validate_dataset(inputs, targets, n_samples, N)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=DATA_DIR, suffix=".npz", delete=False) as file:
            temporary = Path(file.name)
            np.savez_compressed(
                file, inputs=inputs, targets=targets, cache_version=CACHE_VERSION,
                N=N, n_samples=n_samples, seed=seed,
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return inputs, targets
