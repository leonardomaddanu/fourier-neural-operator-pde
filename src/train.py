"""Train and time a Fourier neural operator on Darcy flow at two resolutions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from data import get_dataset, solve_darcy
from model import FNO2d, relative_l2_loss

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class Normalization:
    """Training-only scalar statistics shared by both spatial resolutions."""

    permeability_mean: float
    permeability_std: float
    target_rms: float

    def inputs(self, values: Tensor) -> Tensor:
        return torch.cat(
            ((values[:, 0:1]-self.permeability_mean)/self.permeability_std,
             2.0*values[:, 1:3]-1.0), dim=1,
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware_info(device: torch.device) -> dict[str, object]:
    cpu_model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    info: dict[str, object] = {
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_model,
        "cpu_model": cpu_model,
        "torch_cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu_memory_bytes"] = properties.total_memory
        info["compute_capability"] = f"{properties.major}.{properties.minor}"
    return info


def loader(
    arrays: tuple[FloatArray, FloatArray], batch_size: int,
    device: torch.device, shuffle: bool = False, seed: int = 42,
) -> DataLoader:
    return DataLoader(
        TensorDataset(*(torch.from_numpy(array) for array in arrays)),
        batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )


def evaluate(
    model: FNO2d, batches: DataLoader, norm: Normalization, device: torch.device,
) -> dict[str, float]:
    """Aggregate the arithmetic mean of per-sample relative discrete L2 norms."""
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in batches:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            prediction = model(norm.inputs(inputs))*norm.target_rms
            relative = (torch.norm((prediction-targets).flatten(1), dim=1)
                        / torch.norm(targets.flatten(1), dim=1))
            if not torch.isfinite(relative).all():
                raise FloatingPointError("Non-finite test relative L2 error.")
            errors.append(relative.cpu().numpy())
    values = np.concatenate(errors).astype(np.float64)
    return {
        "mean_relative_l2": float(values.mean()),
        "std_relative_l2": float(values.std()),
        "max_relative_l2": float(values.max()),
    }


def timing_stats(seconds: list[float]) -> dict[str, float]:
    values = np.asarray(seconds, dtype=np.float64)
    return {"mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)),
            "std_seconds": float(values.std())}


def benchmark(
    model: FNO2d, inputs: FloatArray, norm: Normalization, device: torch.device,
) -> dict[str, object]:
    """Time 5 warm-ups and 50 calls to each solver on the same test coefficient."""
    resident = torch.from_numpy(inputs[0:1]).to(device)
    coefficient = inputs[0, 0].astype(np.float64)
    model.eval()
    fno_times: list[float] = []
    with torch.no_grad():
        for _ in range(5):
            prediction = model(norm.inputs(resident))*norm.target_rms
        synchronize(device)
        for _ in range(50):
            synchronize(device)
            started = time.perf_counter()
            prediction = model(norm.inputs(resident))*norm.target_rms
            synchronize(device)
            fno_times.append(time.perf_counter()-started)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite benchmark prediction.")
    for _ in range(5):
        solve_darcy(coefficient)
    sparse_times: list[float] = []
    for _ in range(50):
        started = time.perf_counter()
        reference = solve_darcy(coefficient)
        sparse_times.append(time.perf_counter()-started)
    if not np.isfinite(reference).all():
        raise FloatingPointError("Non-finite timed Darcy solution.")
    fno, sparse = timing_stats(fno_times), timing_stats(sparse_times)
    return {
        "N": int(inputs.shape[-1]), "batch_size": 1,
        "warmup_calls_per_solver": 5, "timed_calls_per_solver": 50,
        "sample": "test sample 0, repeated coefficient",
        "fno": fno, "spsolve": sparse,
        "speedup": sparse["mean_seconds"]/fno["mean_seconds"],
        "scope": "FNO: device-resident raw input, normalization, forward, output scaling; "
                 "excludes host/device transfers. spsolve: CPU coefficient, validation, "
                 "sparse assembly, factorization and solve; excludes dataset loading.",
    }


def plot_results(
    model: FNO2d, test64: tuple[FloatArray, FloatArray],
    test128: tuple[FloatArray, FloatArray], norm: Normalization,
    history: list[dict[str, float | int]], device: torch.device,
) -> list[Path]:
    os.makedirs(DOCS_DIR, exist_ok=True)
    curve = DOCS_DIR/"training_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = [row["epoch"] for row in history]
    ax.semilogy(epochs, [100*row["train_relative_l2"] for row in history],
                label="Train, 64 x 64")
    ax.semilogy(epochs, [100*row["test64_relative_l2"] for row in history],
                label="Test, 64 x 64")
    ax.set(xlabel="Epoch", ylabel="Mean relative L2 error (%)",
           title="Darcy FNO2d training")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve, dpi=300)
    plt.close(fig)

    prediction_path = DOCS_DIR/"prediction_superres.png"
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    model.eval()
    with torch.no_grad():
        for row, arrays in enumerate((test64, test128)):
            inputs, targets = arrays
            N = inputs.shape[-1]
            values = torch.from_numpy(inputs[0:1]).to(device)
            prediction = (model(norm.inputs(values))*norm.target_rms)[0, 0].cpu().numpy()
            reference = targets[0, 0]
            fields = (inputs[0, 0], reference, prediction, np.abs(reference-prediction))
            titles = ("Permeability a(x,y)", "Reference (spsolve)",
                      "FNO2d Prediction", "Absolute Error |u - u_hat|")
            low = min(float(reference.min()), float(prediction.min()))
            high = max(float(reference.max()), float(prediction.max()))
            h = 1.0/(N+1)
            extent = (h/2, 1-h/2, h/2, 1-h/2)
            for column, (field, title) in enumerate(zip(fields, titles)):
                limits = {"vmin": low, "vmax": high} if column in (1, 2) else {}
                image = axes[row, column].imshow(
                    field, origin="lower", extent=extent,
                    cmap="magma" if column == 3 else "viridis", **limits,
                )
                axes[row, column].set(
                    title=title, xlabel="x",
                    ylabel=f"{N} x {N}"+(" zero-shot\n y" if row else "\n y"),
                    xlim=(0, 1), ylim=(0, 1),
                )
                fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(prediction_path, dpi=300)
    plt.close(fig)
    return [curve, prediction_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if min(args.epochs, args.n_train, args.n_test, args.batch_size) < 1:
        parser.error("Epochs, dataset counts, and batch size must be positive.")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must lie in [0, 2**32).")
    started = time.perf_counter()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = hardware_info(device)
    print("Hardware:", json.dumps(hardware, indent=2), flush=True)
    print("Configuration:", json.dumps(vars(args), sort_keys=True), flush=True)
    print("Dataset seeds: train64=42, test64=100, test128=200. "
          "Test sets are independent realizations, not paired grids.", flush=True)

    datasets: dict[str, tuple[FloatArray, FloatArray]] = {}
    for name, count, N, seed in (
        ("train64", args.n_train, 64, 42),
        ("test64", args.n_test, 64, 100),
        ("test128", args.n_test, 128, 200),
    ):
        print(f"Loading/generating {name}: n={count}, N={N}, seed={seed}", flush=True)
        datasets[name] = get_dataset(count, N, seed)
        print(f"  inputs={datasets[name][0].shape}, targets={datasets[name][1].shape}",
              flush=True)
    train = datasets["train64"]
    norm = Normalization(
        float(train[0][:, 0].mean(dtype=np.float64)),
        float(train[0][:, 0].std(dtype=np.float64)),
        float(np.sqrt(np.mean(np.square(train[1], dtype=np.float64)))),
    )
    if min(norm.permeability_std, norm.target_rms) <= 0:
        raise ValueError("Training normalization is degenerate.")
    print("Training-only normalization:", json.dumps(vars(norm)), flush=True)
    train_loader = loader(train, args.batch_size, device, shuffle=True, seed=args.seed)
    train_eval = loader(train, args.batch_size, device)
    test64 = loader(datasets["test64"], args.batch_size, device)
    test128 = loader(datasets["test128"], args.batch_size, device)
    model = FNO2d(modes=12, width=32, n_layers=4).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    real_dof = sum(p.numel()*(2 if p.is_complex() else 1)
                   for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {parameters}; real parameter components: {real_dof}",
          flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, float | int]] = []
    synchronize(device)
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        epoch_started = time.perf_counter()
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")/norm.target_rms
            optimizer.zero_grad(set_to_none=True)
            loss = relative_l2_loss(model(norm.inputs(inputs)), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_error = evaluate(model, train_eval, norm, device)["mean_relative_l2"]
        test_error = evaluate(model, test64, norm, device)["mean_relative_l2"]
        synchronize(device)
        history.append({
            "epoch": epoch, "train_relative_l2": train_error,
            "test64_relative_l2": test_error, "lr": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter()-epoch_started,
        })
        print(f"Epoch {epoch:3d}/{args.epochs} | train={100*train_error:.5f}% "
              f"test64={100*test_error:.5f}% | {history[-1]['seconds']:.2f}s",
              flush=True)
    synchronize(device)
    training_seconds = time.perf_counter()-training_started
    final64 = evaluate(model, test64, norm, device)
    final128 = evaluate(model, test128, norm, device)
    print(f"Final 64x64 mean relative L2: {final64['mean_relative_l2']:.10f} "
          f"({100*final64['mean_relative_l2']:.6f}%)", flush=True)
    print(f"Final 128x128 zero-shot mean relative L2: {final128['mean_relative_l2']:.10f} "
          f"({100*final128['mean_relative_l2']:.6f}%)", flush=True)
    timings = [benchmark(model, datasets[name][0], norm, device)
               for name in ("test64", "test128")]
    print("\n| Grid | FNO2d latency (ms) | spsolve latency (ms) | Speedup |", flush=True)
    print("| --- | ---: | ---: | ---: |", flush=True)
    for row in timings:
        print(f"| {row['N']} x {row['N']} | {1000*row['fno']['mean_seconds']:.6f} "
              f"| {1000*row['spsolve']['mean_seconds']:.6f} "
              f"| {row['speedup']:.4f}x |", flush=True)
    print("Timing scope:", timings[0]["scope"], flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"e{args.epochs}_n{args.n_train}_t{args.n_test}_s{args.seed}"
    checkpoint = DATA_DIR/f"fno_{tag}.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args),
                "normalization": vars(norm), "history": history}, checkpoint)
    check_accuracy = args.n_train >= 800
    passed = final64["mean_relative_l2"] < 0.05 and final128["mean_relative_l2"] < 0.05
    report = {
        "config": vars(args), "hardware": hardware, "parameters_numel": parameters,
        "real_scalar_degrees_of_freedom": real_dof,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
        "dataset_seeds": {"train64": 42, "test64": 100, "test128": 200},
        "test_relationship": "Independent realizations at each resolution; disjoint seeds.",
        "normalization": vars(norm),
        "normalization_scope": "Training only; (a-mean)/std, coordinates 2*x-1, "
                               "targets divided by global training RMS.",
        "model": {"modes": 12, "width": 32, "n_layers": 4},
        "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 1e-4,
                      "scheduler": "CosineAnnealingLR", "T_max": args.epochs},
        "evaluation": {"64": final64, "128": final128},
        "error_definition": "Arithmetic mean over samples of norm(pred-target)/norm(target).",
        "timings": timings, "history": history, "training_seconds": training_seconds,
        "runtime_seconds_before_plots": time.perf_counter()-started,
        "checkpoint": checkpoint.relative_to(REPO_ROOT).as_posix(),
        "source_sha256": {name: hashlib.sha256((REPO_ROOT/name).read_bytes()).hexdigest()
                          for name in ("src/data.py", "src/model.py", "src/train.py")},
        "accuracy_gate_applied": check_accuracy, "both_errors_under_5pct": passed,
    }
    report_path = DATA_DIR/f"experiment_{tag}.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Local checkpoint: {checkpoint}\nLocal measurements: {report_path}", flush=True)
    if check_accuracy:
        assert final64["mean_relative_l2"] < 0.05, "64x64 mean error must be below 5%."
        assert final128["mean_relative_l2"] < 0.05, "128x128 zero-shot mean error must be below 5%."
        for path in plot_results(model, datasets["test64"], datasets["test128"],
                                 norm, history, device):
            print(f"300 DPI figure: {path}", flush=True)
        print("Accuracy checks passed: both mean test errors are below 5%.", flush=True)
    else:
        print("Smoke test complete; accuracy checks and saved plots apply only "
              "to runs with n_train >= 800.", flush=True)


if __name__ == "__main__":
    main()
