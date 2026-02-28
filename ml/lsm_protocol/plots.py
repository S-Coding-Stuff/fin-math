"""Plot generation utilities for the LSM-ML evaluation protocol."""

import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _configure_matplotlib_env() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config = Path(tempfile.gettempdir()) / "mplconfig"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config)
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = Path(tempfile.gettempdir()) / "xdg-cache"
        xdg_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)


_configure_matplotlib_env()
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None
    mcolors = None


def save_plot_figure(fig: Any, path: Path) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def create_learning_diagnostic_plots(*, training_trace: pd.DataFrame, output_dir: Path) -> list[Path]:
    if plt is None or training_trace.empty:
        return []

    trace = training_trace.dropna(subset=["fit_mse"]).copy()
    if trace.empty:
        return []

    trace["backward_step"] = trace["steps"] - trace["t"]
    grouped = trace.groupby(["model", "backward_step"], as_index=False).agg(
        fit_mse_mean=("fit_mse", "mean"),
        fit_mse_q10=("fit_mse", lambda s: float(np.quantile(s, 0.10))),
        fit_mse_q90=("fit_mse", lambda s: float(np.quantile(s, 0.90))),
        selected_mean=("selected", "mean"),
    )

    outputs: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    for model in sorted(grouped["model"].unique()):
        part = grouped[grouped["model"] == model].sort_values("backward_step")
        x = part["backward_step"].to_numpy()
        y = part["fit_mse_mean"].to_numpy()
        low = part["fit_mse_q10"].to_numpy()
        high = part["fit_mse_q90"].to_numpy()
        ax1.plot(x, y, marker="o", linewidth=1.6, markersize=3, label=model.upper())
        ax1.fill_between(x, low, high, alpha=0.15)
    ax1.set_title("Learning Curve: Per-Step Fit MSE")
    ax1.set_xlabel("Backward training step (1 = maturity-1)")
    ax1.set_ylabel("Fit MSE")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.25)
    ax1.legend()
    learning_path = output_dir / "american_lsm_ml_learning_curve.png"
    save_plot_figure(fig1, learning_path)
    outputs.append(learning_path)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for model in sorted(grouped["model"].unique()):
        part = grouped[grouped["model"] == model].sort_values("backward_step")
        ax2.plot(
            part["backward_step"].to_numpy(),
            part["selected_mean"].to_numpy(),
            marker="o",
            linewidth=1.6,
            markersize=3,
            label=model.upper(),
        )
    ax2.set_title("In-the-Money Sample Count by Backward Step")
    ax2.set_xlabel("Backward training step (1 = maturity-1)")
    ax2.set_ylabel("Average selected paths")
    ax2.grid(alpha=0.25)
    ax2.legend()
    selected_path = output_dir / "american_lsm_ml_selected_paths_curve.png"
    save_plot_figure(fig2, selected_path)
    outputs.append(selected_path)

    return outputs


def create_baseline_bucket_rmse_plot(*, baseline_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    if plt is None or baseline_summary.empty:
        return []

    data = baseline_summary[baseline_summary["experiment"] == "baseline_rmse"].copy()
    if data.empty:
        return []

    bucket_order = ["ITM", "ATM", "OTM", "Total"]
    model_order = sorted(data["model"].unique())
    x = np.arange(len(bucket_order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(model_order):
        part = data[data["model"] == model].set_index("bucket").reindex(bucket_order)
        y = part["rmse_mean"].to_numpy(dtype=float)
        low = part["rmse_ci95_low"].to_numpy(dtype=float)
        high = part["rmse_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - low, high - y])
        ax.bar(
            x + (i - (len(model_order) - 1) / 2.0) * width,
            y,
            width=width,
            yerr=yerr,
            capsize=3,
            label=model.upper(),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_order)
    ax.set_title("Baseline RMSE by Moneyness Bucket (95% CI)")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()

    path = output_dir / "american_lsm_ml_baseline_bucket_rmse.png"
    save_plot_figure(fig, path)
    return [path]


def create_total_rmse_sensitivity_plot(
    *,
    summary: pd.DataFrame,
    experiment: str,
    variable_name: str,
    title: str,
    output_name: str,
    output_dir: Path,
) -> list[Path]:
    if plt is None or summary.empty:
        return []

    data = summary[(summary["experiment"] == experiment) & (summary["bucket"] == "Total")].copy()
    if data.empty:
        return []

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in sorted(data["model"].unique()):
        part = data[data["model"] == model].sort_values(variable_name)
        x = part[variable_name].to_numpy(dtype=float)
        y = part["rmse_mean"].to_numpy(dtype=float)
        low = part["rmse_ci95_low"].to_numpy(dtype=float)
        high = part["rmse_ci95_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.6, label=model.upper())
        ax.fill_between(x, low, high, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Total RMSE")
    ax.grid(alpha=0.25)
    ax.legend()

    path = output_dir / output_name
    save_plot_figure(fig, path)
    return [path]


def create_price_error_diagnostic_plots(*, all_records: pd.DataFrame, output_dir: Path) -> list[Path]:
    if plt is None or all_records.empty:
        return []

    data = all_records[all_records["experiment"] == "baseline_rmse"].copy()
    if data.empty:
        return []

    outputs: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    lo = float(data["error"].min())
    hi = float(data["error"].max())
    if math.isclose(lo, hi):
        lo -= 1e-6
        hi += 1e-6
    bins = np.linspace(lo, hi, 30)
    for model in sorted(data["model"].unique()):
        part = data[data["model"] == model]
        ax1.hist(
            part["error"].to_numpy(dtype=float),
            bins=bins,
            density=True,
            alpha=0.35,
            label=model.upper(),
        )
    ax1.set_title("Pricing Error Distribution (Price - Benchmark)")
    ax1.set_xlabel("Error")
    ax1.set_ylabel("Density")
    ax1.grid(alpha=0.25)
    ax1.legend()
    hist_path = output_dir / "american_lsm_ml_error_distribution.png"
    save_plot_figure(fig1, hist_path)
    outputs.append(hist_path)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for model in sorted(data["model"].unique()):
        part = data[data["model"] == model]
        ax2.scatter(
            part["benchmark"].to_numpy(dtype=float),
            part["price"].to_numpy(dtype=float),
            s=14,
            alpha=0.45,
            label=model.upper(),
        )
    min_xy = float(min(data["benchmark"].min(), data["price"].min()))
    max_xy = float(max(data["benchmark"].max(), data["price"].max()))
    ax2.plot([min_xy, max_xy], [min_xy, max_xy], linestyle="--", linewidth=1.1, color="black")
    ax2.set_title("Predicted Price vs Benchmark")
    ax2.set_xlabel("Benchmark")
    ax2.set_ylabel("Predicted")
    ax2.grid(alpha=0.25)
    ax2.legend()
    scatter_path = output_dir / "american_lsm_ml_price_vs_benchmark.png"
    save_plot_figure(fig2, scatter_path)
    outputs.append(scatter_path)

    return outputs


def create_academic_error_surface_3d_plot(*, all_records: pd.DataFrame, output_dir: Path) -> list[Path]:
    if plt is None or mcolors is None or all_records.empty:
        return []

    data = all_records[all_records["experiment"] == "baseline_rmse"].copy()
    if data.empty:
        return []

    surf_data = data.groupby(["model", "moneyness", "T"], as_index=False)["error"].mean()
    if surf_data.empty:
        return []

    models = sorted(surf_data["model"].unique())
    ncols = len(models)
    fig = plt.figure(figsize=(5.3 * ncols, 5.4))

    vmin = float(surf_data["error"].min())
    vmax = float(surf_data["error"].max())
    abs_max = max(abs(vmin), abs(vmax))
    if abs_max <= 0:
        abs_max = 1e-6
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    mappable = None
    for i, model in enumerate(models, start=1):
        ax = fig.add_subplot(1, ncols, i, projection="3d")
        part = surf_data[surf_data["model"] == model]
        x = part["moneyness"].to_numpy(dtype=float)
        y = part["T"].to_numpy(dtype=float)
        z = part["error"].to_numpy(dtype=float)
        trisurf = ax.plot_trisurf(
            x,
            y,
            z,
            cmap="coolwarm",
            norm=norm,
            linewidth=0.2,
            antialiased=True,
            alpha=0.95,
        )
        mappable = trisurf
        ax.set_title(model.upper())
        ax.set_xlabel("Moneyness (S0/K)")
        ax.set_ylabel("Maturity (T)")
        ax.set_zlabel("Error")
        ax.view_init(elev=24, azim=-132)

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.72, pad=0.08)
        cbar.set_label("Pricing Error (Predicted - Benchmark)")

    fig.suptitle("3D Pricing Error Surface by Model", y=0.98)
    path = output_dir / "american_lsm_ml_error_surface_3d.png"
    save_plot_figure(fig, path)
    return [path]


def generate_protocol_plots(
    *,
    all_records: pd.DataFrame,
    all_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    training_trace: pd.DataFrame,
    output_dir: Path,
    enabled: bool,
) -> list[Path]:
    if not enabled or plt is None:
        return []

    paths: list[Path] = []
    paths.extend(create_learning_diagnostic_plots(training_trace=training_trace, output_dir=output_dir))
    paths.extend(create_baseline_bucket_rmse_plot(baseline_summary=baseline_summary, output_dir=output_dir))
    paths.extend(
        create_total_rmse_sensitivity_plot(
            summary=all_summary,
            experiment="paths_sensitivity",
            variable_name="num_paths",
            title="Total RMSE vs Number of Training Paths",
            output_name="american_lsm_ml_paths_sensitivity_rmse.png",
            output_dir=output_dir,
        )
    )
    paths.extend(
        create_total_rmse_sensitivity_plot(
            summary=all_summary,
            experiment="steps_sensitivity",
            variable_name="steps",
            title="Total RMSE vs Number of Time Steps",
            output_name="american_lsm_ml_steps_sensitivity_rmse.png",
            output_dir=output_dir,
        )
    )
    paths.extend(create_price_error_diagnostic_plots(all_records=all_records, output_dir=output_dir))
    paths.extend(create_academic_error_surface_3d_plot(all_records=all_records, output_dir=output_dir))
    return paths


def generate_academic_error_surface_from_csv(
    *,
    raw_csv_path: Path | str = Path("results/summary/american_lsm_ml_protocol_raw.csv"),
    output_dir: Path | str = Path("results/summary"),
) -> Path | None:
    """Generate the paper-grade 3D pricing error surface from saved raw results."""
    if plt is None or mcolors is None:
        return None

    raw_path = Path(raw_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(raw_path)
    paths = create_academic_error_surface_3d_plot(all_records=records, output_dir=out_dir)
    return paths[0] if paths else None


# Backward-compatible aliases for existing imports.
save_figure = save_plot_figure
plot_learning_diagnostics = create_learning_diagnostic_plots
plot_baseline_bucket_rmse = create_baseline_bucket_rmse_plot
plot_total_rmse_sensitivity = create_total_rmse_sensitivity_plot
plot_price_error_diagnostics = create_price_error_diagnostic_plots
plot_academic_error_surface_3d = create_academic_error_surface_3d_plot
generate_plots = generate_protocol_plots
