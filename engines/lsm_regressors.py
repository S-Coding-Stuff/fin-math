"""Regression helpers for LSM continuation value estimation."""

import numpy as np


def estimate_continuation_nn(
    states: np.ndarray,
    targets: np.ndarray,
    *,
    hidden_sizes: tuple[int, ...] = (32, 32),
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    scale_inputs: bool = True,
    min_samples: int = 8,
    seed: int | None = None,
) -> np.ndarray:
    """Estimate continuation values using a small PyTorch MLP.

    Returns predictions aligned to the input states order.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Neural-network LSM regression requires PyTorch. Install with `pip install torch`."
        ) from exc

    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")

    if seed is not None:
        torch.manual_seed(seed)

    features = np.asarray(states, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    elif features.ndim != 2:
        raise ValueError("states must be a 1D or 2D array.")
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)

    if features.shape[0] != targets.shape[0]:
        raise ValueError("states and targets must have the same number of samples.")

    if features.shape[0] < min_samples:
        return np.full_like(targets.squeeze(-1), float(np.mean(targets)))

    x = features
    if scale_inputs:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(targets)

    layers: list[nn.Module] = []
    in_dim = x.shape[1]
    for size in hidden_sizes:
        layers.append(nn.Linear(in_dim, size))
        layers.append(nn.ReLU())
        in_dim = size
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_samples = x_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_t[idx]
            yb = y_t[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(x_t).squeeze(-1).cpu().numpy()
    return preds


__all__ = ["estimate_continuation_nn"]
