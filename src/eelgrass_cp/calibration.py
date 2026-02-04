from typing import Callable, Tuple

import numpy as np
import torch
import pandas as pd


class TemperatureScaler(torch.nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = torch.nn.Parameter(torch.tensor(np.log(init_T), dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_T)


def fit_temperature(logits_np: np.ndarray, labels_np: np.ndarray, device: str, max_iter: int = 100) -> float:
    X = torch.from_numpy(logits_np).float().to(device)
    y = torch.from_numpy(labels_np).long().to(device)
    scaler = TemperatureScaler().to(device)
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = ce(scaler(X), y)
        loss.backward()
        return loss

    best_loss, best_state = float("inf"), None
    for _ in range(20):
        loss = opt.step(closure)
        if loss.item() + 1e-7 < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().clone() for k, v in scaler.state_dict().items()}
        else:
            break

    if best_state:
        scaler.load_state_dict(best_state)

    with torch.no_grad():
        T = float(torch.exp(scaler.log_T).item())
    return T


def stratified_indices(y: np.ndarray, n_total: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = max(1, n_total // max(1, len(classes)))
    chosen = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        k = min(per, len(idx_c))
        if k > 0:
            chosen.append(rng.choice(idx_c, size=k, replace=False))
    k_now = sum(len(a) for a in chosen)
    if k_now < n_total:
        pool = np.setdiff1d(np.arange(len(y)), np.concatenate(chosen) if chosen else [])
        extra_k = min(len(pool), n_total - k_now)
        if extra_k > 0:
            chosen.append(rng.choice(pool, size=extra_k, replace=False))
    return np.concatenate(chosen) if chosen else np.array([], dtype=int)


def normalize_var(v: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((v - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)


def fit_sigma_fn_from_cal(Vn: np.ndarray, scores: np.ndarray, cal_idx: np.ndarray, bins: int = 20) -> Tuple[Callable[[np.ndarray], np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Fit sigma_hat(V) = E[s | V] using quantile binning on calibration indices.
    Returns (sigma_fn, (bin_centers, bin_means)).
    """
    df = pd.DataFrame({"var": Vn[cal_idx], "score": scores[cal_idx]})
    df["var_bin"] = pd.qcut(df["var"], q=bins, duplicates="drop")
    bin_means = df.groupby("var_bin", observed=False)["score"].mean().reset_index()
    bin_centers = np.array([iv.mid for iv in bin_means["var_bin"]], dtype=float)
    sigma_means = bin_means["score"].values.astype(float)

    def sigma_fn(x: np.ndarray) -> np.ndarray:
        return np.interp(x, bin_centers, sigma_means, left=sigma_means[0], right=sigma_means[-1])

    return sigma_fn, (bin_centers, sigma_means)
