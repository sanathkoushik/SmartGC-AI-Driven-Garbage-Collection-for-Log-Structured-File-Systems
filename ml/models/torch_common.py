"""Shared PyTorch training/inference plumbing for the LSTM rungs."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ml.models.base import BaseIntervalModel


def _torch():
    import torch
    return torch


def set_torch_seed(seed: int) -> None:
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)


class TorchIntervalModel(BaseIntervalModel):
    """A BaseIntervalModel backed by an ``nn.Module`` that maps [B, T, F] -> [B]
    in the scaled-target space."""

    module_cls = None  # set by subclass

    def __init__(self, scaler: dict, config: Optional[dict] = None):
        super().__init__(scaler, config)
        torch = _torch()
        mlcfg = (self.config.get("ml") or {}) if isinstance(self.config, dict) else {}
        self.mlcfg = mlcfg
        self.seed = int(self.config.get("random_seed", 42)) if isinstance(self.config, dict) else 42
        set_torch_seed(self.seed)
        self.device = torch.device("cpu")
        self.n_features = len(scaler["features"])
        self.module = self._build_module()
        self.module.to(self.device)

    def _build_module(self):
        raise NotImplementedError

    # -- training ------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch = _torch()
        nn = torch.nn
        cfg = self.mlcfg
        epochs = int(cfg.get("epochs", 25))
        bs = int(cfg.get("batch_size", 64))
        lr = float(cfg.get("learning_rate", 1e-3))
        patience = int(cfg.get("early_stop_patience", 0))  # 0 => disabled
        loss_name = str(cfg.get("loss_function", "SmoothL1")).lower()
        criterion = nn.SmoothL1Loss() if "smooth" in loss_name or "l1" in loss_name else nn.MSELoss()
        opt = torch.optim.Adam(self.module.parameters(), lr=lr)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(Xt, yt)
        gen = torch.Generator().manual_seed(self.seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, generator=gen)

        val = None
        if X_val is not None and len(X_val):
            val = (torch.tensor(X_val, dtype=torch.float32),
                   torch.tensor(y_val, dtype=torch.float32))

        best_val, best_state, since_improved = float("inf"), None, 0
        for ep in range(epochs):
            self.module.train()
            tot = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                pred = self.module(xb.to(self.device)).squeeze(-1)
                loss = criterion(pred, yb.to(self.device))
                loss.backward()
                opt.step()
                tot += float(loss) * len(xb)
            tr = tot / max(1, len(ds))
            if val is not None:
                self.module.eval()
                with torch.no_grad():
                    vp = self.module(val[0].to(self.device)).squeeze(-1)
                    vl = float(criterion(vp, val[1].to(self.device)))
                if vl < best_val - 1e-5:
                    best_val, since_improved = vl, 0
                    best_state = {k: v.clone() for k, v in self.module.state_dict().items()}
                else:
                    since_improved += 1
                tag = f" val={vl:.4f}"
            else:
                tag = ""
            print(f"  [{self.name}] epoch {ep + 1:>2}/{epochs}  train={tr:.4f}{tag}")
            if val is not None and patience and since_improved >= patience:
                print(f"  [{self.name}] early stop at epoch {ep + 1} "
                      f"(no val gain for {patience} epochs; best val={best_val:.4f})")
                break
        if best_state is not None:
            self.module.load_state_dict(best_state)
        return self

    # -- inference ---------------------------------------------------
    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0,))
        torch = _torch()
        self.module.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                xb = torch.tensor(X[i:i + 512], dtype=torch.float32, device=self.device)
                outs.append(self.module(xb).squeeze(-1).cpu().numpy())
        y_scaled = np.concatenate(outs) if outs else np.zeros((0,))
        return self._from_scaled_target(y_scaled)

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.module.parameters()))

    # -- persistence ----------------------------------------------
    def save(self, path_dir: str) -> None:
        super().save(path_dir)
        torch = _torch()
        torch.save(self.module.state_dict(), os.path.join(path_dir, f"{self.name}.pt"))

    def load(self, path_dir: str) -> "TorchIntervalModel":
        torch = _torch()
        sd = torch.load(os.path.join(path_dir, f"{self.name}.pt"), map_location=self.device)
        self.module.load_state_dict(sd)
        return self
