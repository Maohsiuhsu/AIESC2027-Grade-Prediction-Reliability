"""TabM and FT-Transformer through their OFFICIAL implementations (Experiment A).

* TabM: `tabm` package (Gorishniy et al., Yandex Research) - tabm.TabM.make(...)
* FT-Transformer: `rtdl_revisiting_models` package (Gorishniy et al., 2021) - FTTransformer with
  get_default_kwargs() and make_default_optimizer()

All preprocessing is fitted on the outer training fold only: categorical level codes (levels with
fewer than `min_count` training rows and unseen levels map to UNK = 0), median imputation and a
QuantileTransformer for numerics, target standardisation. Early stopping uses a 10% grouped inner
validation split of the training fold (same grouping unit as the outer split). A wall-clock cap
stops training and flags `budget_exceeded`.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import QuantileTransformer

CAT = ["course_key", "required_elective", "course_category", "student_college", "student_dept", "division", "program"]
NUM = ["credits", "section_size", "section_n_student_colleges"]


class OfficialNN(BaseEstimator, RegressorMixin):
    def __init__(self, kind="tabm", group_col="student_id", seed=42, max_epochs=30, patience=5, batch=512,
                 min_count=5, time_cap_s=1200, threads=32, k=32):
        self.kind, self.group_col, self.seed, self.max_epochs, self.patience = kind, group_col, seed, max_epochs, patience
        self.batch, self.min_count, self.time_cap_s, self.threads, self.k = batch, min_count, time_cap_s, threads, k

    # ---- preprocessing (fitted on training rows only) ----------------------------------------
    def _fit_prep(self, X):
        self.maps_ = {}
        for c in CAT:
            vc = X[c].astype("string").fillna("__MISSING__").value_counts()
            keep = sorted(vc[vc >= self.min_count].index)
            self.maps_[c] = {v: i + 1 for i, v in enumerate(keep)}
        num = X[NUM].apply(pd.to_numeric, errors="coerce")
        self.median_ = num.median()
        self.qt_ = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(X)),
                                       subsample=10**9, random_state=self.seed).fit(num.fillna(self.median_).to_numpy())

    def _prep(self, X):
        import torch
        xc = np.stack([X[c].astype("string").fillna("__MISSING__").map(self.maps_[c]).fillna(0).astype(int).to_numpy()
                       for c in CAT], 1)
        xn = self.qt_.transform(X[NUM].apply(pd.to_numeric, errors="coerce").fillna(self.median_).to_numpy())
        return torch.as_tensor(xn, dtype=torch.float32), torch.as_tensor(xc, dtype=torch.long)

    def _model(self):
        card = [len(self.maps_[c]) + 1 for c in CAT]
        if self.kind == "tabm":
            import tabm
            m = tabm.TabM.make(n_num_features=len(NUM), cat_cardinalities=card, d_out=1, k=self.k)
            import torch
            opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=3e-4)
        else:
            import rtdl_revisiting_models as rtdl
            m = rtdl.FTTransformer(n_cont_features=len(NUM), cat_cardinalities=card, d_out=1,
                                   **rtdl.FTTransformer.get_default_kwargs(n_blocks=2))
            opt = m.make_default_optimizer()
        return m, opt

    def _forward(self, m, xn, xc):
        out = m(xn, xc)
        return out.squeeze(-1)                     # tabm: (B, k); ft: (B,)

    def fit(self, X: pd.DataFrame, y):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        g = X[self.group_col].astype(str).to_numpy()
        ug = np.unique(g)
        val_groups = set(rng.choice(ug, size=max(1, len(ug) // 10), replace=False))
        va = np.array([v in val_groups for v in g])
        self._fit_prep(X[~va])
        self.y_mean_, self.y_std_ = y[~va].mean(), y[~va].std() + 1e-9
        xn, xc = self._prep(X)
        z = torch.as_tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)
        tr_idx, va_idx = np.flatnonzero(~va), np.flatnonzero(va)
        m, opt = self._model()
        best, best_state, bad, t0 = np.inf, None, 0, time.time()
        self.history_, self.budget_exceeded_ = [], False
        gen = torch.Generator().manual_seed(self.seed)
        for ep in range(self.max_epochs):
            m.train()
            perm = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
            for s in range(0, len(perm), self.batch):
                b = torch.as_tensor(perm[s:s + self.batch])
                pred = self._forward(m, xn[b], xc[b])
                target = z[b].unsqueeze(-1).expand_as(pred) if pred.dim() == 2 else z[b]
                loss = torch.nn.functional.mse_loss(pred, target)   # tabm: mean over the k heads
                opt.zero_grad()
                loss.backward()
                opt.step()
            vmae = float(np.mean(np.abs(self._predict_z(m, xn[va_idx], xc[va_idx]) - z[va_idx].numpy()))) * self.y_std_
            self.history_.append(round(vmae, 4))
            if vmae < best - 1e-4:
                best, bad = vmae, 0
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
            if time.time() - t0 > self.time_cap_s:
                self.budget_exceeded_ = True
                break
            if bad >= self.patience:
                break
        m.load_state_dict(best_state)
        self.model_ = m
        self.fit_info_ = {"epochs": len(self.history_), "best_inner_val_mae": round(best, 4),
                          "budget_exceeded": self.budget_exceeded_, "seconds": round(time.time() - t0, 1),
                          "n_params": int(sum(p.numel() for p in m.parameters()))}
        return self

    def _predict_z(self, m, xn, xc):
        import torch
        m.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(xn), 4096):
                p = self._forward(m, xn[s:s + 4096], xc[s:s + 4096])
                out.append((p.mean(-1) if p.dim() == 2 else p).numpy())
        return np.concatenate(out)

    def predict(self, X: pd.DataFrame):
        xn, xc = self._prep(X.reset_index(drop=True))
        return self._predict_z(self.model_, xn, xc) * self.y_std_ + self.y_mean_
