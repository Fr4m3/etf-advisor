"""
optimizer.py
=============================================================================
Ottimizzazione di portafoglio (Teoria di Markowitz / Max Sharpe).

Obiettivo:
    massimizzare lo Sharpe Ratio  (E[Rp] - Rf) / sigma_p
    soggetto a vincoli di rischio dell'utente (peso azionario target) e
    vincoli "long-only" (pesi >= 0, somma = 1).

A parità di tracking dell'indice, il sistema favorisce gli ETF UCITS con
TER minore e dimensione fondo maggiore (gestito già a monte dai filtri e
dal punteggio di costo). L'ottimizzatore qui massimizza lo Sharpe sulle
serie storiche recuperate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from etf_universe import ETF_BY_TICKER


from etf_universe import ETF_BY_TICKER
from finance import etf_macro_vector, MACRO_BUCKETS


def max_sharpe_weights(
    returns: pd.DataFrame,
    rf: float,
    equity_target: float | None = None,
    equity_tol: float = 0.07,
    geo_target: dict | None = None,
    geo_tol: float = 0.04,
    min_weight: float = 0.0,
    class_targets: dict | None = None,
    class_tol: float = 0.02,
) -> dict:
    """Calcola i pesi ottimali (Max Sharpe) su `returns`.

    Parametri
    ---------
    returns       : DataFrame rendimenti giornalieri (colonne = ticker)
    rf            : tasso risk-free annuo (decimale)
    equity_target : peso azionario desiderato (0..1). Se None, libero.
    equity_tol    : tolleranza sul vincolo equity (+/-)
    geo_target    : dict {macro_bucket: frazione} per vincolare la composizione
                    della PARTE AZIONARIA (es. {"USA":0.6, "Europa":0.2}).
                    I bucket ammessi sono in finance.MACRO_BUCKETS.
    geo_tol       : tolleranza sui vincoli geografici (+/-)
    min_weight    : peso minimo ammesso (0 = long-only puro)

    Ritorna dict {ticker: peso} (solo pesi > 0.5%).
    """
    cols = list(returns.columns)
    n = len(cols)
    if n == 0:
        return {}

    mu = returns.mean().values * 252.0           # rendimenti attesi annui
    cov = returns.cov().values * 252.0           # matrice di covarianza annua

    # Maschere per classe di attivo
    def _mask(cls):
        return np.array(
            [1.0 if ETF_BY_TICKER.get(c, {}).get("asset_class") == cls else 0.0
             for c in cols], dtype=float
        )
    equity_idx = _mask("Equity")
    bond_idx = _mask("Bond")
    alt_idx = _mask("Alternatives")
    # Vettore macro-geografico per ogni colonna
    macro_vecs = [etf_macro_vector(ETF_BY_TICKER.get(c, {})) for c in cols]

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        if vol <= 1e-9:
            return 1e6
        return -(ret - rf) / vol

    # Vincolo somma = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # ---- Vincoli per classe (Azionario / Obbligazionario / Materie Prime) ----
    use_classes = False
    eq_for_geo = equity_target
    if class_targets:
        avail = {
            "Equity": equity_idx.sum() > 0,
            "Bond": bond_idx.sum() > 0,
            "Alternatives": alt_idx.sum() > 0,
        }
        eff = {cls: t for cls, t in class_targets.items()
               if avail.get(cls, False) and t > 1e-6}
        if eff:
            dropped = sum(t for cls, t in class_targets.items()
                          if not (avail.get(cls, False) and t > 1e-6))
            tot = sum(eff.values())
            if tot > 0 and dropped > 0:
                for cls in eff:
                    eff[cls] += dropped * (eff[cls] / tot)
            for cls, tgt in eff.items():
                m = {"Equity": equity_idx, "Bond": bond_idx,
                     "Alternatives": alt_idx}[cls]
                lo = max(0.0, tgt - class_tol)
                hi = min(1.0, tgt + class_tol)
                constraints.append(
                    {"type": "ineq", "fun": lambda w, mm=m, hh=hi: hh - (mm @ w)})
                constraints.append(
                    {"type": "ineq", "fun": lambda w, mm=m, ll=lo: (mm @ w) - ll})
            eq_for_geo = eff.get("Equity", equity_target)
            use_classes = True

    # ---- Vincolo peso azionario (solo se class_targets NON fornito) ----
    if not use_classes and equity_target is not None and equity_idx.sum() > 0:
        lo = max(0.0, equity_target - equity_tol)
        hi = min(1.0, equity_target + equity_tol)
        constraints.append(
            {"type": "ineq", "fun": lambda w: hi - (equity_idx @ w)})
        constraints.append(
            {"type": "ineq", "fun": lambda w: (equity_idx @ w) - lo})

    # ---- Vincoli geografici sulla PARTE AZIONARIA ----
    if geo_target and eq_for_geo is not None and equity_idx.sum() > 0:
        for b, target_frac in geo_target.items():
            if b not in MACRO_BUCKETS:
                continue
            coeff = np.array(
                [equity_idx[i] * macro_vecs[i].get(b, 0.0) for i in range(n)]
            )
            val = target_frac * eq_for_geo
            lo_g = max(0.0, val - geo_tol)
            hi_g = min(1.0, val + geo_tol)
            constraints.append(
                {"type": "ineq", "fun": lambda w, c=coeff, h=hi_g: h - (c @ w)})
            constraints.append(
                {"type": "ineq", "fun": lambda w, c=coeff, l=lo_g: (c @ w) - l})

    bounds = [(min_weight, 1.0) for _ in range(n)]

    # Punto di partenza
    if use_classes:
        x0 = np.zeros(n)
        for cls, tgt in eff.items():
            m = {"Equity": equity_idx, "Bond": bond_idx,
                 "Alternatives": alt_idx}[cls]
            if m.sum() > 0:
                x0 += m * (tgt / m.sum())
    elif equity_target is not None and equity_idx.sum() > 0:
        x0 = np.where(equity_idx == 1, equity_target / equity_idx.sum(),
                      (1 - equity_target) / max(1, n - equity_idx.sum()))
    else:
        x0 = np.full(n, 1.0 / n)

    res = minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not res.success:
        # Fallback: peso uniforme vincolato al target equity
        w = x0.copy()
    else:
        w = res.x

    w = np.clip(w, 0, None)
    if w.sum() > 0:
        w = w / w.sum()

    weights = {cols[i]: float(round(w[i], 4)) for i in range(n) if w[i] > 0.005}
    return weights


def min_variance_weights(
    returns: pd.DataFrame, equity_target=None, equity_tol=0.07
) -> dict:
    """Variante: minimizza la varianza (risk parity di base)."""
    cols = list(returns.columns)
    n = len(cols)
    if n == 0:
        return {}
    cov = returns.cov().values * 252.0
    equity_idx = np.array(
        [1 if ETF_BY_TICKER.get(c, {}).get("asset_class") == "Equity" else 0
         for c in cols], dtype=float
    )

    def var(w):
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if equity_target is not None and equity_idx.sum() > 0:
        lo = max(0.0, equity_target - equity_tol)
        hi = min(1.0, equity_target + equity_tol)
        constraints.append({"type": "ineq", "fun": lambda w: hi - (equity_idx @ w)})
        constraints.append({"type": "ineq", "fun": lambda w: (equity_idx @ w) - lo})
    bounds = [(0.0, 1.0) for _ in range(n)]
    x0 = np.full(n, 1.0 / n)
    res = minimize(var, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1000})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    if w.sum() > 0:
        w = w / w.sum()
    return {cols[i]: float(round(w[i], 4)) for i in range(n) if w[i] > 0.005}
