"""
montecarlo.py
=============================================================================
Simulazione Monte Carlo del capitale futuro (Moto Browniano Geometrico).

Genera N percorsi di prezzo a partire da:
  - mu    : rendimento atteso annuo del portafoglio (da storico)
  - sigma : volatilità annua del portafoglio (da storico)
  - capital: capitale iniziale

Restituisce, per ogni orizzonte richiesto, tre scenari:
  - Pessimistico (10° percentile)
  - Atteso       (50° percentile / mediano)
  - Ottimistico  (90° percentile)
più le serie temporali delle bande per il grafico.
"""

from __future__ import annotations

import numpy as np


def monte_carlo(
    mu: float,
    sigma: float,
    capital: float,
    horizons: tuple[int, ...] = (3, 5, 10, 15),
    max_years: int = 15,
    n_paths: int = 3000,
    steps_per_year: int = 12,
    seed: int | None = 42,
) -> dict:
    """Simulazione Monte Carlo GBM.

    Ritorna dict con:
      't'      : array anni (0..max_years)
      'p10'    : serie 10° percentile (valore portafoglio)
      'p50'    : serie mediana
      'p90'    : serie 90° percentile
      'terminal': {orizzonte: {p10,p50,p90}}
      'capital': capitale iniziale
      'sample_paths': (n_sample, steps) alcuni percorsi per il plot
    """
    rng = np.random.default_rng(seed)
    max_years = max(max_years, max(horizons))
    steps = int(max_years * steps_per_year)
    dt = 1.0 / steps_per_year

    z = rng.normal(size=(n_paths, steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_ret = drift + diffusion * z
    cum_log = np.cumsum(log_ret, axis=1)
    values = capital * np.exp(cum_log)  # (n_paths, steps)

    # Bande percentile nel tempo
    p10 = np.percentile(values, 10, axis=0)
    p50 = np.percentile(values, 50, axis=0)
    p90 = np.percentile(values, 90, axis=0)
    t = np.arange(1, steps + 1) / steps_per_year

    # Terminali agli orizzonti richiesti
    terminal = {}
    for h in sorted(horizons):
        idx = min(steps, max(1, int(round(h * steps_per_year))))
        col = values[:, idx - 1]
        terminal[h] = {
            "p10": float(np.percentile(col, 10)),
            "p50": float(np.percentile(col, 50)),
            "p90": float(np.percentile(col, 90)),
        }

    # Alcuni percorsi di esempio per il grafico
    n_sample = min(40, n_paths)
    sample_paths = values[:n_sample, :]

    return {
        "t": t,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "terminal": terminal,
        "capital": capital,
        "sample_paths": sample_paths,
    }


def monte_carlo_assets(
    asset_mu: np.ndarray,
    asset_cov: np.ndarray,
    weights: dict,
    capital: float,
    rebalance: str = "none",
    horizons: tuple[int, ...] = (3, 5, 10, 15),
    max_years: int = 15,
    n_paths: int = 3000,
    steps_per_year: int = 12,
    seed: int | None = 42,
) -> dict:
    """Simulazione Monte Carlo a livello di SINGOLI ASSET con ribilanciamento.

    Permette di applicare una politica di ribilanciamento (annuale/
    trimestrale/mensile) coerente con il backtest storico, simulando
    percorsi correlati tra gli ETF tramite la matrice di covarianza.

    Parametri
    ---------
    asset_mu  : rendimenti attesi ANNUI per asset (vettore, ordine = weights)
    asset_cov : matrice di covarianza ANNUA (ordine = weights)
    weights   : dict {ticker: peso} (somma=1)
    rebalance : 'none' | 'annual' | 'quarterly' | 'monthly'

    Ritorna la stessa struttura di monte_carlo().
    """
    rng = np.random.default_rng(seed)
    names = list(weights.keys())
    n_assets = len(names)
    w0 = np.array([weights[n] for n in names], dtype=float)
    w0 = w0 / w0.sum()

    # Garantisce definizione positiva per la fattorizzazione di Cholesky
    asset_cov = np.asarray(asset_cov, dtype=float) + 1e-8 * np.eye(n_assets)
    L = np.linalg.cholesky(asset_cov)

    max_years = max(max_years, max(horizons))
    steps = int(max_years * steps_per_year)
    dt = 1.0 / steps_per_year

    # Rendimenti log giornalieri per asset (correlati)
    z = rng.normal(size=(n_paths, steps, n_assets))
    diff = np.einsum("ij,psj->psi", L, z) * np.sqrt(dt)      # parte casuale
    drift = (asset_mu - 0.5 * np.diag(asset_cov)) / steps_per_year
    asset_logret = drift + diff                              # (n_paths, steps, n)
    cum = np.cumsum(asset_logret, axis=1)
    asset_growth = np.exp(cum)                               # crescita cumulata

    # Fattore di crescita SINGLE-STEP
    single = np.empty_like(asset_growth)
    single[:, 0, :] = asset_growth[:, 0, :]
    single[:, 1:, :] = asset_growth[:, 1:, :] / asset_growth[:, :-1, :]

    # Ribilanciamento
    rb_map = {
        "annual": steps_per_year,
        "quarterly": max(1, steps_per_year // 4),
        "monthly": max(1, steps_per_year // 12),
        "none": 10**9,
    }
    rb_steps = rb_map.get(rebalance, 10**9)

    pos = np.tile(w0, (n_paths, 1)).astype(float)   # valore per asset (somma=1)
    port_vals = np.zeros((n_paths, steps))
    for s in range(steps):
        pos = pos * single[:, s, :]
        if rebalance != "none" and rb_steps < 10**8 and (s + 1) % rb_steps == 0:
            tot = pos.sum(axis=1, keepdims=True)
            pos = w0 * tot
        port_vals[:, s] = pos.sum(axis=1)

    values = capital * port_vals

    p10 = np.percentile(values, 10, axis=0)
    p50 = np.percentile(values, 50, axis=0)
    p90 = np.percentile(values, 90, axis=0)
    t = np.arange(1, steps + 1) / steps_per_year

    terminal = {}
    for h in sorted(horizons):
        idx = min(steps, max(1, int(round(h * steps_per_year))))
        col = values[:, idx - 1]
        terminal[h] = {
            "p10": float(np.percentile(col, 10)),
            "p50": float(np.percentile(col, 50)),
            "p90": float(np.percentile(col, 90)),
        }

    n_sample = min(40, n_paths)
    sample_paths = values[:n_sample, :]
    return {
        "t": t, "p10": p10, "p50": p50, "p90": p90,
        "terminal": terminal, "capital": capital,
        "sample_paths": sample_paths,
    }
