"""
finance.py
=============================================================================
Calcoli finanziari sul portafoglio: rendimenti, volatilità, Sharpe,
Maximum Drawdown, TER ponderato, esposizione geografica/settoriale.

Tutte le funzioni lavorano su:
  - `prices`: DataFrame di prezzi storici (colonne = ticker)
  - `weights`: dict {ticker: peso (0..1), somma=1}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_universe import ETF_BY_TICKER


# Macro-aree geografiche usate per l'ottimizzazione su base geografica.
# Ogni ETF viene mappato dalle sue esposizioni regionali in questi bucket.
MACRO_BUCKETS = ["USA", "Europa", "Pacifico", "Emergenti",
                 "Altri Sviluppati", "Obbligazionario"]

# 6 macro-aree geografiche coerenti con l'ottimizzatore di Pagina 2/4.
GEO_KEYS = ["USA", "Europa", "Giappone", "Emergenti", "Pacifico Sviluppato",
            "Altri Sviluppati"]
# 11 settori coerenti con l'ottimizzatore di Pagina 2/4.
SECTOR_KEYS = ["Tecnologia", "Finanziari", "Salute", "Industria",
              "Consumi Discrezionali", "Staples (Beni di Base)", "Energia",
              "Utilities", "Materiali", "Comunicazioni", "Real Estate"]

# Mappa esposizioni regionali dell'ETF -> 6 GEO_KEYS (usate da Pagina 2/4)
REGION_TO_GEO = {
    "USA": "USA", "Stati Uniti": "USA",
    "Europa": "Europa", "Germania": "Europa", "Francia": "Europa",
    "Regno Unito": "Europa", "Svizzera": "Altri Sviluppati",
    "Giappone (Pacifico)": "Giappone", "Giappone": "Giappone",
    "Emergenti": "Emergenti", "Cina": "Emergenti", "India": "Emergenti",
    "Pacifico Sviluppato": "Pacifico Sviluppato",
    "Altri Sviluppati": "Altri Sviluppati",
    "Sviluppati (Global REIT)": "Altri Sviluppati", "Globale": "Altri Sviluppati",
}

REGION_TO_MACRO = {
    "USA": "USA",
    "Europa": "Europa",
    "Pacifico Sviluppato": "Pacifico",
    "Giappone (Pacifico)": "Pacifico",
    "Emergenti": "Emergenti",
    "Altri Sviluppati": "Altri Sviluppati",
    "Obbligazionario Globale (hedged EUR)": "Obbligazionario",
    "Obbligazionario Eurozona": "Obbligazionario",
    "Obbligazionario Eurozona (Corp)": "Obbligazionario",
    "Obbligazionario Globale Corp (hedged EUR)": "Obbligazionario",
    "Sviluppati (Global REIT)": "Altri Sviluppati",
    "Globale": "Altri Sviluppati",
}

# Mappa nomi Paese (JustETF IT) -> macro-region usate dall'app
COUNTRY_TO_MACRO = {
    "Stati Uniti": "USA", "Canada": "USA",
    "Germania": "Europa", "Francia": "Europa", "Italia": "Europa",
    "Spagna": "Europa", "Paesi Bassi": "Europa", "Svezia": "Europa",
    "Belgio": "Europa", "Austria": "Europa", "Finlandia": "Europa",
    "Norvegia": "Europa", "Danimarca": "Europa", "Portogallo": "Europa",
    "Irlanda": "Europa", "Lussemburgo": "Europa", "Polonia": "Europa",
    "Europa": "Europa", "Zona Euro": "Europa",
    "Giappone": "Giappone", "Regno Unito": "Regno Unito",
    "Svizzera": "Svizzera", "Australia": "Australia",
    "Nuova Zelanda": "Pacifico Sviluppato", "Singapore": "Pacifico Sviluppato",
    "Hong Kong": "Pacifico Sviluppato",
    "Cina": "Cina", "India": "India",
    "Brasile": "Emergenti", "Messico": "Emergenti", "Sudafrica": "Emergenti",
    "Russia": "Emergenti", "Turchia": "Emergenti", "Corea": "Emergenti",
    "Taiwan": "Emergenti", "Tailandia": "Emergenti", "Indonesia": "Emergenti",
    "Malesia": "Emergenti", "Filippine": "Emergenti", "Cile": "Emergenti",
    "Colombia": "Emergenti", "Perù": "Emergenti", "Egitto": "Emergenti",
    "Ungheria": "Emergenti", "Repubblica Ceca": "Emergenti", "Grecia": "Emergenti",
    "Altri": "Altri Sviluppati", "Altro": "Altri Sviluppati",
}

# Mappa nomi Settore (JustETF IT) -> settori usati dall'app
SECTOR_MAP = {
    "Informatica": "Tecnologia", "Finanza": "Finanziari",
    "Industria": "Industria",
    "Beni di consumo non ciclici": "Staples (Beni di Base)",
    "Beni di consumo ciclici": "Consumi Discrezionali",
    "Salute": "Salute", "Energia": "Energia", "Materiali": "Materiali",
    "Immobiliare": "Real Estate", "Telecomunicazioni": "Comunicazioni",
    "Utility": "Utilities", "Altri": "Altro", "Altro": "Altro",
}


def etf_macro_vector(etf: dict) -> dict:
    """Vettore dell'ETF nei macro-bucket geografici (somma = 1)."""
    vec = {b: 0.0 for b in MACRO_BUCKETS}
    for region, frac in etf.get("region", {}).items():
        b = REGION_TO_MACRO.get(region, "Altri Sviluppati")
        vec[b] += frac
    return vec


def etf_geo_vector(etf: dict) -> dict:
    """Vettore dell'ETF nei 6 macro-bucket geografici (GEO_KEYS)."""
    vec = {b: 0.0 for b in GEO_KEYS}
    for region, frac in etf.get("region", {}).items():
        b = REGION_TO_GEO.get(region)
        if b:
            vec[b] += frac
    return vec


def etf_sector_vector(etf: dict) -> dict:
    """Vettore dell'ETF nei 11 settori (SECTOR_KEYS)."""
    vec = {b: 0.0 for b in SECTOR_KEYS}
    for s, frac in etf.get("sectors", {}).items():
        if s in vec:
            vec[s] += frac
    return vec


# ---------------------------------------------------------------------------
# Rendimenti e statistiche base
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def annualized_return(returns: pd.DataFrame) -> pd.Series:
    """Rendimento medio annuo per colonna (media dei rendimenti giornalieri *252)."""
    return returns.mean() * 252


def annualized_vol(returns: pd.DataFrame) -> pd.Series:
    """Volatilità annualizzata per colonna (dev. std giornaliera * sqrt(252))."""
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.DataFrame, rf: float) -> pd.Series:
    """Sharpe ratio annuo per colonna: (E[R]-Rf)/sigma."""
    ann_ret = annualized_return(returns)
    ann_vol = annualized_vol(returns)
    return (ann_ret - rf) / ann_vol.replace(0, np.nan)


def max_drawdown(prices: pd.Series) -> float:
    """Massimo drawdown storico (%) da picco a minimo. Ritorna valore negativo."""
    roll_max = prices.cummax()
    dd = prices / roll_max - 1.0
    return float(dd.min())


# ---------------------------------------------------------------------------
# Statistiche di portafoglio (dati i pesi)
# ---------------------------------------------------------------------------
def portfolio_annual_metrics(returns: pd.DataFrame, weights: dict, rf: float):
    """Ritorna (rendimento_annuo, vol_annua, sharpe) del portafoglio."""
    cols = [c for c in returns.columns if c in weights]
    w = np.array([weights[c] for c in cols], dtype=float)
    if w.sum() == 0:
        return 0.0, 0.0, 0.0
    w = w / w.sum()
    mu_daily = returns[cols].mean().values
    cov_daily = returns[cols].cov().values
    ann_ret = float(w @ mu_daily * 252)
    ann_vol = float(np.sqrt(w @ cov_daily @ w * 252))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    return ann_ret, ann_vol, sharpe


def portfolio_price_series(prices: pd.DataFrame, weights: dict,
                             rebalance: str = "none") -> pd.Series:
    """Serie del valore (indice 100) del portafoglio.

    `rebalance`:
      'none'       -> buy & hold (i pesi derivano e non vengono riallineati)
      'annual'     -> ribilanciamento annuale al target
      'quarterly'  -> ribilanciamento trimestrale
      'monthly'    -> ribilanciamento mensile
    """
    cols = [c for c in prices.columns if c in weights and weights[c] > 0]
    if not cols:
        return pd.Series(dtype=float)
    w0 = np.array([weights[c] for c in cols], dtype=float)
    w0 = w0 / w0.sum()

    daily_ret = prices[cols].pct_change().fillna(0.0)
    n = len(prices)
    rb_days = {"annual": 252, "quarterly": 63, "monthly": 21, "none": 10**9}
    period = rb_days.get(rebalance, 10**9)

    pos = w0.copy().astype(float)          # valore allocato per asset (somma=1)
    value = np.zeros(n)
    value[0] = 1.0
    for t in range(1, n):
        pos = pos * (1.0 + daily_ret.iloc[t].values)
        if rebalance != "none" and period < 10**8 and (t % period == 0):
            tot = pos.sum()
            pos = w0 * tot
        value[t] = pos.sum()
    return pd.Series(value / value[0] * 100.0, index=prices.index)


def weighted_ter(weights: dict) -> float:
    """TER medio ponderato del portafoglio (in %)."""
    total = sum(weights.values())
    if total == 0:
        return 0.0
    s = sum(ETF_BY_TICKER[t]["ter"] * w for t, w in weights.items()
            if t in ETF_BY_TICKER)
    return s / total


def weighted_cost_eur(weights: dict, capital: float) -> float:
    """Costo annuo stimato in € sul capitale iniziale."""
    return weighted_ter(weights) / 100.0 * capital


def asset_annual_moments(returns: pd.DataFrame, order: list) -> tuple:
    """Ritorna (vettore rendimenti attesi annui, matrice di covarianza annua)
    allineati all'ordine `order` dei ticker."""
    sub = returns[order]
    mu = sub.mean().values * 252.0
    cov = sub.cov().values * 252.0
    return mu, cov


# ---------------------------------------------------------------------------
# Esposizione geografica e settoriale aggregate
# ---------------------------------------------------------------------------
def aggregate_exposure(weights: dict, field: str) -> dict:
    """Aggrega l'esposizione `region` o `sectors` dei pesi di portafoglio.

    `field` = 'region' o 'sectors'. Ritorna dict {categoria: peso %}.
    """
    out: dict[str, float] = {}
    for t, w in weights.items():
        etf = ETF_BY_TICKER.get(t)
        if etf is None or w <= 0:
            continue
        breakdown = etf.get(field, {})
        for cat, frac in breakdown.items():
            out[cat] = out.get(cat, 0.0) + w * frac
    return {k: round(v * 100, 2) for k, v in out.items() if v > 0}
