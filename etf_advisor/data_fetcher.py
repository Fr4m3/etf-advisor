"""
data_fetcher.py
=============================================================================
Recupero dati di mercato (prezzi storici) e tasso risk-free.

Strategia "robusta":
  1. Prova a scaricare i prezzi reali da Yahoo Finance (yfinance) per i
     ticker UCITS dell'universo.
  2. Se un ticker manca o il download fallisce, genera una serie storica
     *sintetica* coerente con la classe di attivo (così l'app funziona
     sempre, anche offline o dietro firewall).
  3. Il tasso risk-free è configurabile; di default usa un valore di
     riferimento del tasso BCE/Euribor (modificabile nell'interfaccia).

Inclusione del modulo di scraping JustETF (best-effort): fetch_justetf()
effettua una richiesta HTTP "stealth" alla scheda dell'ETF e tenta di
estrarre TER e dimensione del fondo. Da usare come arricchimento opzionale,
non come dipendenza critica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_universe import ETF_UNIVERSE, ETF_BY_TICKER

# Tasso risk-free di default (riferimento BCE/Euribor, modificabile da UI).
DEFAULT_RISK_FREE = 0.025


# ---------------------------------------------------------------------------
# Tasso risk-free
# ---------------------------------------------------------------------------
def get_risk_free_rate(override: float | None = None) -> float:
    """Restituisce il tasso risk-free annuo (decimale, es. 0.025 = 2.5%).

    Se `override` è fornito lo usa, altrimenti il default.
    (Estensione possibile: recuperare live da yfinance '^IRX' o API BCE.)
    """
    if override is not None:
        return float(override)
    return DEFAULT_RISK_FREE


# ---------------------------------------------------------------------------
# Generazione sintetica (fallback offline)
# ---------------------------------------------------------------------------
def _synthetic_prices(ticker: str, period: str = "5y") -> pd.Series:
    etf = ETF_BY_TICKER.get(ticker)
    if etf is None:
        # ETF sconosciuto: parametri generici azionari
        mu, sigma = 0.07, 0.15
    else:
        mu, sigma = etf["exp_return"], etf["vol"]

    years = 5
    if isinstance(period, str) and "y" in period:
        try:
            years = int(period.replace("y", ""))
        except ValueError:
            years = 5
    periods = max(252, years * 252)

    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    daily_mu = (mu - 0.5 * sigma**2) / 252
    daily_sigma = sigma / np.sqrt(252)
    rets = rng.normal(daily_mu, daily_sigma, periods)
    price = 100.0 * np.exp(np.cumsum(rets))
    n = len(price)
    end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=end, periods=n)
    if len(idx) != n:
        # fallback robusto: calendario giornaliero
        idx = pd.date_range(end=end, periods=n, freq="D")
    return pd.Series(price, index=idx, name=ticker)


# ---------------------------------------------------------------------------
# Download reale da Yahoo Finance + fallback
# ---------------------------------------------------------------------------
def fetch_prices(tickers, period: str = "5y", use_live: bool = True) -> pd.DataFrame:
    """Restituisce un DataFrame di prezzi storici (colonne = ticker).

    `tickers` può essere una stringa (singolo) o una lista.
    Per ogni ticker, se il download live fallisce si usa la serie sintetica.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)

    live = pd.DataFrame()
    if use_live:
        try:
            import yfinance as yf
            yf_map = {ETF_BY_TICKER[t]["yf_ticker"]: t for t in tickers
                      if t in ETF_BY_TICKER}
            # Se un ticker non ha yf_ticker, lo trattiamo come sintetico
            try:
                raw = yf.download(
                    list(yf_map.keys()), period=period,
                    auto_adjust=True, progress=False,
                )
            except Exception:
                raw = pd.DataFrame()

            if isinstance(raw, pd.DataFrame) and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw["Close"]
                else:
                    close = raw[["Close"]] if "Close" in raw else raw
                # rinomina le colonne coi ticker interni
                rename = {k: v for k, v in yf_map.items()}
                close = close.rename(columns=rename)
                live = close
        except Exception:
            live = pd.DataFrame()

    # Assembla risultato: live dove disponibile, altrimenti sintetico
    result = pd.DataFrame()
    for t in tickers:
        has_live = t in live.columns and live[t].notna().sum() > 20
        if has_live:
            result[t] = live[t]
        else:
            result[t] = _synthetic_prices(t, period)

    return result.dropna(how="all").sort_index()


# ---------------------------------------------------------------------------
# Scraping JustETF (best-effort, opzionale)
# ---------------------------------------------------------------------------
def fetch_justetf(isin: str, timeout: int = 15) -> dict:
    """Tenta di recuperare metadati da JustETF per ISIN.

    Restituisce {"isin","name","ter","fund_size_m","region","sectors","ok"}.
    `region`/`sectors` sono dict {macro_label: frazione} ricavati dalle tabelle
    "Paesi"/"Settori" della scheda JustETF (best-effort). In caso di
    blocco/errore ritorna ok=False con campi vuoti.
    """
    out = {"isin": isin, "name": None, "ter": None,
           "fund_size_m": None, "region": {}, "sectors": {}, "ok": False}
    try:
        import requests, re
        from finance import COUNTRY_TO_MACRO, SECTOR_MAP
        url = f"https://www.justetf.com/it/etf-profile.html?isin={isin}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "it-IT,it;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return out
        html = r.text
        # Nome del fondo dal <title>
        mt = re.search(r"<title>([^|<]+)", html)
        if mt:
            out["name"] = mt.group(1).strip()
        # TER: pattern "TER: 0,20 %" (notazione europea)
        m = re.search(r"TER[:\s]+([0-9]+,[0-9]+)\s*%", html)
        if m:
            out["ter"] = float(m.group(1).replace(",", "."))
        # Dimensione fondo: "Fondo: 12.345 mil. €" (varie grafie)
        m = re.search(r"([0-9]{1,3}(?:\.[0-9]{3})*)\s*mil\.?\s*€", html)
        if m:
            out["fund_size_m"] = float(m.group(1).replace(".", ""))
        # Paesi -> regioni (macro-bucket dell'app)
        cnames = re.findall(
            r'tl_etf-holdings_countries_value_name">([^<]+)</td>', html)
        cpcts = re.findall(
            r'tl_etf-holdings_countries_value_percentage">([0-9]+,[0-9]+)%', html)
        region = {}
        for nm, pc in zip(cnames, cpcts):
            nm = nm.strip()
            try:
                val = float(pc.replace(",", ".")) / 100.0
            except ValueError:
                continue
            bucket = COUNTRY_TO_MACRO.get(nm, "Altri Sviluppati")
            region[bucket] = region.get(bucket, 0.0) + val
        # Settori -> settori dell'app
        snames = re.findall(
            r'tl_etf-holdings_sectors_value_name">([^<]+)</td>', html)
        spcts = re.findall(
            r'tl_etf-holdings_sectors_value_percentage">([0-9]+,[0-9]+)%', html)
        sectors = {}
        for nm, pc in zip(snames, spcts):
            nm = nm.strip()
            try:
                val = float(pc.replace(",", ".")) / 100.0
            except ValueError:
                continue
            bucket = SECTOR_MAP.get(nm, "Altro")
            sectors[bucket] = sectors.get(bucket, 0.0) + val
        out["region"] = region
        out["sectors"] = sectors
        out["name"] = out["name"] or ETF_BY_ISIN_NAME(isin)
        out["ok"] = any([
            out["ter"] is not None,
            out["fund_size_m"] is not None,
            bool(region), bool(sectors),
        ])
    except Exception:
        pass
    return out


def ETF_BY_ISIN_NAME(isin: str):
    # helper locale per evitare import circolare di nome
    from etf_universe import ETF_BY_ISIN
    return ETF_BY_ISIN.get(isin, {}).get("name")


# ---------------------------------------------------------------------------
# Aggiornamento metadati (TER / dimensione) da JustETF
# ---------------------------------------------------------------------------
def refresh_metadata(etfs=None, timeout: int = 15) -> list:
    """Aggiorna TER e dimensione fondo degli ETF leggendo JustETF per ISIN.

    Mutando gli oggetti dell'universo in-place, i valori aggiornati si
    riflettono ovunque (filtri, tabelle, ottimizzatore). In caso di errore o
    blocco, il valore di riferimento viene mantenuto.

    Ritorna una lista di dict di stato (ticker, ter_old/new, size_old/new, ok, msg).
    """
    if etfs is None:
        from etf_universe import ETF_UNIVERSE
        etfs = ETF_UNIVERSE
    results = []
    for e in etfs:
        status = {
            "ticker": e["ticker"], "isin": e["isin"], "name": e["name"],
            "ter_old": e["ter"], "size_old": e["fund_size_m"],
            "ter_new": e["ter"], "size_new": e["fund_size_m"],
            "ok": False, "msg": "",
        }
        try:
            r = fetch_justetf(e["isin"], timeout=timeout)
            if r.get("ok"):
                if r.get("ter") is not None:
                    e["ter"] = r["ter"]
                    status["ter_new"] = r["ter"]
                if r.get("fund_size_m") is not None:
                    e["fund_size_m"] = r["fund_size_m"]
                    status["size_new"] = r["fund_size_m"]
                status["ok"] = True
                status["msg"] = "aggiornato"
            else:
                status["msg"] = "non disponibile (valore di riferimento mantenuto)"
        except Exception as ex:  # noqa: BLE001
            status["msg"] = f"errore: {ex}"
        results.append(status)
    return results


# ---------------------------------------------------------------------------
# Metadati (tabella di selezione)
# ---------------------------------------------------------------------------
def universe_table() -> pd.DataFrame:
    """DataFrame riepilogativo dell'universo ETF (per la UI)."""
    rows = []
    for e in ETF_UNIVERSE:
        rows.append({
            "Ticker": e["ticker"],
            "ISIN": e["isin"],
            "Nome": e["name"],
            "Asset Class": e["asset_class"],
            "Regione": ", ".join(list(e["region"].keys())[:2]),
            "TER (%)": e["ter"],
            "Dim. Fondo (M€)": e["fund_size_m"],
            "Accumulo": "Sì" if e["accumulation"] else "No",
            "UCITS": "Sì",
        })
    return pd.DataFrame(rows)
