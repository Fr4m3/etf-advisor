"""
etf_universe.py
=============================================================================
Universo curato di ETF UCITS reali (dati anagrafici di riferimento) utilizzati
dall'applicazione come "lista bianca" selezionabile.

NOTA SULLA QUALITÀ DEI DATI
---------------------------
- I campi anagrafici (ISIN, ticker, nome, TER, dimensione fondo, tipo
  accumulo/distribuzione, classe di attivo, esposizione geografica e
  settoriale) sono *parametri di riferimento* inseriti a mano e da
  aggiornare periodicamente.
- I prezzi storici e i rendimenti reali vengono recuperati a runtime da
  Yahoo Finance (vedi data_fetcher.py). Se il download fallisce, il sistema
  genera una serie storica *sintetica* (corretta per classe di attivo) così
  l'applicazione resta sempre dimostrabile offline.

Le esposizioni geografiche/settoriali sono approssimazioni basate sui
prospetti ufficiali degli indici sottostanti (MSCI / FTSE / Bloomberg) e
servono a produrre le dashboard di allocazione.
"""

from __future__ import annotations

# Suddivisione settoriale "Broad Market" di riferimento (stile GICS),
# approssimata per gli ETF azionari globali/paese.
BROAD_SECTORS = {
    "Tecnologia": 0.22,
    "Salute": 0.12,
    "Finanziari": 0.13,
    "Consumi Discrezionali": 0.11,
    "Comunicazioni": 0.08,
    "Industria": 0.10,
    "Staples (Beni di Base)": 0.06,
    "Energia": 0.05,
    "Utilities": 0.03,
    "Materiali": 0.03,
    "Real Estate": 0.04,
    "Altro": 0.03,
}

# Universo ETF UCITS (tutti conformi UCITS, regolamento europeo).
#   isin            : codice ISIN ufficiale
#   ticker          : ticker usato internamente (yfinance)
#   yf_ticker       : ticker Yahoo Finance per il download dei prezzi
#   name            : nome completo del fondo
#   asset_class     : 'Equity' | 'Bond'
#   region          : esposizione geografica (dict, somma ~1.0)
#   sectors         : esposizione settoriale (dict, somma ~1.0)
#   ter             : Total Expense Ratio annuo (in %)
#   fund_size_m     : dimensione del fondo in milioni di EUR (applicare filtro >=100)
#   accumulation    : True = a accumulo, False = a distribuzione
#   exp_return      : rendimento atteso ANNUO (solo fallback sintetico)
#   vol             : volatilità ANNUA (solo fallback sintetico)
ETF_UNIVERSE = [
    # ---------------- AZIONARI (Equity) ----------------
    {
        "isin": "IE00BK5BQT80",
        "ticker": "VWCE",
        "yf_ticker": "VWCE.DE",
        "name": "Vanguard FTSE All-World UCITS ETF",
        "asset_class": "Equity",
        "region": {
            "USA": 0.60, "Europa": 0.12, "Pacifico Sviluppato": 0.10,
            "Emergenti": 0.12, "Altri Sviluppati": 0.06,
        },
        "sectors": dict(BROAD_SECTORS),
        "ter": 0.22, "fund_size_m": 14000, "accumulation": True,
        "exp_return": 0.085, "vol": 0.15,
    },
    {
        "isin": "IE00B4L5Y983",
        "ticker": "SWDA",
        "yf_ticker": "SWDA.MI",
        "name": "iShares Core MSCI World UCITS ETF",
        "asset_class": "Equity",
        "region": {
            "USA": 0.70, "Europa": 0.15, "Pacifico Sviluppato": 0.10,
            "Altri Sviluppati": 0.05,
        },
        "sectors": dict(BROAD_SECTORS),
        "ter": 0.20, "fund_size_m": 85000, "accumulation": True,
        "exp_return": 0.085, "vol": 0.15,
    },
    {
        "isin": "IE00B5BMR087",
        "ticker": "CSPX",
        "yf_ticker": "CSPX.L",
        "name": "iShares Core S&P 500 UCITS ETF",
        "asset_class": "Equity",
        "region": {"USA": 1.00},
        "sectors": dict(BROAD_SECTORS),
        "ter": 0.07, "fund_size_m": 95000, "accumulation": True,
        "exp_return": 0.090, "vol": 0.16,
    },
    {
        "isin": "IE00B52XQP83",
        "ticker": "IEMA",
        "yf_ticker": "IEMA.MI",
        "name": "iShares Core MSCI Europe UCITS ETF",
        "asset_class": "Equity",
        "region": {"Europa": 1.00},
        "sectors": dict(BROAD_SECTORS),
        "ter": 0.12, "fund_size_m": 9000, "accumulation": True,
        "exp_return": 0.070, "vol": 0.17,
    },
    {
        "isin": "IE00B4L5YX21",
        "ticker": "SJPA",
        "yf_ticker": "SJPA.MI",
        "name": "iShares Core MSCI Japan UCITS ETF",
        "asset_class": "Equity",
        "region": {"Giappone (Pacifico)": 1.00},
        "sectors": dict(BROAD_SECTORS),
        "ter": 0.12, "fund_size_m": 3500, "accumulation": True,
        "exp_return": 0.070, "vol": 0.16,
    },
    {
        "isin": "IE00BKM4GZ66",
        "ticker": "EIMI",
        "yf_ticker": "EIMI.MI",
        "name": "iShares Core MSCI EM IMI UCITS ETF",
        "asset_class": "Equity",
        "region": {"Emergenti": 1.00},
        "sectors": dict(BROAD_SECTORS),
        "ter": 0.18, "fund_size_m": 5500, "accumulation": True,
        "exp_return": 0.085, "vol": 0.20,
    },
    # ---------------- OBBLIGAZIONARI (Bond) ----------------
    {
        "isin": "IE00BDBRDM35",
        "ticker": "AGGH",
        "yf_ticker": "AGGH.MI",
        "name": "iShares Core Global Aggregate Bond UCITS ETF",
        "asset_class": "Bond",
        "region": {"Obbligazionario Globale (hedged EUR)": 1.00},
        "sectors": {"Obbligazioni": 1.00},
        "ter": 0.10, "fund_size_m": 11000, "accumulation": True,
        "exp_return": 0.025, "vol": 0.05,
    },
    {
        "isin": "IE00B4WXJD03",
        "ticker": "IEGA",
        "yf_ticker": "IEGA.MI",
        "name": "iShares Core Euro Govt Bond UCITS ETF",
        "asset_class": "Bond",
        "region": {"Obbligazionario Eurozona": 1.00},
        "sectors": {"Obbligazioni": 1.00},
        "ter": 0.09, "fund_size_m": 7000, "accumulation": True,
        "exp_return": 0.020, "vol": 0.04,
    },
    {
        "isin": "IE00B3F81R35",
        "ticker": "IEAC",
        "yf_ticker": "IEAC.MI",
        "name": "iShares Euro Corp Bond UCITS ETF",
        "asset_class": "Bond",
        "region": {"Obbligazionario Eurozona (Corp)": 1.00},
        "sectors": {"Obbligazioni": 1.00},
        "ter": 0.09, "fund_size_m": 8000, "accumulation": True,
        "exp_return": 0.030, "vol": 0.05,
    },
    {
        "isin": "IE00B7GZW936",
        "ticker": "IBGM",
        "yf_ticker": "IBGM.MI",
        "name": "iShares Global Corp Bond UCITS ETF",
        "asset_class": "Bond",
        "region": {"Obbligazionario Globale Corp (hedged EUR)": 1.00},
        "sectors": {"Obbligazioni": 1.00},
        "ter": 0.15, "fund_size_m": 4000, "accumulation": True,
        "exp_return": 0.035, "vol": 0.06,
    },
]

# Categoria di default per gli ETF gia' presenti
_DEFAULT_CAT = {
    "VWCE": "Azionario Broad", "SWDA": "Azionario Broad",
    "CSPX": "Azionario Broad", "IEMA": "Azionario Broad",
    "SJPA": "Azionario Broad", "EIMI": "Azionario Broad",
    "AGGH": "Obbligazionario", "IEGA": "Obbligazionario",
    "IEAC": "Obbligazionario", "IBGM": "Obbligazionario",
}
for _e in ETF_UNIVERSE:
    _e.setdefault("category", _DEFAULT_CAT.get(_e["ticker"], "Altro"))

# ---------------------------------------------------------------------------
# Estensione: ETF SETTORIALI, REAL ESTATE e MATERIE PRIME (tutti UCITS)
# I ticker XETRA (XDxx.DE) sono i settoriali S&P 500 di iShares (Acc, USD).
# Metadati (ISIN/TER/dimensione) di riferimento; prezzi reali da Yahoo Finance
# con fallback sintetico.
# ---------------------------------------------------------------------------
NEW_ETFS = [
    {"isin": "IE00B3WTLP36", "ticker": "XDWT", "yf_ticker": "XDWT.DE",
     "name": "iShares S&P 500 Info Technology Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Tecnologia": 1.0},
     "ter": 0.15, "fund_size_m": 20000, "accumulation": True,
     "exp_return": 0.16, "vol": 0.22},
    {"isin": "IE00B3WTLR50", "ticker": "XDWP", "yf_ticker": "XDWP.DE",
     "name": "iShares S&P 500 Health Care Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Salute": 1.0},
     "ter": 0.15, "fund_size_m": 8000, "accumulation": True,
     "exp_return": 0.10, "vol": 0.16},
    {"isin": "IE00B3WTLN28", "ticker": "XDWF", "yf_ticker": "XDWF.DE",
     "name": "iShares S&P 500 Financials Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Finanziari": 1.0},
     "ter": 0.15, "fund_size_m": 5000, "accumulation": True,
     "exp_return": 0.10, "vol": 0.18},
    {"isin": "IE00B3WTLJ19", "ticker": "XDWC", "yf_ticker": "XDWC.DE",
     "name": "iShares S&P 500 Consumer Discretionary Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Consumi Discrezionali": 1.0},
     "ter": 0.15, "fund_size_m": 4000, "accumulation": True,
     "exp_return": 0.12, "vol": 0.20},
    {"isin": "IE00B3WTLH94", "ticker": "XDWE", "yf_ticker": "XDWE.DE",
     "name": "iShares S&P 500 Energy Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Energia": 1.0},
     "ter": 0.15, "fund_size_m": 3000, "accumulation": True,
     "exp_return": 0.08, "vol": 0.22},
    {"isin": "IE00B3WTLF78", "ticker": "XDWU", "yf_ticker": "XDWU.DE",
     "name": "iShares S&P 500 Utilities Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Utilities": 1.0},
     "ter": 0.15, "fund_size_m": 1500, "accumulation": True,
     "exp_return": 0.07, "vol": 0.14},
    {"isin": "IE00B3WTLM11", "ticker": "XDWS", "yf_ticker": "XDWS.DE",
     "name": "iShares S&P 500 Consumer Staples Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Staples (Beni di Base)": 1.0},
     "ter": 0.15, "fund_size_m": 2500, "accumulation": True,
     "exp_return": 0.08, "vol": 0.13},
    {"isin": "IE00B3WTLK95", "ticker": "XDWI", "yf_ticker": "XDWI.DE",
     "name": "iShares S&P 500 Industrials Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Industria": 1.0},
     "ter": 0.15, "fund_size_m": 3000, "accumulation": True,
     "exp_return": 0.10, "vol": 0.17},
    {"isin": "IE00B3WTLQ43", "ticker": "XDWM", "yf_ticker": "XDWM.DE",
     "name": "iShares S&P 500 Materials Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Materiali": 1.0},
     "ter": 0.15, "fund_size_m": 1500, "accumulation": True,
     "exp_return": 0.09, "vol": 0.19},
    {"isin": "IE00B3WTLG85", "ticker": "XDWO", "yf_ticker": "XDWO.DE",
     "name": "iShares S&P 500 Communication Sector UCITS ETF",
     "asset_class": "Equity", "category": "Azionario Settoriale",
     "region": {"USA": 1.0}, "sectors": {"Comunicazioni": 1.0},
     "ter": 0.15, "fund_size_m": 2000, "accumulation": True,
     "exp_return": 0.11, "vol": 0.19},
    {"isin": "IE00B1FZS350", "ticker": "IWDP", "yf_ticker": "IWDP.L",
     "name": "iShares Developed Markets Property Yield UCITS ETF",
     "asset_class": "Equity", "category": "Real Estate",
     "region": {"Sviluppati (Global REIT)": 1.0}, "sectors": {"Real Estate": 1.0},
     "ter": 0.40, "fund_size_m": 1250, "accumulation": False,
     "exp_return": 0.07, "vol": 0.18},
    {"isin": "IE00BDFL4P12", "ticker": "IGCC", "yf_ticker": "IGCC.L",
     "name": "iShares Diversified Commodity Swap UCITS ETF",
     "asset_class": "Alternatives", "category": "Materie Prime",
     "region": {"Globale": 1.0}, "sectors": {"Materie Prime": 1.0},
     "ter": 0.19, "fund_size_m": 2260, "accumulation": True,
     "exp_return": 0.04, "vol": 0.15},
]
ETF_UNIVERSE.extend(NEW_ETFS)

# Indici di comodo
ETF_BY_TICKER = {e["ticker"]: e for e in ETF_UNIVERSE}
ETF_BY_ISIN = {e["isin"]: e for e in ETF_UNIVERSE}
ETF_BY_YF = {e["yf_ticker"]: e for e in ETF_UNIVERSE}


def get_etf(ticker: str) -> dict:
    """Restituisce i metadati di un ETF dal suo ticker interno."""
    return ETF_BY_TICKER.get(ticker)
