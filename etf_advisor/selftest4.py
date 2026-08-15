"""Test: allocazione custom + vincoli geografici sull'ottimizzatore."""
import numpy as np
import pandas as pd

import data_fetcher
from finance import daily_returns, aggregate_exposure, etf_macro_vector, MACRO_BUCKETS
from optimizer import max_sharpe_weights
from etf_universe import ETF_UNIVERSE

tickers = [e["ticker"] for e in ETF_UNIVERSE]
prices = data_fetcher.fetch_prices(tickers, period="5y", use_live=False)
returns = daily_returns(prices).dropna()
rf = 0.025

print("=== 1) ALLOCazione CUSTOM (equity 30% invece di 50%) ===")
w = max_sharpe_weights(returns, rf, equity_target=0.30)
eq_w = sum(v for t, v in w.items()
           if ETF_UNIVERSE and __import__('etf_universe').ETF_BY_TICKER[t]['asset_class'] == 'Equity')
print(f"  equity effettivo: {eq_w*100:.1f}% (target 30%)  | nETF={len(w)}")

print("\n=== 2) GEO: Home bias Europa (USA 35 / Europa 45) ===")
geo = {"USA": 0.35, "Europa": 0.45, "Pacifico": 0.07,
       "Emergenti": 0.10, "Altri Sviluppati": 0.03}
w2 = max_sharpe_weights(returns, rf, equity_target=0.50, geo_target=geo)
# esposizione macro effettiva (su TUTTO il portafoglio)
macro = {b: 0.0 for b in MACRO_BUCKETS}
for t, v in w2.items():
    mv = etf_macro_vector(__import__('etf_universe').ETF_BY_TICKER[t])
    for b in MACRO_BUCKETS:
        macro[b] += v * mv[b]
print("  Esposizione macro effettiva (%):")
for b in MACRO_BUCKETS:
    print(f"    {b:18s}: {macro[b]*100:5.1f}%   (target geo USA/EUR/PAC/EME/ALT applicati alla parte azionaria 50%)")
# Verifica che la parte azionaria rispetti ~ USA35/EUR45
eq_w2 = sum(v for t, v in w2.items()
            if __import__('etf_universe').ETF_BY_TICKER[t]['asset_class'] == 'Equity')
usa_eq = sum(v * etf_macro_vector(__import__('etf_universe').ETF_BY_TICKER[t]).get('USA',0)
             for t, v in w2.items()
             if __import__('etf_universe').ETF_BY_TICKER[t]['asset_class'] == 'Equity')
eur_eq = sum(v * etf_macro_vector(__import__('etf_universe').ETF_BY_TICKER[t]).get('Europa',0)
             for t, v in w2.items()
             if __import__('etf_universe').ETF_BY_TICKER[t]['asset_class'] == 'Equity')
print(f"  Parte azionaria: {eq_w2*100:.0f}% | USA su equity: {usa_eq/eq_w2*100:.0f}% (target 35) | Europa su equity: {eur_eq/eq_w2*100:.0f}% (target 45)")

print("\n=== 3) GEO: Solo Sviluppati (Emergenti 0%) ===")
geo3 = {"USA": 0.65, "Europa": 0.15, "Pacifico": 0.15, "Emergenti": 0.0, "Altri Sviluppati": 0.05}
w3 = max_sharpe_weights(returns, rf, equity_target=0.75, geo_target=geo3)
eme = aggregate_exposure(w3, "region").get("Emergenti", 0.0)
print(f"  Esposizione Emergenti nel portafoglio: {eme:.1f}% (deve essere ~0)")
assert eme < 5.0, "Troppa esposizione emergente!"
assert abs(eq_w - 0.30) < 0.10, "Equity custom non rispettato"
print("\nSELFTEST4_OK")
