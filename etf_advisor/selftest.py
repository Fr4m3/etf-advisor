"""Test end-to-end della pipeline finanziaria (dati sintetici, no rete)."""
import numpy as np
import pandas as pd

import data_fetcher
from finance import (daily_returns, portfolio_annual_metrics,
                     portfolio_price_series, max_drawdown, weighted_ter,
                     weighted_cost_eur, aggregate_exposure)
from optimizer import max_sharpe_weights
import montecarlo
from questionario import run_questionnaire
from etf_universe import ETF_UNIVERSE

# 1) Profilo (tutte risposte "3" -> Aggressivo)
answers = {"loss": 3, "horizon": 3, "knowledge": 3, "saving": 3}
profile = run_questionnaire(answers)
print("PROFILO:", profile.name, "equity_target=", profile.equity_target)

# 2) Prezzi sintetici per tutti i ticker
tickers = [e["ticker"] for e in ETF_UNIVERSE]
prices = data_fetcher.fetch_prices(tickers, period="5y", use_live=False)
print("Prezzi shape:", prices.shape, "| colonne:", list(prices.columns))

returns = daily_returns(prices).dropna()
print("Returns shape:", returns.shape)

# 3) Ottimizzatore Max Sharpe con target equity 0.5 (profilo bilanciato di test)
weights = max_sharpe_weights(returns, rf=0.025, equity_target=0.5)
print("\nPESI OTTIMI (Max Sharpe, target eq=0.5):")
for t, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {t}: {w*100:.2f}%")

# 4) Metriche
ann_ret, ann_vol, sharpe = portfolio_annual_metrics(returns, weights, 0.025)
pseries = portfolio_price_series(prices, weights)
mdd = max_drawdown(pseries)
wter = weighted_ter(weights)
cost = weighted_cost_eur(weights, 10000)
print(f"\nRend atteso: {ann_ret*100:.2f}% | Vol: {ann_vol*100:.2f}% | "
      f"Sharpe: {sharpe:.2f} | MaxDD: {mdd*100:.1f}%")
print(f"TER medio: {wter:.3f}% | Costo €/anno (10k): {cost:.2f} €")

# 5) Esposizioni
geo = aggregate_exposure(weights, "region")
sec = aggregate_exposure(weights, "sectors")
print("\nGEO:", geo)
print("SETTORI:", sec)

# 6) Monte Carlo
mc = montecarlo.monte_carlo(mu=ann_ret, sigma=ann_vol, capital=10000,
                            horizons=(3, 5, 10, 15), n_paths=2000)
print("\nMONTE CARLO (terminali):")
for h in (3, 5, 10, 15):
    s = mc["terminal"][h]
    print(f"  {h}a -> 10°:{s['p10']:,.0f}  50°:{s['p50']:,.0f}  90°:{s['p90']:,.0f}")

print("\nSELFTEST_OK")
