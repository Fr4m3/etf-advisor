"""Test ribilanciamento + Monte Carlo a livello di asset + confronto profili."""
import numpy as np
import pandas as pd

import data_fetcher
from finance import (daily_returns, portfolio_price_series, max_drawdown,
                     weighted_ter, asset_annual_moments)
from optimizer import max_sharpe_weights
import montecarlo
from questionario import map_profile
from etf_universe import ETF_UNIVERSE, ETF_BY_TICKER

tickers = [e["ticker"] for e in ETF_UNIVERSE]
prices = data_fetcher.fetch_prices(tickers, period="5y", use_live=False)
returns = daily_returns(prices).dropna()
rf = 0.025

print("=== RIBILANCIAMENTO (backtest storico, stesso portafoglio) ===")
w = max_sharpe_weights(returns, rf, 0.5)
for rb in ["none", "annual", "quarterly", "monthly"]:
    ps = portfolio_price_series(prices, w, rb)
    ret = ps.pct_change().dropna()
    ar = ret.mean()*252; av = ret.std()*np.sqrt(252)
    mdd = max_drawdown(ps)
    print(f"  {rb:9s} -> Rend {ar*100:6.2f}%  Vol {av*100:5.2f}%  "
          f"Sharpe {(ar-rf)/av:.2f}  MaxDD {mdd*100:6.1f}%")

print("\n=== MONTE CARLO ASSET-LEVEL con ribilanciamento ===")
order = list(w.keys())
am, ac = asset_annual_moments(returns, order)
for rb in ["none", "annual", "monthly"]:
    mc = montecarlo.monte_carlo_assets(am, ac, w, 10000, rb, (3, 5, 10, 15), n_paths=1500)
    s15 = mc["terminal"][15]
    print(f"  {rb:9s} -> 15a  10°:{s15['p10']:,.0f}  50°:{s15['p50']:,.0f}  "
          f"90°:{s15['p90']:,.0f}  | length p50={len(mc['p50'])}")

print("\n=== CONFRONTO PROFILI (stesso universo, capitale, ribilanciamento) ===")
eq = sum(1 for t in tickers if ETF_BY_TICKER[t]["asset_class"] == "Equity")
bd = len(tickers) - eq
for name, score in [("Conservativo", 5), ("Bilanciato", 8),
                    ("Crescita", 10), ("Aggressivo", 12)]:
    p = map_profile(score)
    tgt = p.equity_target if (eq > 0 and bd > 0) else None
    ww = max_sharpe_weights(returns, rf, tgt)
    oi = list(ww.keys())
    a2, c2 = asset_annual_moments(returns, oi)
    mc2 = montecarlo.monte_carlo_assets(a2, c2, ww, 10000, "annual", (10, 15), n_paths=1500)
    print(f"  {name:12s} eq={int(p.equity_target*100):3d}%  nETF={len(ww):2d}  "
          f"TER={weighted_ter(ww):.3f}%  Med15={mc2['terminal'][15]['p50']:,.0f}€")

print("\nSELFTEST2_OK")
