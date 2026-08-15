"""Test export PDF/CSV + refresh metadati (fetch_justetf mockato, no rete)."""
import pandas as pd

import data_fetcher
from etf_universe import ETF_UNIVERSE, ETF_BY_TICKER
import report_export


def fake_justetf(isin, timeout=15):
    # Simula una risposta "live" con valori diversi dal riferimento
    return {
        "isin": isin, "name": "MOCK", "ter": 0.11, "fund_size_m": 12345, "ok": True,
    }


print("=== REFRESH METADATI (mock) ===")
orig = data_fetcher.fetch_justetf
data_fetcher.fetch_justetf = fake_justetf
before = {e["ticker"]: (e["ter"], e["fund_size_m"]) for e in ETF_UNIVERSE}
status = data_fetcher.refresh_metadata()
data_fetcher.fetch_justetf = orig
after = {e["ticker"]: (e["ter"], e["fund_size_m"]) for e in ETF_UNIVERSE}
changed = [t for t in before if before[t] != after[t]]
print(f"  ETF aggiornati: {len(changed)}/{len(before)}  es. {changed[:3]}")
print(f"  ETF_BY_TICKER coerente: {ETF_BY_TICKER['VWCE']['ter'] == after['VWCE'][0]}")
assert changed, "Nessun ETF aggiornato!"
assert ETF_BY_TICKER['VWCE']['ter'] == after['VWCE'][0]

print("\n=== EXPORT PDF/CSV ===")
metrics = {"ann_ret": 0.07, "ann_vol": 0.055, "sharpe": 0.83,
           "mdd": -0.073, "ter": 0.106, "cost": 10.6}
weights_df = pd.DataFrame([
    {"Ticker": "CSPX", "ISIN": "IE00B5BMR087", "Nome": "iShares Core S&P 500",
     "Classe": "Equity", "Peso (%)": 27.18, "TER (%)": 0.07, "Costo €/anno": 2.85},
    {"Ticker": "IEGA", "ISIN": "IE00B4WXJD03", "Nome": "iShares Euro Govt Bond",
     "Classe": "Bond", "Peso (%)": 40.72, "TER (%)": 0.09, "Costo €/anno": 0.0},
])
scenario_df = pd.DataFrame([
    {"Orizzonte": "3 anni", "Atteso (50°)": "12.277 €", "Ottimistico (90°)": "13.963 €"},
    {"Orizzonte": "15 anni", "Atteso (50°)": "28.241 €", "Ottimistico (90°)": "37.089 €"},
])
geo = {"USA": 38.6, "Europa": 5.1, "Obbligazionario Eurozona": 45.0}
sec = {"Tecnologia": 12.0, "Finanziari": 7.0, "Obbligazioni": 46.0}
mc = {
    "t": list(range(1, 16)),
    "p10": [10000 * (1.02 ** i) for i in range(15)],
    "p50": [10000 * (1.07 ** i) for i in range(15)],
    "p90": [10000 * (1.12 ** i) for i in range(15)],
    "capital": 10000,
}

csv = report_export.df_to_csv_bytes(weights_df)
print(f"  CSV bytes: {len(csv)}  inizia con: {csv[:18]!r}")

pdf = report_export.build_pdf("Bilanciato", 10000, 2.5, metrics,
                              weights_df, scenario_df, geo, sec, mc,
                              font_path=report_export.find_unicode_fonts()[0],
                              font_bold=report_export.find_unicode_fonts()[1])
print(f"  PDF bytes: {len(pdf)}  header: {pdf[:5]!r}")
assert pdf[:4] == b"%PDF", "PDF non valido!"
with open("sample_report.pdf", "wb") as f:
    f.write(pdf)
print("  PDF salvato in sample_report.pdf")

print("\nSELFTEST3_OK")
