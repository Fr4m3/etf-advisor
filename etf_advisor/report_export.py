"""
report_export.py
=============================================================================
Esportazione del Report in PDF e CSV.

- I grafici (donut geo/settoriali, Monte Carlo) vengono renderizzati in PNG
  con matplotlib (nessuna dipendenza da server esterni/kaleido).
- Il PDF è costruito con fpdf2, usando un font Unicode (DejaVuSans, bundled
  con matplotlib) per supportare l'EUR (€) e le accentate italiane.
- Il CSV è generato direttamente da pandas.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF


# ---------------------------------------------------------------------------
# Font Unicode (DejaVuSans da matplotlib) per supportare € e accenti
# ---------------------------------------------------------------------------
def find_unicode_fonts() -> tuple:
    """Ritorna (percorso_regolare, percorso_grassetto) del font DejaVuSans.

    DejaVuSans supporta € e le accentate italiane. Se non disponibile,
    ritorna (None, None) e l'app usera' i font core (Helvetica).
    """
    try:
        import matplotlib.font_manager as fm
        reg = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        bold = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
        reg = reg if reg and os.path.exists(reg) else None
        bold = bold if bold and os.path.exists(bold) else None
        return reg, bold
    except Exception:
        return None, None


def find_unicode_font() -> str | None:
    """Wrapper retrocompatibile: ritorna solo il font regolare."""
    return find_unicode_fonts()[0]


def _sanitize(s: str) -> str:
    """Sostituisce i caratteri non latin-1 (usato se il font Unicode non c'è)."""
    if not isinstance(s, str):
        return s
    repl = {"€": "EUR", "→": "->", "•": "-", "’": "'", "‘": "'",
            "“": '"', "”": '"', "−": "-"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


# ---------------------------------------------------------------------------
# Generazione grafici PNG (matplotlib)
# ---------------------------------------------------------------------------
def _donut_png(labels, values, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    wedges, _ = ax.pie(
        values, startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white"),
        colors=plt.cm.Set2.colors,
    )
    ax.legend(
        wedges,
        [f"{l}  {v:.1f}%" for l, v in zip(labels, values)],
        loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=7, frameon=False,
    )
    ax.set_title(title, fontsize=10)
    ax.axis("equal")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _mc_png(t, p10, p50, p90, capital, title: str = "Proiezione Monte Carlo") -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(t, p10, p90, color="green", alpha=0.15, label="10°-90°")
    ax.plot(t, p50, color="#2563eb", lw=2, label="Atteso (50°)")
    ax.plot(t, p90, color="green", lw=1, ls=":", label="Ottimistico (90°)")
    ax.plot(t, p10, color="red", lw=1, ls=":", label="Pessimistico (10°)")
    ax.axhline(capital, color="black", ls="--", lw=1, label="Capitale iniziale")
    ax.set_xlabel("Anni")
    ax.set_ylabel("Valore (EUR)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Costruzione PDF
# ---------------------------------------------------------------------------
def build_pdf(profile_name, capital, rf_pct, metrics, weights_df, scenario_df,
              geo, sec, mc, font_path=None, font_bold=None) -> bytes:
    """Ritorna il PDF del report come bytes."""
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    if font_path and os.path.exists(font_path):
        pdf.add_font("U", "", font_path)
        fam = "U"
        if font_bold and os.path.exists(font_bold):
            pdf.add_font("U", "B", font_bold)   # necessario per le intestazioni
    else:
        fam = "Helvetica"

    def T(s):
        return s if fam != "Helvetica" else _sanitize(str(s))

    pdf.set_font(fam, "", 11)
    pdf.add_page()

    pdf.cell(0, 9, T("ETF Advisor - Report di Portafoglio"), ln=1)
    pdf.set_font_size(9)
    pdf.cell(0, 6, T(f"Profilo: {profile_name}"), ln=1)
    pdf.cell(0, 6, T(f"Capitale iniziale: {capital:,.0f} EUR   |   "
                     f"Tasso risk-free: {rf_pct:.2f}%"), ln=1)
    pdf.ln(2)

    # Indicatori
    pdf.set_font_size(10)
    pdf.cell(0, 6, T("Indicatori di rischio e rendimento"), ln=1)
    lines = [
        f"Rendimento atteso annuo: {metrics['ann_ret']*100:.2f}%",
        f"Volatilita annualizzata: {metrics['ann_vol']*100:.2f}%",
        f"Sharpe Ratio: {metrics['sharpe']:.2f}",
        f"Max Drawdown storico: {metrics['mdd']*100:.1f}%",
        f"TER medio di portafoglio: {metrics['ter']:.3f}%",
        f"Costo stimato: {metrics['cost']:,.2f} EUR/anno",
    ]
    for l in lines:
        pdf.cell(0, 5.5, T(l), ln=1)
    pdf.ln(2)

    # Grafici: due donut affiancati + Monte Carlo
    geo_png = _donut_png(list(geo.keys()), list(geo.values()), "Esposizione Geografica")
    sec_png = _donut_png(list(sec.keys()), list(sec.values()), "Esposizione Settoriale")
    mc_png = _mc_png(mc["t"], mc["p10"], mc["p50"], mc["p90"], mc["capital"])

    y0 = pdf.get_y()
    pdf.image(io.BytesIO(geo_png), x=12, y=y0, w=86)
    pdf.image(io.BytesIO(sec_png), x=108, y=y0, w=86)
    pdf.set_y(y0 + 72)
    pdf.image(io.BytesIO(mc_png), x=12, w=180)
    pdf.ln(3)

    # Tabelle
    pdf.set_font_size(10)
    pdf.cell(0, 6, T("Composizione del portafoglio"), ln=1)
    _table(pdf, weights_df, fam, T)
    pdf.ln(2)
    pdf.cell(0, 6, T("Scenari Monte Carlo"), ln=1)
    _table(pdf, scenario_df, fam, T)
    pdf.ln(3)

    pdf.set_font_size(7)
    pdf.multi_cell(0, 4, T("Avvertenza: documento educativo/dimostrativo. Non "
                           "costituisce consulenza finanziaria. Verificare sempre "
                           "KID/KIID ufficiali prima di investire."))

    data = pdf.output(dest="S")
    if isinstance(data, str):
        data = data.encode("latin-1")
    return bytes(data)


def _table(pdf, df: pd.DataFrame, fam, T):
    headers = [T(str(c)) for c in df.columns]
    rows = df.astype(str).values.tolist()
    pdf.set_font_size(8)
    with pdf.table() as table:
        hrow = table.row()
        for c in headers:
            hrow.cell(c)
        for r in rows:
            rrow = table.row()
            for cell in r:
                rrow.cell(T(str(cell)))
    pdf.set_font_size(10)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
