"""
app.py
=============================================================================
Applicazione web "ETF Advisor" — profilazione investitore + raccomandazione
e ottimizzazione di un portafoglio di ETF UCITS.

Esegui con:
    streamlit run app.py

Struttura a 3 pagine (gestite da uno switch nella sidebar):
  Pagina 1 : Questionario Comportamentale (wizard step-by-step)
  Pagina 2 : ETF Consigliati & Selezione UCITS (scoring + ottimizzatore)
  Pagina 3 : Report Tecnico & Dashboard Analitica (grafici interattivi)

Nessun dato inserito viene salvato: tutto vive in st.session_state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from etf_universe import ETF_UNIVERSE, ETF_BY_TICKER
from questionario import QUESTIONS, run_questionnaire, compute_score, map_profile
import data_fetcher
from finance import (
    daily_returns, portfolio_annual_metrics, portfolio_price_series,
    max_drawdown, weighted_ter, weighted_cost_eur, aggregate_exposure,
    asset_annual_moments,
)
from optimizer import max_sharpe_weights
import montecarlo
import report_export
from report_export import find_unicode_fonts

# Font Unicode per il PDF (€ e accenti); None se non disponibile
FONT_REG, FONT_BOLD = find_unicode_fonts()

# ---------------------------------------------------------------------------
# Configurazione pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ETF Advisor — Profilazione & Portafoglio UCITS",
    page_icon="📊",
    layout="wide",
)

# Stile minimale
st.markdown("""
<style>
.big-title {font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem;}
.sub {color: #6b7280; margin-bottom: 1rem;}
.card {background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px;
       padding:1rem 1.2rem; margin-bottom:1rem;}
.metric-good {color:#16a34a; font-weight:700;}
.disclaimer {font-size:0.78rem; color:#9ca3af; margin-top:2rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stato di sessione (persiste tra le pagine)
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "questionario"
if "q_answers" not in st.session_state:
    st.session_state.q_answers = {q["id"]: None for q in QUESTIONS}
if "q_step" not in st.session_state:
    st.session_state.q_step = 0
if "profile" not in st.session_state:
    st.session_state.profile = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = None  # dict con prices/returns/weights


# ---------------------------------------------------------------------------
# Sidebar: parametri globali + navigazione
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 ETF Advisor")
    st.markdown("Profilazione rischio → portafoglio UCITS ottimizzato.")

    page = st.radio(
        "Navigazione",
        ["1 · Questionario", "2 · ETF Consigliati", "3 · Report & Dashboard",
         "4 · Il mio portafoglio"],
        index=["questionario", "etf", "report", "myportfolio"].index(st.session_state.page),
    )
    st.session_state.page = {
        "1 · Questionario": "questionario",
        "2 · ETF Consigliati": "etf",
        "3 · Report & Dashboard": "report",
        "4 · Il mio portafoglio": "myportfolio",
    }[page]

    st.divider()
    capital = st.number_input("Capitale iniziale (€)", 100, 10_000_000, 10_000, 500)
    rf_pct = st.number_input("Tasso risk-free (BCE/Euribor) %", 0.0, 8.0, 2.5, 0.25)
    rf = rf_pct / 100.0
    rebalance_label = st.selectbox(
        "Politica di ribilanciamento",
        ["Nessuna (buy & hold)", "Annuale", "Trimestrale", "Mensile"], index=1,
    )
    st.session_state.rebalance = {
        "Nessuna (buy & hold)": "none",
        "Annuale": "annual", "Trimestrale": "quarterly", "Mensile": "monthly",
    }[rebalance_label]
    st.caption("Il tasso risk-free è usato per lo Sharpe Ratio.")

# ---------------------------------------------------------------------------
# Cache dati di mercato
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Scarico i prezzi di mercato (Yahoo Finance)…")
def _cached_prices(tickers_tuple, period, use_live):
    return data_fetcher.fetch_prices(list(tickers_tuple), period=period, use_live=use_live)


# ===========================================================================
# PAGINA 1 — QUESTIONARIO COMPORTAMENTALE
# ===========================================================================
def page_questionnaire():
    st.markdown('<div class="big-title">1 · Questionario di Profilazione</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub">Rispondi con tranquillità: non esistono risposte '
                'giuste o sbagliate, solo la tua situazione reale.</div>',
                unsafe_allow_html=True)

    questions = QUESTIONS
    step = st.session_state.q_step
    q = questions[step]

    # Progress bar
    st.progress((step + 1) / len(questions))
    st.markdown(f"**Sezione {step + 1}/{len(questions)} — {q['section']}**")

    st.markdown(f"### {q['question']}")
    opts = [o["label"] for o in q["options"]]
    current = st.session_state.q_answers[q["id"]]
    # default_idx=0 se nessuna risposta ancora selezionata (evita index=None
    # che su Streamlit recenti restituisce None e fa crashare opts.index)
    default_idx = next((i for i, o in enumerate(q["options"])
                        if o["score"] == current), 0)

    choice = st.radio(
        "Scegli la risposta che ti rappresenta meglio:",
        opts, index=default_idx, key=f"radio_{q['id']}",
    )
    if choice is None:          # sicurezza aggiuntiva su versioni diverse
        choice = opts[0]
    # Salva il punteggio della risposta scelta
    chosen = q["options"][opts.index(choice)]
    st.session_state.q_answers[q["id"]] = chosen["score"]

    col_back, col_next, col_spacer = st.columns([1, 1, 3])
    with col_back:
        if st.button("← Indietro", disabled=(step == 0)):
            st.session_state.q_step = max(0, step - 1)
            st.rerun()
    with col_next:
        if step < len(questions) - 1:
            if st.button("Avanti →"):
                st.session_state.q_step = min(len(questions) - 1, step + 1)
                st.rerun()
        else:
            if st.button("✅ Calcola il mio profilo", type="primary"):
                score = compute_score(st.session_state.q_answers)
                st.session_state.profile = run_questionnaire(st.session_state.q_answers)
                # Reset eventuale portafoglio precedente
                st.session_state.portfolio = None
                st.session_state.page = "etf"
                st.success(f"Punteggio totale: {score}/12 → Profilo: "
                           f"**{st.session_state.profile.name}**")
                st.rerun()

    # Anteprima risposte date
    with st.expander("Riepilogo risposte"):
        for qq in questions:
            sc = st.session_state.q_answers[qq["id"]]
            st.write(f"- **{qq['section']}**: "
                     f"{'non risposto' if sc is None else 'punteggio ' + str(sc)}")

    if st.session_state.profile is None:
        st.markdown('<div class="disclaimer">Strumento educativo. Non costituisce '
                    'consulenza finanziaria.</div>', unsafe_allow_html=True)
        return

    # Se il profilo è già stato calcolato, mostra il riepilogo
    show_profile_card(st.session_state.profile)
    if st.button("➡️ Vai agli ETF consigliati"):
        st.session_state.page = "etf"
        st.rerun()


def show_profile_card(profile):
    st.markdown("---")
    st.markdown(f"### Il tuo profilo: **{profile.name}**")
    st.markdown(f"<div class='card'>{profile.description}</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Asset Allocation Target",
                  f"{int(profile.equity_target*100)}% Azionario / "
                  f"{int(profile.bond_target*100)}% Obbligazionario")
    with c2:
        st.write("**Cosa significa nella pratica:**")
        st.write(f"Su 10.000 € investiti, circa "
                 f"{int(profile.equity_target*100)}€ finirebbero in azioni "
                 f"(crescita) e {int(profile.bond_target*100)}€ in obbligazioni "
                 f"(stabilità).")


PROFILE_SCORES = {"Conservativo": 5, "Bilanciato": 8, "Crescita": 10, "Aggressivo": 12}


def profile_by_name(name: str):
    """Restituisce l'oggetto Profile a partire dal nome."""
    return map_profile(PROFILE_SCORES[name])


# ===========================================================================
# PAGINA 2 — ETF CONSIGLIATI & SELEZIONE UCITS
# ===========================================================================
def page_etf():
    st.markdown('<div class="big-title">2 · ETF Consigliati & Selezione UCITS</div>',
                unsafe_allow_html=True)
    if st.session_state.profile is None:
        st.warning("Completa prima il questionario (Pagina 1).")
        return

    profile = st.session_state.profile
    show_profile_card(profile)

    st.markdown("#### Filtri rigidi UCITS & Liquidità")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        acc_mode = st.radio(
            "Tipo di accumulo",
            ["Tutti", "Solo Accumulo", "Solo Distribuzione"],
            index=0, horizontal=True,
            help="Accumulo (Acc): i dividendi sono reinvestiti. "
                 "Distribuzione (Dist): i dividendi sono pagati in conto.",
        )
    with f_col2:
        min_size = st.slider("Dimensione fondo minima (M€)", 0, 500, 100, 50)
    with f_col3:
        max_ter = st.slider("TER massimo (%)", 0.05, 0.50, 0.30, 0.01)

    period = st.selectbox("Storico per ottimizzazione", ["3y", "5y", "10y"], index=1)
    use_live = st.checkbox("Usa dati di mercato reali (Yahoo Finance)", value=True)

    # Categorie (settoriali / real estate / materie prime / broad / bond)
    all_cats = sorted({e["category"] for e in ETF_UNIVERSE})
    cats = st.multiselect("Categorie ETF da includere", all_cats, default=all_cats)

    # ---- Metadati live da JustETF ----
    st.markdown("#### 🔄 Metadati live (JustETF)")
    st.caption("Aggiorna TER e dimensione fondo dagli ISIN ufficiali. In caso di "
               "blocco del sito, i valori di riferimento restano invariati.")
    if st.button("🔄 Aggiorna TER / dimensioni da JustETF"):
        with st.spinner("Recupero metadati da JustETF…"):
            status = data_fetcher.refresh_metadata()
        st.session_state.meta_status = status
        st.session_state.meta_refreshed = True
    if st.session_state.get("meta_status"):
        st.success("Metadati aggiornati (i valori non recuperati mantengono "
                   "il riferimento).")
        with st.expander("Dettaglio aggiornamento"):
            mrows = [{
                "Ticker": s["ticker"], "TER prec.": s["ter_old"],
                "TER nuovo": s["ter_new"], "Dim. prec. (M€)": s["size_old"],
                "Dim. nuova (M€)": s["size_new"], "Stato": s["msg"],
            } for s in st.session_state.meta_status]
            st.dataframe(pd.DataFrame(mrows), use_container_width=True, hide_index=True)

    # Applica filtri all'universo (tutti gli ETF sono già UCITS per costruzione)
    def _acc_ok(e):
        if acc_mode == "Tutti":
            return True
        if acc_mode == "Solo Accumulo":
            return bool(e["accumulation"])
        return not e["accumulation"]

    filtered = [
        e for e in ETF_UNIVERSE
        if _acc_ok(e)
        and e["fund_size_m"] >= min_size
        and e["ter"] <= max_ter
        and e["category"] in cats
    ]
    # (Tutti gli ETF dell'universo sono UCITS per costruzione)
    st.caption("✅ Filtro UCITS attivo: tutti gli ETF dell'universo sono conformi "
               "al regolamento UE (tassazione/regole europee).")

    if not filtered:
        st.error("Nessun ETF soddisfa i filtri. Allenta i vincoli.")
        return

    filtered_tickers = [e["ticker"] for e in filtered]

    # Tabella candidati
    st.markdown("#### Universo selezionabile (dopo i filtri)")
    tbl = pd.DataFrame([{
        "Ticker": e["ticker"], "ISIN": e["isin"], "Nome": e["name"],
        "Classe": e["asset_class"], "Categoria": e["category"],
        "TER (%)": e["ter"],
        "Dim. (M€)": e["fund_size_m"], "Accumulo": "Sì" if e["accumulation"] else "No",
    } for e in filtered])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ---- Allocazione personalizzata & ottimizzazione geografica ----
    st.markdown("#### ⚖️ Allocazione e Ottimizzazione")
    st.markdown("**Allocazione per classe: Azionario / Obbligazionario / Materie Prime**")
    eq_def = int(profile.equity_target * 100)
    comm_def = 5
    eq_alloc = st.slider("📈 Azionario (%)", 0, 100, eq_def, 5, key="alloc_eq")
    comm_max = max(0, 100 - eq_alloc)
    comm_val = min(st.session_state.get("alloc_comm", comm_def), comm_max)
    comm_alloc = st.slider("🪙 Materie Prime (%)", 0, comm_max, comm_val, 1,
                           key="alloc_comm")
    bond_alloc = 100 - eq_alloc - comm_alloc
    st.write(f"🏦 Obbligazionario (derivato): **{bond_alloc}%**")
    eq_target = eq_alloc / 100.0
    class_targets = {"Equity": eq_target,
                     "Bond": bond_alloc / 100.0,
                     "Alternatives": comm_alloc / 100.0}

    st.markdown("**🌍 Ottimizzazione geografica (sulla parte azionaria)**")
    geo_mode = st.radio(
        "Strategia geografica",
        ["Disattivata", "Market-cap globale", "Home bias Europa",
         "Solo Sviluppati", "Personalizzata"],
        horizontal=True,
    )
    geo_presets = {
        "Market-cap globale": {"USA": 0.60, "Europa": 0.15, "Pacifico": 0.10,
                               "Emergenti": 0.12, "Altri Sviluppati": 0.03},
        "Home bias Europa": {"USA": 0.35, "Europa": 0.45, "Pacifico": 0.07,
                             "Emergenti": 0.10, "Altri Sviluppati": 0.03},
        "Solo Sviluppati": {"USA": 0.65, "Europa": 0.15, "Pacifico": 0.15,
                            "Emergenti": 0.0, "Altri Sviluppati": 0.05},
    }
    geo_target = None
    if geo_mode == "Personalizzata":
        g1, g2, g3, g4 = st.columns(4)
        usa = g1.slider("USA %", 0, 100, 60, 5)
        eur = g2.slider("Europa %", 0, 100, 15, 5)
        eme = g3.slider("Emergenti %", 0, 100, 12, 5)
        pac = g4.slider("Pacifico %", 0, 100, 10, 5)
        s = usa + eur + eme + pac
        if s > 0:
            geo_target = {"USA": usa / s, "Europa": eur / s,
                          "Emergenti": eme / s, "Pacifico": pac / s}
        st.caption(f"Somma slider: {s}% (normalizzata automaticamente)")
    elif geo_mode in geo_presets:
        geo_target = geo_presets[geo_mode]

    if st.button("🚀 Calcola portafoglio ottimale (Max Sharpe)", type="primary"):
        with st.spinner("Ottimizzazione in corso…"):
            prices = _cached_prices(tuple(filtered_tickers), period, use_live)
            returns = daily_returns(prices).dropna()

            eq_count = sum(1 for t in filtered_tickers
                           if ETF_BY_TICKER[t]["asset_class"] == "Equity")
            bd_count = len(filtered_tickers) - eq_count
            alt_count = sum(1 for t in filtered_tickers
                            if ETF_BY_TICKER[t]["asset_class"] == "Alternatives")

            if class_targets.get("Alternatives", 0.0) > 1e-6 and alt_count == 0:
                st.warning("Hai richiesto una quota di **Materie Prime**, ma nessun "
                           "ETF 'Materie Prime' è nei filtri selezionati. Aggiungi la "
                           "categoria 'Materie Prime' (sopra) per includerla; il target "
                           "verrà riassorbito in azionario/obbligazionario.")

            weights = max_sharpe_weights(
                returns, rf, geo_target=geo_target, class_targets=class_targets)

        st.session_state.portfolio = {
            "prices": prices, "returns": returns, "weights": weights,
            "profile": profile, "period": period, "use_live": use_live,
            "filtered_tickers": filtered_tickers,
            "equity_target": eq_target, "geo_mode": geo_mode,
            "class_targets": class_targets,
        }
        st.success(f"Portafoglio calcolato con {len(weights)} ETF. "
                   f"Vai alla Pagina 3 per il report.")
        _show_weights_table(weights, capital)

    elif st.session_state.portfolio is not None:
        st.info("Portafoglio già calcolato. Ricalcola per applicare i nuovi filtri "
                "oppure vai al Report.")
        _show_weights_table(st.session_state.portfolio["weights"], capital)


def _show_weights_table(weights, capital):
    st.markdown("#### 📌 Portafoglio raccomandato (Max Sharpe)")
    rows = []
    for t, w in sorted(weights.items(), key=lambda x: -x[1]):
        e = ETF_BY_TICKER[t]
        rows.append({
            "Ticker": t, "ISIN": e["isin"], "Nome": e["name"],
            "Classe": e["asset_class"], "Peso (%)": round(w * 100, 2),
            "TER (%)": e["ter"],
            "Costo €/anno": round(w * capital * e["ter"] / 100.0, 2),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    return df

    # CTA verso report
    if st.button("➡️ Apri il Report & Dashboard"):
        st.session_state.page = "report"
        st.rerun()


# ===========================================================================
# PAGINA 3 — REPORT & DASHBOARD
# ===========================================================================
def page_report():
    st.markdown('<div class="big-title">3 · Report Tecnico & Dashboard</div>',
                unsafe_allow_html=True)
    port = st.session_state.portfolio
    if port is None:
        st.warning("Calcola prima un portafoglio alla Pagina 2.")
        return

    prices = port["prices"]
    returns = port["returns"]
    weights = port["weights"]
    profile = port["profile"]

    st.markdown(f"**Profilo:** {profile.name} · "
                f"**Capitale:** {capital:,.0f} € · "
                f"**Risk-free:** {rf_pct:.2f}%")
    geo_used = port.get('geo_mode', 'Disattivata')
    ct = port.get("class_targets")
    if ct:
        eq_u = int(round(ct.get("Equity", 0) * 100))
        bd_u = int(round(ct.get("Bond", 0) * 100))
        al_u = int(round(ct.get("Alternatives", 0) * 100))
        alloc_txt = (f"**{eq_u}% Azionario / {bd_u}% Obbligazionario / "
                     f"{al_u}% Materie Prime**")
    else:
        eq_used = port.get('equity_target', profile.equity_target)
        alloc_txt = (f"**{int(eq_used*100)}% Azionario / "
                     f"{int((1-eq_used)*100)}% Obbligazionario**")
    st.caption(f'Allocazione applicata: {alloc_txt} · '
               f'Ottimizzazione geografica: **{geo_used}**')

    # ---- Metriche di portafoglio (coerenti con la politica di ribilanciamento) ----
    rb = st.session_state.get("rebalance", "none")
    ann_ret, ann_vol, sharpe = portfolio_annual_metrics(returns, weights, rf)
    pseries = portfolio_price_series(prices, weights, rb)
    ps_ret = pseries.pct_change().dropna()
    ann_ret = float(ps_ret.mean() * 252)
    ann_vol = float(ps_ret.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    mdd = max_drawdown(pseries)
    wter = weighted_ter(weights)
    cost_eur = weighted_cost_eur(weights, capital)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rendimento atteso (annuo)", f"{ann_ret*100:.2f}%")
    k2.metric("Volatilità annualizzata", f"{ann_vol*100:.2f}%")
    with k3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with k4:
        st.metric("Max Drawdown storico", f"{mdd*100:.1f}%",
                  help="Perdita massima dal picco al minimo sullo storico.")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("TER medio di portafoglio", f"{wter:.3f}%",
                  help="Costo totale annuo medio ponderato.")
    with c2:
        st.metric("Costo stimato (€/anno)", f"{cost_eur:,.2f} €",
                  help=f"Su capitale iniziale di {capital:,.0f} €.")

    st.divider()

    # ---- Allocazione geografica, settoriale e per classe ----
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("#### 🌍 Esposizione Geografica")
        geo = aggregate_exposure(weights, "region")
        if geo:
            fig_geo = px.pie(
                names=list(geo.keys()), values=list(geo.values()),
                hole=0.5, title="",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_geo.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_geo, use_container_width=True)
    with g2:
        st.markdown("#### 🏭 Esposizione Settoriale")
        sec = aggregate_exposure(weights, "sectors")
        if sec:
            fig_sec = px.pie(
                names=list(sec.keys()), values=list(sec.values()),
                hole=0.5, title="",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_sec.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_sec, use_container_width=True)
    with g3:
        st.markdown("#### ⚖️ Allocazione per Classe")
        cls_labels = {"Equity": "Azionario", "Bond": "Obbligazionario",
                      "Alternatives": "Materie Prime"}
        cls_w = {}
        for t, w in weights.items():
            c = ETF_BY_TICKER[t]["asset_class"]
            lbl = cls_labels.get(c, c)
            cls_w[lbl] = cls_w.get(lbl, 0.0) + w
        if cls_w:
            fig_cls = px.pie(
                names=list(cls_w.keys()), values=list(cls_w.values()),
                hole=0.5, title="",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig_cls.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_cls, use_container_width=True)

    st.divider()

    # ---- Simulazione Monte Carlo (a livello di asset, con ribilanciamento) ----
    st.markdown("#### 🔮 Proiezione Monte Carlo del Capitale Futuro")
    st.caption("Simulazione di 3.000 percorsi (Moto Browniano Geometrico) "
               "basati su rendimento atteso e volatilità storici del "
               "portafoglio, con politica di ribilanciamento selezionata.")

    order = list(weights.keys())
    asset_mu, asset_cov = asset_annual_moments(returns, order)
    mc = montecarlo.monte_carlo_assets(
        asset_mu=asset_mu, asset_cov=asset_cov, weights=weights,
        capital=capital, rebalance=rb, horizons=(3, 5, 10, 15),
        max_years=15, n_paths=3000,
    )

    t = mc["t"]
    fig_mc = go.Figure()
    # banda ottimistico/pessimistico
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([mc["p90"], mc["p10"][::-1]]),
        fill="toself", fillcolor="rgba(34,197,94,0.15)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name="Scenari 10°–90°",
    ))
    # percorsi di esempio
    for p in mc["sample_paths"][::4]:
        fig_mc.add_trace(go.Scatter(
            x=t, y=p, line=dict(color="gray", width=0.5),
            opacity=0.12, showlegend=False, hoverinfo="skip",
        ))
    fig_mc.add_trace(go.Scatter(x=t, y=mc["p50"], name="Atteso (50°)",
                                line=dict(color="#2563eb", width=3)))
    fig_mc.add_trace(go.Scatter(x=t, y=mc["p90"], name="Ottimistico (90°)",
                                line=dict(color="#16a34a", width=2, dash="dot")))
    fig_mc.add_trace(go.Scatter(x=t, y=mc["p10"], name="Pessimistico (10°)",
                                line=dict(color="#dc2626", width=2, dash="dot")))
    fig_mc.add_hline(y=capital, line=dict(color="black", width=1, dash="dash"),
                     annotation_text="Capitale iniziale")
    fig_mc.update_layout(
        xaxis_title="Anni", yaxis_title="Valore portafoglio (€)",
        height=460, legend=dict(orientation="h", y=-0.15),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Tabella scenari
    st.markdown("##### Risultati per orizzonte")
    scen_rows = []
    for h in (3, 5, 10, 15):
        s = mc["terminal"][h]
        scen_rows.append({
            "Orizzonte": f"{h} anni",
            "Pessimistico (10°)": f"{s['p10']:,.0f} €",
            "Atteso (50°)": f"{s['p50']:,.0f} €",
            "Ottimistico (90°)": f"{s['p90']:,.0f} €",
            "Rend. medio atteso": f"{(s['p50']/capital)**(1/h)-1:.1%} /anno",
        })
    scenario_df = pd.DataFrame(scen_rows)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Confronto tra profili ----
    st.markdown("#### ⚖️ Confronto tra profili di rischio")
    st.caption("Stesso universo filtrato, stesso capitale e stessa politica di "
               "ribilanciamento. Ogni profilo ottimizza il proprio portafoglio "
               "(Max Sharpe) col proprio target azionario.")
    cmp_opts = ["Conservativo", "Bilanciato", "Crescita", "Aggressivo"]
    default_cmp = list(dict.fromkeys([profile.name, "Conservativo", "Aggressivo"]))
    cmp_sel = st.multiselect("Profili da confrontare", cmp_opts, default=default_cmp)

    if cmp_sel:
        filtered_tickers = port.get("filtered_tickers", list(returns.columns))
        eq_count = sum(1 for t in filtered_tickers
                       if ETF_BY_TICKER[t]["asset_class"] == "Equity")
        bd_count = len(filtered_tickers) - eq_count
        fig_cmp = go.Figure()
        cmp_rows = []
        for name in cmp_sel:
            pf = profile_by_name(name)
            tgt = pf.equity_target if (eq_count > 0 and bd_count > 0) else None
            w = max_sharpe_weights(returns, rf, tgt)
            if not w:
                continue
            order_i = list(w.keys())
            am, ac = asset_annual_moments(returns, order_i)
            mc_i = montecarlo.monte_carlo_assets(
                am, ac, w, capital, rb, (3, 5, 10, 15))
            fig_cmp.add_trace(go.Scatter(
                x=mc_i["t"], y=mc_i["p50"],
                name=f"{name} (eq {int(pf.equity_target*100)}%)",
                line=dict(width=3)))
            ps_i = portfolio_price_series(prices, w, rb)
            psr = ps_i.pct_change().dropna()
            ar = psr.mean() * 252
            av = psr.std() * np.sqrt(252)
            sh = (ar - rf) / av if av > 0 else 0
            cmp_rows.append({
                "Profilo": name,
                "Target Azionario": f"{int(pf.equity_target*100)}%",
                "TER %": round(weighted_ter(w), 3),
                "Sharpe": round(sh, 2),
                "Vol %": round(av * 100, 1),
                "Mediana 10a": f"{mc_i['terminal'][10]['p50']:,.0f} €",
                "Mediana 15a": f"{mc_i['terminal'][15]['p50']:,.0f} €",
            })
        fig_cmp.add_hline(y=capital, line=dict(color="black", width=1, dash="dash"),
                         annotation_text="Capitale iniziale")
        fig_cmp.update_layout(
            xaxis_title="Anni", yaxis_title="Valore mediano (€)", height=460,
            legend=dict(orientation="h", y=-0.15), margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ---- Tabella composizione finale ----
    st.markdown("#### 📋 Composizione finale del portafoglio")
    weights_df = _show_weights_table(weights, capital)

    # ---- Esportazione ----
    st.markdown("#### 📤 Esporta il Report")
    ec1, ec2, ep = st.columns(3)
    with ec1:
        st.download_button(
            "⬇️ Composizione (CSV)",
            report_export.df_to_csv_bytes(weights_df), "portafoglio.csv", "text/csv")
    with ec2:
        st.download_button(
            "⬇️ Scenari (CSV)",
            report_export.df_to_csv_bytes(scenario_df), "scenari.csv", "text/csv")
    with ep:
        metrics = {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
                   "mdd": mdd, "ter": wter, "cost": cost_eur}
        pdf_bytes = report_export.build_pdf(
            profile.name, capital, rf_pct, metrics, weights_df, scenario_df,
            geo, sec, mc, font_path=FONT_REG, font_bold=FONT_BOLD)
        st.download_button(
            "⬇️ Report completo (PDF)", pdf_bytes, "report_etf_advisor.pdf",
            "application/pdf")

    st.markdown('<div class="disclaimer">⚠️ Dati e simulazioni a scopo '
                'educativo/dimostrativo. Non costituiscono consulenza '
                'finanziaria. Verifica SEMPRE TER, dimensione e documentazione '
                'ufficiale (KID/KIID) prima di investire.</div>',
                unsafe_allow_html=True)


# ===========================================================================
# PAGINA 4 — IL MIO PORTAFOGLIO (analisi holdings inseriti dall'utente)
# ===========================================================================
def _resolve_holding(code):
    """Restituisce (etf_dict, prices_or_None) per un codice (ticker o ISIN).

    Se il codice e' nell'universo usa i metadati noti; altrimenti prova a
    recuperare metadati (JustETF per ISIN) e/o prezzi (Yahoo per ticker).
    """
    code = str(code).strip().upper()
    if code in ETF_BY_TICKER:
        return ETF_BY_TICKER[code], None
    if code in ETF_BY_ISIN:
        return ETF_BY_ISIN[code], None
    # ETF personalizzato (non nell'universo)
    is_isin = bool(re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", code))
    meta = data_fetcher.fetch_justetf(code) if is_isin else None
    prices = None
    if not is_isin:
        try:
            p = data_fetcher.fetch_prices([code], "5y", True)
            if p is not None and not p.empty and p.shape[1] > 0:
                prices = p
        except Exception:
            prices = None
    region = (meta or {}).get("region") or {"Sconosciuto": 1.0}
    sectors = (meta or {}).get("sectors") or {"Altro": 1.0}
    etf = {
        "ticker": code, "name": (meta or {}).get("name") or code,
        "asset_class": "Equity", "category": "Personalizzato",
        "region": region, "sectors": sectors,
        "ter": (meta or {}).get("ter") or 0.0,
        "fund_size_m": (meta or {}).get("fund_size_m") or 0,
        "accumulation": True, "exp_return": 0.06, "vol": 0.15,
        "yf_ticker": code if prices is not None else None,
        "custom": True, "meta_ok": bool(meta and meta.get("ok")),
        "_prices": prices,
    }
    return etf, prices


def page_my_portfolio():
    st.markdown('<div class="big-title">4 · Il mio portafoglio</div>',
                unsafe_allow_html=True)
    st.markdown("Inserisci gli ETF che possiedi e l'importo investito: il sistema "
                "calcola esposizione **geografica** e **settoriale**, i **costi (TER)** "
                "e — se i prezzi sono disponibili — **Sharpe, volatilità e drawdown** "
                "del *tuo* portafoglio. Puoi usare gli ETF dell'universo oppure "
                "incollare un **ISIN** o un **ticker Yahoo** (es. `IE00BK5BQT80` o "
                "`VWCE.DE`): i metadati vengono recuperati automaticamente da "
                "JustETF / Yahoo Finance.")
    st.caption("Suggerimento: incolla un ISIN per ottenere esposizione geografica/"
               "settoriale e TER da JustETF; incolla un ticker yfinance (es. VWCE.DE) "
               "per ottenere anche le serie storiche dei prezzi.")

    if "portfolio_holdings" not in st.session_state:
        st.session_state["portfolio_holdings"] = pd.DataFrame(columns=["Codice", "Importo (€)"])

    def _append(code, amt):
        df = st.session_state["portfolio_holdings"]
        st.session_state["portfolio_holdings"] = pd.concat(
            [df, pd.DataFrame([{"Codice": code, "Importo (€)": amt}])],
            ignore_index=True)

    st.markdown("**Aggiungi rapido dall'universo:**")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q_sel = st.selectbox("ETF dell'universo",
                             [f"{e['ticker']} — {e['name']}" for e in ETF_UNIVERSE],
                             label_visibility="collapsed")
    with c2:
        q_amt = st.number_input("Importo (€)", 0, step=100, value=1000,
                                label_visibility="collapsed")
    with c3:
        if st.button("➕ Aggiungi"):
            _append(q_sel.split(" — ")[0], q_amt)

    st.markdown("**Oppure incolla un ISIN / ticker personalizzato:**")
    c4, c5, c6 = st.columns([2, 1, 1])
    with c4:
        cust = st.text_input("ISIN o ticker (es. IE00BK5BQT80 / VWCE.DE)",
                             label_visibility="collapsed")
    with c5:
        cust_amt = st.number_input("Importo pers. (€)", 0, step=100, value=1000,
                                   key="cust_amt", label_visibility="collapsed")
    with c6:
        if st.button("➕ Aggiungi personalizzato") and cust.strip():
            _append(cust.strip().upper(), cust_amt)

    edited = st.data_editor(
        st.session_state["portfolio_holdings"], num_rows="dynamic",
        column_config={
            "Codice": st.column_config.TextColumn("ISIN / Ticker"),
            "Importo (€)": st.column_config.NumberColumn("Importo investito (€)",
                                                         min_value=0, step=100),
        },
        use_container_width=True, key="portfolio_editor",
    )

    # Ribilanciamento del portafoglio personale
    _rb_map = {"Nessuno": "none", "Annuale": "annual",
               "Trimestrale": "quarterly", "Mensile": "monthly"}
    rebal_label = st.selectbox(
        "🔄 Politica di ribilanciamento",
        list(_rb_map.keys()), index=0,
        help="Ribilanciamento periodico del tuo portafoglio: riconduce i pesi "
             "al target a intervalli fissi.",
    )
    rebal_mode = _rb_map[rebal_label]

    if st.button("📊 Analizza il mio portafoglio", type="primary"):
        rows = [r for _, r in edited.iterrows()
                if str(r.get("Codice", "")).strip() and (r.get("Importo (€)") or 0) > 0]
        if not rows:
            st.warning("Aggiungi almeno un ETF con importo > 0.")
            return
        amt = {}
        for r in rows:
            cc = str(r["Codice"]).strip().upper()
            amt[cc] = amt.get(cc, 0.0) + float(r["Importo (€)"])
        total = sum(amt.values())
        weights = {c: v / total for c, v in amt.items()}

        # Risolvi metadati per ogni codice
        lookup = {c: _resolve_holding(c)[0] for c in weights}

        def _agg(w, key):
            out = {}
            for c, frac in w.items():
                for k, v in lookup[c].get(key, {}).items():
                    out[k] = out.get(k, 0.0) + frac * v
            return out
        geo = _agg(weights, "region")
        sec = _agg(weights, "sectors")
        wter = sum(weights[c] * lookup[c]["ter"] for c in weights)
        cost = sum(weights[c] * total * lookup[c]["ter"] / 100.0 for c in weights)

        # Prezzi: universo via _cached_prices, custom da fetch diretto
        price_cols = {}
        known = [c for c in weights if not lookup[c].get("custom")]
        if known:
            pr = _cached_prices(tuple(known), "5y", True)
            for c in known:
                yf = lookup[c]["yf_ticker"]
                if yf in pr.columns and pr[yf].notna().sum() > 20:
                    price_cols[c] = pr[yf]
        for c in weights:
            if c in price_cols:
                continue
            p = lookup[c].get("_prices")
            if p is not None and not p.empty and p.columns[0] in p and \
               p[p.columns[0]].notna().sum() > 20:
                price_cols[c] = p[p.columns[0]]

        have_prices = bool(price_cols) and \
            pd.DataFrame(price_cols).dropna().shape[0] > 20
        pmat = pd.DataFrame(price_cols).dropna() if price_cols else pd.DataFrame()
        if have_prices:
            rb = rebal_mode
            weights_p = {c: weights[c] for c in pmat.columns}
            pseries = portfolio_price_series(pmat, weights_p, rb)
            psr = pseries.pct_change().dropna()
            ann_ret = float(psr.mean() * 252)
            ann_vol = float(psr.std() * np.sqrt(252))
            sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
            mdd = max_drawdown(pseries)
            returns = daily_returns(pmat).dropna()
        else:
            ann_ret = ann_vol = sharpe = mdd = None
            returns = pd.DataFrame()

        st.markdown(f"**Capitale totale:** {total:,.0f} €")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TER medio", f"{wter:.3f}%")
        m2.metric("Costo stimato", f"{cost:,.2f} €/anno")
        if have_prices:
            m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            m4.metric("Max Drawdown", f"{mdd*100:.1f}%")
        else:
            m3.metric("Volatilità", "n.d.")
            m4.metric("Rend. atteso", "n.d.")
            st.caption("Metriche storiche non disponibili (prezzi non recuperati "
                       "o serie troppo breve). Sono comunque calcolate esposizioni e costi.")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 🌍 Esposizione Geografica")
            if geo:
                fig_geo = px.pie(names=list(geo.keys()), values=list(geo.values()),
                                 hole=0.5, color_discrete_sequence=px.colors.qualitative.Set2)
                fig_geo.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_geo, use_container_width=True)
        with g2:
            st.markdown("#### 🏭 Esposizione Settoriale")
            if sec:
                fig_sec = px.pie(names=list(sec.keys()), values=list(sec.values()),
                                 hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_sec.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_sec, use_container_width=True)

        st.markdown("#### 📋 Composizione inserita")
        rows_out = []
        for c, w in sorted(weights.items(), key=lambda x: -x[1]):
            e = lookup[c]
            rows_out.append({
                "Codice": c, "Nome": e["name"], "Importo (€)": round(w * total, 2),
                "Peso (%)": round(w * 100, 2), "TER (%)": e["ter"],
                "Costo €/anno": round(w * total * e["ter"] / 100.0, 2),
            })
        st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)

        # Note sugli ETF personalizzati
        custom_codes = [c for c in weights if lookup[c].get("custom")]
        if custom_codes:
            ok = [c for c in custom_codes if lookup[c].get("meta_ok")]
            no = [c for c in custom_codes if not lookup[c].get("meta_ok")]
            if ok:
                st.info("ETF personalizzati riconosciuti da JustETF: " + ", ".join(ok))
            if no:
                st.warning("ETF personalizzati non riconosciuti (esposizione/costi "
                           "indisponibili): " + ", ".join(no) +
                           ". Usa un ISIN valido per i metadati o un ticker yfinance per i prezzi.")

        if have_prices:
            st.markdown("#### 🔮 Proiezione Monte Carlo")
            order = list(pmat.columns)
            am, ac = asset_annual_moments(returns, order)
            mc = montecarlo.monte_carlo_assets(
                am, ac, weights_p, total, rb, (3, 5, 10, 15), n_paths=3000)
            t = mc["t"]
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([mc["p90"], mc["p10"][::-1]]),
                fill="toself", fillcolor="rgba(34,197,94,0.15)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name="Scenari 10°–90°"))
            for p in mc["sample_paths"][::4]:
                fig_mc.add_trace(go.Scatter(x=t, y=p, line=dict(color="gray", width=0.5),
                                            opacity=0.12, showlegend=False, hoverinfo="skip"))
            fig_mc.add_trace(go.Scatter(x=t, y=mc["p50"], name="Atteso (50°)",
                                        line=dict(color="#2563eb", width=3)))
            fig_mc.add_trace(go.Scatter(x=t, y=mc["p90"], name="Ottimistico (90°)",
                                        line=dict(color="#16a34a", width=2, dash="dot")))
            fig_mc.add_trace(go.Scatter(x=t, y=mc["p10"], name="Pessimistico (10°)",
                                        line=dict(color="#dc2626", width=2, dash="dot")))
            fig_mc.add_hline(y=total, line=dict(color="black", width=1, dash="dash"),
                             annotation_text="Capitale iniziale")
            fig_mc.update_layout(xaxis_title="Anni", yaxis_title="Valore portafoglio (€)",
                                 height=460, legend=dict(orientation="h", y=-0.15),
                                 margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig_mc, use_container_width=True)
            scen_rows = []
            for h in (3, 5, 10, 15):
                s = mc["terminal"][h]
                scen_rows.append({
                    "Orizzonte": f"{h} anni",
                    "Pessimistico (10°)": f"{s['p10']:,.0f} €",
                    "Atteso (50°)": f"{s['p50']:,.0f} €",
                    "Ottimistico (90°)": f"{s['p90']:,.0f} €",
                })
            st.dataframe(pd.DataFrame(scen_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# ROUTER
# ===========================================================================
if st.session_state.page == "questionario":
    page_questionnaire()
elif st.session_state.page == "etf":
    page_etf()
elif st.session_state.page == "report":
    page_report()
elif st.session_state.page == "myportfolio":
    page_my_portfolio()
