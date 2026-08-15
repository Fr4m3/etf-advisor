# ETF Advisor — Profilazione Investitore & Portafoglio UCITS

App web **Full-Stack Python** per la profilazione comportamentale dell'investitore
e la raccomandazione/ottimizzazione di un portafoglio di **ETF UCITS** (regolamento
UE), con dashboard analitica interattiva.

## Come avviare

```bash
cd etf_advisor
pip install -r requirements.txt
streamlit run app.py
```

Apri `http://localhost:8501`.

## Funzionalità aggiuntive

- **Ribilanciamento periodico** (sidebar): `Nessuna (buy & hold)`, `Annuale`, `Trimestrale`, `Mensile`. Influenza sia il backtest storico sia la proiezione Monte Carlo (simulazione a livello di singoli asset con matrice di covarianza, non solo portafoglio aggregato).
- **Allocazione personalizzata** (Pagina 2): slider `Peso Azionario personalizzato (%)` che sovrascrive il target del profilo; la parte obbligazionaria diventa automaticamente `100 − Azionario`.
- **Ottimizzazione geografica** (Pagina 2): presets (`Market-cap globale`, `Home bias Europa`, `Solo Sviluppati`) o sliders personalizzati (USA/Europa/Emergenti/Pacifico) che vincolano la composizione della **parte azionaria** tramite vincoli lineari nell'ottimizzatore Max Sharpe.
- **Confronto tra profili** (Pagina 3): sovrappone le mediane Monte Carlo di più profili (Conservativo/Bilanciato/Crescita/Aggressivo) sullo stesso universo filtrato, capitale e politica di ribilanciamento, con tabella di TER/Sharpe/Vol/Mediane a 10-15 anni.
- **Metadati live da JustETF** (Pagina 2): bottone che aggiorna TER e dimensione fondo dagli ISIN ufficiali (fallback silenzioso se il sito blocca).
- **Export Report** (Pagina 3): composizione e scenari in **CSV**, report completo in **PDF** (grafici + tabelle, font Unicode per €/accenti).
- **Deploy containerizzato**: `Dockerfile` + `docker-compose.yml`.

  ```bash
  docker compose up --build   # poi apri http://localhost:8501
  ```

## Flusso (3 pagine)

1. **Questionario Comportamentale** — wizard step-by-step con linguaggio
   *non tecnico* (esempi di vita reale: "Hai 10.000€, il conto segna -30%…").
   4 sezioni → punteggio 4–12 → profilo (Conservativo / Bilanciato / Crescita /
   Aggressivo) con **Asset Allocation Target** (es. 20/80, 50/50, 75/25, 90/10).
2. **ETF Consigliati & Selezione UCITS** — filtri rigidi (solo UCITS, accumulo,
   dimensione fondo ≥ 100 M€, TER contenuto) + **ottimizzatore Max Sharpe**
   (Markowitz) vincolato al peso azionario target. Tabella con Ticker, ISIN,
   Nome, Classe, Peso %, TER %, Costo €/anno.
3. **Report & Dashboard** — donut geografica/settoriale, metriche (Sharpe,
   Volatilità, Max Drawdown, TER ponderato, costo €), e **simulazione Monte
   Carlo** del capitale a 3/5/10/15 anni (scenari 10°/50°/90° percentile).

## Architettura

| File | Responsabilità |
|------|----------------|
| `questionario.py` | Domande + mappatura punteggi → profilo → allocation target |
| `etf_universe.py` | Universo curato di ETF UCITS reali (ISIN, TER, dimensione, esposizioni geo/settoriali) |
| `data_fetcher.py` | Download prezzi `yfinance` + fallback sintetico offline + scraping JustETF (best-effort) + tasso risk-free |
| `finance.py` | Rendimenti, volatilità, **Sharpe**, **Max Drawdown**, TER ponderato, aggregazioni geo/settoriali |
| `optimizer.py` | **Max Sharpe / Min Variance** (scipy SLSQP) con vincoli long-only ed equity target |
| `montecarlo.py` | Simulazione Monte Carlo (Moto Browniano Geometrico) con bande percentile |
| `app.py` | UI Streamlit a 3 pagine, stato condiviso via `st.session_state` |

## Metodologia

- **Ottimizzazione:** massimizza `Sharpe = (E[Rp] − Rf) / σp` soggetto a
  Σw = 1, w ≥ 0 e peso azionario ≈ target utente.
- **Minimizzazione costi:** a parità di indice, il filtro predilige ETF con
  **TER minore** e **dimensione fondo maggiore**.
- **Risk-free:** tasso BCE/Euribor (default 2,5%, modificabile in sidebar).

## Note sui dati

- I **prezzi storici** sono reali (Yahoo Finance). Se la rete non risponde, il
  sistema genera serie sintetiche coerenti per classe di attivo.
- I **metadati anagrafici** (TER, dimensione, esposizioni) sono parametri di
  riferimento da aggiornare periodicamente; `fetch_justetf()` può arricchirli
  ma non è una dipendenza critica.
- Il modulo è **educativo/dimostrativo**: non costituisce consulenza
  finanziaria. Verificare sempre KID/KIID ufficiali prima di investire.

## Deploy (Docker)

```bash
# Build ed esecuzione
cd etf_advisor
docker compose up --build
# oppure solo docker
docker build -t etf-advisor .
docker run -p 8501:8501 etf-advisor
```

L'app gira in modalità headless e resta raggiungibile su `0.0.0.0:8501`.

## Test rapido (no rete)

```bash
python selftest.py   # esegue l'intera pipeline su dati sintetici
```
