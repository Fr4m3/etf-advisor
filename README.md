# ETF Advisor

App web **Full-Stack Python** per la profilazione comportamentale dell'investitore
e la raccomandazione/ottimizzazione di un portafoglio di **ETF UCITS** (regolamento
UE), con dashboard analitica interattiva.

L'applicazione è in `etf_advisor/` (vedi `etf_advisor/README.md` per la documentazione
completa, l'architettura e la metodologia).

## Avvio locale
```bash
cd etf_advisor
pip install -r ../requirements.txt   # oppure requirements.txt dentro etf_advisor
python -m streamlit run app.py
```
Apri `http://localhost:8501`.

## Funzionalità principali
- **Questionario comportamentale** (wizard) → profilo di rischio + Asset Allocation Target.
- **Allocazione personalizzata** (slider azionario/bond) e **ottimizzazione geografica**
  (presets + sliders) tramite ottimizzatore **Max Sharpe** (Markowitz, vincoli lineari).
- **Dashboard**: donut geo/settoriali, Sharpe/Vol/MaxDD/TER, **Monte Carlo** con
  ribilanciamento, **confronto tra profili**.
- **Metadati live da JustETF** ed **export in CSV / PDF**.

## Deploy (Streamlit Community Cloud — gratuito, da GitHub)
1. Crea un repo pubblico su GitHub e pusha questo progetto (vedi comandi sotto).
2. Vai su https://share.streamlit.io → “New app” → collega il repo.
3. **Main file path:** `etf_advisor/app.py`
4. **Requirements:** il `requirements.txt` è già alla radice del repo.
5. Clicca **Deploy**. L'app gira online (fetch prezzi da Yahoo Finance all'avvio).

> Alternative di hosting: Render, Hugging Face Spaces, o `docker compose up` (vedi
> `etf_advisor/Dockerfile` e `etf_advisor/docker-compose.yml`).

⚠️ Strumento educativo/dimostrativo: non costituisce consulenza finanziaria.
