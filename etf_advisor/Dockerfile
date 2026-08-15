# Immagine leggera per ospitare l'app Streamlit ETF Advisor
FROM python:3.12-slim

WORKDIR /app

# Dipendenze di sistema (utili per numpy/pandas/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Porta esposta da Streamlit
EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Avvio non interattivo, accessibile da rete
CMD ["streamlit", "run", "app.py", \
     "--server.headless", "true", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]
