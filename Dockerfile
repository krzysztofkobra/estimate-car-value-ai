# Reconcile: runtime.txt = python-3.10.15 (trzymamy się 3.10, README zaktualizowane).
FROM python:3.10-slim

# libgomp1 wymagane przez LightGBM.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Najpierw zależności (lepsze cache warstw).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Uruchamiamy jako użytkownik non-root (bezpieczeństwo).
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000

# Health-check oparty o /health (model załadowany). Bez dodatkowych pakietów — używa stdlib pythona.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'))" || exit 1

# Shell-form dla ekspansji ${PORT} z domyślną wartością.
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
