FROM python:3.11-bookworm

# Lingkungan
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates openssl curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /code
USER app

# Expose default (HF Spaces inject PORT env)
EXPOSE 7860

# Healthcheck: pakai /health (tidak butuh DB/Auth) dan curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-7860}/health || exit 1

# Jalankan uvicorn dengan PORT dari env Spaces
# Pastikan module path sesuai: app.main:app (ganti jika berbeda)
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
