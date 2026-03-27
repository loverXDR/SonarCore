FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml ./


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .




CMD ["uvicorn", "Api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
