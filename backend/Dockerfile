FROM python:3.13.14-slim-bookworm@sha256:9d7f287598e1a5a978c015ee176d8216435aaf335ed69ac3c38dd1bbb10e8d64 AS system

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    openssl \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

FROM system AS certificates

WORKDIR /certs

FROM system AS base

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS sandbox

COPY sandbox_runner.py ./
COPY config ./config
COPY extractors ./extractors
COPY services ./services
COPY utils ./utils
COPY validators ./validators

CMD ["python", "sandbox_runner.py"]

FROM base AS app

COPY run.py ./
COPY sandbox_runner.py ./
COPY Dockerfile ./
COPY app ./app
COPY config ./config
COPY data ./data
COPY extractors ./extractors
COPY models ./models
COPY services ./services
COPY utils ./utils
COPY validators ./validators
RUN mkdir -p results/original/model_comparisons results/url_corpus/model_comparisons

EXPOSE 5001

CMD ["python", "run.py"]
