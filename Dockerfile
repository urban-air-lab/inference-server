FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Berlin

RUN apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /bin/

WORKDIR /app
COPY . .

RUN uv sync --locked

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uv", "run",  "app/src/inference.py"]