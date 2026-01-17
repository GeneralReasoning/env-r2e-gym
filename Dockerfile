FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y && apt install -y \
    software-properties-common \
    docker.io \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    git \
    git-lfs \
    wget \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv venv --python 3.11

COPY . /app
RUN uv pip install -r /app/requirements.txt

RUN uv run python -c "from datasets import load_dataset; load_dataset('R2E-Gym/R2E-Gym-V1', split='train')"
RUN uv run python -c "from datasets import load_dataset; load_dataset('R2E-Gym/R2E-Gym-Subset', split='train')"

EXPOSE 8080
CMD ["uv", "run", "python", "/app/server.py"]
