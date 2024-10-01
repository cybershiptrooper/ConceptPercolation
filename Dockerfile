# Inspiriation from:
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0
# https://github.com/orgs/python-poetry/discussions/1879#discussioncomment-2255728

FROM python:3.11-buster as builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VERSION=1.7.1

# Install some useful packages
RUN apt update && \
    apt install -y rsync git vim nano curl wget htop tmux zip unzip iputils-ping openssh-server strace nodejs python3 python3-pip ffmpeg zstd gcc \
    && python3 -m pip install --upgrade --no-cache-dir pip requests

# # Install pipx, Poetry and dependencies for poetry install
# RUN pip install pipx && \
# pipx install "poetry==$POETRY_VERSION" && \
# apt-get update -q && \
# apt-get clean && \
# rm -rf /var/lib/apt/lists/*

WORKDIR /ConceptPerlocation

COPY requirements.txt ./
RUN touch README.md

# RUN --mount=type=cache,target=$POETRY_CACHE_DIR /root/.local/bin/poetry install --no-root
RUN python -m venv .venv
RUN .venv/bin/pip install --upgrade pip
RUN .venv/bin/pip install -r requirements.txt
# RUN source .venv/bin/activate

FROM python:3.11-slim-buster as runtime

# Install runtime dependencies
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends tmux && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /ConceptPerlocation

ENV VIRTUAL_ENV=/ConceptPerlocation/.venv \
    PATH="/ConceptPerlocation/.venv/bin:/root/.local/bin/:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY . /ConceptPerlocation

# ENTRYPOINT ["python", "main.py"]
