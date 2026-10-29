# syntax=docker/dockerfile:1.7
#
# CORNERSTONE GUI: the Streamlit GUI and the CORNERSTONE package in one CPU-only image.
#
#   docker compose up --build            # then open http://localhost:8501
#
# The CORNERSTONE package is fetched from GitHub at build time; choose the version with
#   docker compose build --build-arg CORNERSTONE_REF=<branch | tag | commit>
FROM python:3.10-slim

ARG CORNERSTONE_REPO=https://github.com/CORNERSTONE-2025/cornerstone_db_package.git
ARG CORNERSTONE_REF=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    # provenance reads the git state of /opt/cornerstone and /app as a non-root user
    && git config --system --add safe.directory '*'

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /usr/local/bin/uv

# One environment for both the GUI and the package (the runner uses the same interpreter)
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1
COPY docker/requirements-cornerstone.txt docker/requirements-gui.txt /tmp/
RUN uv venv /opt/venv --python /usr/local/bin/python \
    && uv pip install --index-strategy unsafe-best-match \
         --extra-index-url https://download.pytorch.org/whl/cpu \
         -r /tmp/requirements-cornerstone.txt -r /tmp/requirements-gui.txt

# The package checkout keeps its .git so every artifact records the exact commit it ran with
ADD --keep-git-dir=true ${CORNERSTONE_REPO}#${CORNERSTONE_REF} /opt/cornerstone

RUN useradd --create-home --uid 1000 cornerstone \
    && mkdir -p /workspace /data \
    && chown cornerstone:cornerstone /workspace

WORKDIR /app
COPY --chown=cornerstone:cornerstone . /app

ENV CS_GUI_CORNERSTONE_ROOT=/opt/cornerstone \
    CS_GUI_CORNERSTONE_PYTHON=/opt/venv/bin/python \
    CS_GUI_WORKSPACE=/workspace/real \
    CS_GUI_DEMO_WORKSPACE=/workspace/demo \
    CS_GUI_SETTINGS_FILE=/workspace/gui_settings.json \
    CS_GUI_DB_PATH=/data/cornerstone.sqlite \
    MPLBACKEND=Agg \
    CLEARML_OFFLINE_MODE=1

USER cornerstone
VOLUME ["/workspace"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

ENTRYPOINT ["tini", "--", "/app/docker/entrypoint.sh"]
