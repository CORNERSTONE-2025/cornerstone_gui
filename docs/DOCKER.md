# Running the GUI with Docker

The image contains the GUI, the CORNERSTONE package (cloned from GitHub at build time) and a
CPU-only Python environment with all their dependencies. Nothing else needs to be installed.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) with Compose v2 (`docker compose version`).
  Docker Desktop on Windows and macOS works too.
- About 5 GB of free disk space for the image, plus space for your experiments.
- About 1 GB more for the data store (SHARP DEF/NRT, GOES events), which grows slowly.

## Quick start

```bash
git clone https://github.com/CORNERSTONE-2025/cornerstone_gui.git
cd cornerstone_gui
docker compose up --build -d
```

Open **http://localhost:8501**. The first build downloads the dependencies and takes a few minutes.

Without data the GUI starts in **demo mode** on synthetic data, generated automatically. Every
page shows a banner, and the results have no scientific value.

## Using the real data

Open the **Data** page. It shows what the store contains and runs, in the background with a live log,
one of:

- **Update from the sources**: downloads SHARP DEF and NRT from JSOC, the GOES flares from NOAA SWPC and
  the HARP ↔ NOAA map. The first time this takes a few hours (SHARP); afterwards only what changed is
  downloaded (minutes). An interrupted run resumes from the last complete file.
- **Published snapshot**: a Parquet copy of the store published by the project (minutes), then
  *Update from the sources* adds what is newer.
- **Import the former database**: converts a `cornerstone.sqlite` of the former dump (about 2 minutes),
  then *Update from the sources* brings it up to date.

When the store has data, choose **Real** mode in *Home → GUI configuration*.

The store is kept in the Docker volume `db` (mounted at `/data`, folder `datastore/`): it survives
`docker compose down` and image rebuilds, and is deleted only by `docker compose down -v`.

### Using a folder of your computer

Create a `.env` file next to `docker-compose.yml`:

```bash
CORNERSTONE_DATA_DIR=/path/to/folder
```

The store goes to `/path/to/folder/datastore`; a `cornerstone.sqlite` in that folder (or the file named by
`CORNERSTONE_DB_FILE`) is offered for import. The folder must be writable by the container user (uid 1000,
usually the first user of the computer).

## Day-to-day commands

| Command | What it does |
|---|---|
| `docker compose up -d` | start the GUI in the background |
| `docker compose logs -f` | follow the server logs |
| `docker compose down` | stop it (experiments are kept) |
| `docker compose up --build -d` | rebuild after a GUI update (`git pull`) |
| `docker compose exec gui pytest -m "not integration"` | run the unit tests inside the container |

### Updating from an earlier version

`git pull` then `docker compose up --build -d`: the data and the experiments are in volumes or
folders outside the image, so a rebuild keeps them (never use `docker compose down -v`, which deletes
the volumes).

From a version that used the SQLite database: the database stays where it was (volume `db`, or the
folder of `CORNERSTONE_DB_DIR`, still honoured). Open the **Data** page, tab *Import the former
database*, run the import and then *Update from the sources*. Experiments made with the former
labeling remain readable; new labelings use the data store. After the import the `.sqlite` file is no
longer needed.

## Where the experiments are stored

All artifacts, logs and GUI settings live in the Docker volume `workspace`, mounted at `/workspace`
(`real/` and `demo/` subfolders). The volume survives `docker compose down` and image rebuilds; it is
deleted only by `docker compose down -v`.

To copy the experiments out of the container:

```bash
docker compose cp gui:/workspace ./workspace_backup
```

## Options

| Variable (in `.env`) | Default | Meaning |
|---|---|---|
| `CS_GUI_PORT` | `8501` | port on your computer |
| `CORNERSTONE_DATA_DIR` | `db` (Docker volume) | folder holding the data store (`datastore/`) |
| `CORNERSTONE_DB_FILE` | `cornerstone.sqlite` | former SQLite database offered for import, in that folder |
| `CORNERSTONE_REF` | `main` | branch, tag or commit of the CORNERSTONE package used at build time |

To pin the package version (recommended for reproducible experiments):

```bash
CORNERSTONE_REF=897a662 docker compose build
```

The commit actually used is recorded in every artifact (*Experiments → Provenance*).

## Access from another computer

The port is published on `127.0.0.1` only, because the GUI has no login: anyone who can reach it can
start jobs and delete artifacts. To use a GUI running on a remote server, open an SSH tunnel from your
computer:

```bash
ssh -L 8501:localhost:8501 user@server
```

and browse to http://localhost:8501. Only remove `127.0.0.1:` from `docker-compose.yml` on a trusted network.

## Without Compose

```bash
docker build -t cornerstone-gui .
docker run -d --name cornerstone-gui -p 127.0.0.1:8501:8501 --shm-size 2g \
  -v cornerstone-workspace:/workspace \
  -v cornerstone-db:/data \
  cornerstone-gui
```

## Limitations of the image

- CPU only: all models, including ROCKET/MINIROCKET and the PyTorch ones, run on CPU.
- PyTorch models train slowly on CPU: use the subsample option in *Dataset → HDF5 extraction* and few
  epochs for quick tests.
