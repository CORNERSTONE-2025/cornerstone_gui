# Running the GUI with Docker

The image contains the GUI, the CORNERSTONE package (cloned from GitHub at build time) and a
CPU-only Python environment with all their dependencies. Nothing else needs to be installed.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) with Compose v2 (`docker compose version`).
  Docker Desktop on Windows and macOS works too.
- About 5 GB of free disk space for the image, plus space for your experiments.
- Optional: the CORNERSTONE SQLite database (`cornerstone.sqlite`, about 4 GB) to work on real data.

## Quick start

```bash
git clone https://github.com/CORNERSTONE-2025/cornerstone_gui.git
cd cornerstone_gui
docker compose up --build -d
```

Open **http://localhost:8501**. The first build downloads the dependencies and takes a few minutes.

Without a database the GUI starts in **demo mode** on a synthetic database, which is generated
automatically. Every page shows a banner, and the results have no scientific value.

## Using the real database

Put the database in the `data/` folder next to `docker-compose.yml`:

```
cornerstone_gui/
├── docker-compose.yml
└── data/
    └── cornerstone.sqlite
```

The database is mounted **read-only**: the GUI never modifies it. If it lives elsewhere or has another
name, create a `.env` file next to `docker-compose.yml`:

```bash
CORNERSTONE_DB_DIR=/path/to/folder
CORNERSTONE_DB_FILE=my_database.sqlite
```

Then restart with `docker compose up -d` and choose **Real** mode in *Home → GUI configuration*
(the container only starts in demo mode by itself on its first run without a database).

## Day-to-day commands

| Command | What it does |
|---|---|
| `docker compose up -d` | start the GUI in the background |
| `docker compose logs -f` | follow the server logs |
| `docker compose down` | stop it (experiments are kept) |
| `docker compose up --build -d` | rebuild after a GUI update (`git pull`) |
| `docker compose exec gui pytest -m "not integration"` | run the unit tests inside the container |

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
| `CORNERSTONE_DB_DIR` | `./data` | folder holding the database |
| `CORNERSTONE_DB_FILE` | `cornerstone.sqlite` | database file name |
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
  -v /path/to/folder:/data:ro \
  cornerstone-gui
```

## Limitations of the image

- CPU only: ROCKET and MINIROCKET need a CUDA GPU (cupy) and are blocked by the GUI. All other models,
  including the PyTorch ones, run on CPU.
- PyTorch models train slowly on CPU: use the subsample option in *Dataset → HDF5 extraction* and few
  epochs for quick tests.
