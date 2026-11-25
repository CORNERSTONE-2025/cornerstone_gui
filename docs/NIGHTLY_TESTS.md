# Nightly tests

A test campaign of the whole pipeline that runs unattended at night on a dedicated machine, and a
report written in the morning by Claude Code.

```
21:00  cron → scripts/nightly/run.sh      build the image (latest package), run the tests until 07:30
07:30  cron → scripts/nightly/report.sh   Claude Code writes REPORT.md; push to the branch test-reports
```

The tests run in a container of the GUI image, with the same data folder as the GUI (read only in
practice: the update test works on a copy and the last test checks that the store is unchanged).
Claude is used only for the report: it may read files and write in the folder of the run.

## What is tested

Tests are ordered by priority. Before each test the time left to the deadline is compared with its
estimated duration: a test that would not end in time is skipped and reported as such.

| id | what | about |
|---|---|---|
| P0.env | versions (package and GUI commits), CPU, RAM, CUDA, disk, data store present | 1 min |
| P2.manifest | every store file matches the sha256 of the manifest | 5 min |
| P2.sharp | SHARP DEF/NRT: rows vs manifest, duplicated keys, order, missing values, last record; figure | 5 min |
| P2.goes | GOES events per year and class, duplicates, HARP-NOAA map; figure | 1 min |
| P9.validation | wrong parameters (thresholds, windows, sampling, folds, leaky splits…) are refused | 1 min |
| P4.labeling | labels DEF/NRT × C/M/X: YES ⇔ flare ≥ S_p, strat_class consistent, no duplicates, YES(C) ≥ YES(M) ≥ YES(X), timing; figure | 20 min |
| P4.determinism | same labels twice; S_s ≠ S_p and L = 48 h | 5 min |
| P4.legacy | comparison with the static tables (only if `cornerstone.sqlite` is in the data folder) | 10 min |
| P5.def_xgb | real chain DEF M+ 24 h: labels → dataset → split → XGBoost (4 windows, 5 folds) → inference; integrity, leakage, TSS | 30 min |
| P5.nrt_xgb | same chain on NRT | 20 min |
| P5.def_all22 | DEF with the 22 raw SHARP keywords | 20 min |
| P7.reproduce | dataset and XGBoost training run again: identical arrays and metrics | 20 min |
| P8.bundle | export of the chain as a bundle, import into an empty workspace, mode check | 5 min |
| P10.gui_pages | every GUI page renders without exceptions (Streamlit AppTest) | 2 min |
| P1.* | the automatic test suites: package, GUI unit, GUI integration (synthetic data) | 20 min |
| P6.lstm … P6.minirocket | LSTM, InceptionTime, Transformer, MiniRocket: short training (5 epochs, 2 folds, 24 h) + inference | 30–60 min each |
| P3.update | incremental update from JSOC/NOAA on a copy of the store | 30–60 min |
| P6.other_models | the other models offered by the GUI: 1 epoch, 1 fold, 3 h window — which ones work | 1–3 h |
| P11.inceptiontime_full | InceptionTime as the reference experiment (4 windows, 5 folds, 100 epochs), compared with its test TSS | ~6 h on CPU |
| P12.store_unchanged | the data store files are unchanged | 5 min |

The image is CPU only: neural models train on CPU (InceptionTime full ≈ 5–6 h on 22 cores).

## Results

```
~/cornerstone_nightly/<date>/
├── REPORT.md        # Claude's report, in Italian (07:30)
├── summary.md       # automatic table, written after every test
├── results.json     # every test: status, message, measured values, traceback
├── build.json       # Docker image build
├── figures/         # PNG
├── logs/            # one log per test
├── run.log          # the whole night; report.log: the morning step
└── workspace/       # artifacts produced by the tests (kept for the last 3 nights)
```

With push enabled, `REPORT.md`, `summary.md`, `results.json` and the figures are committed to the
branch **`test-reports`** (folder `reports/<date>/`), which has no code: read them on GitHub or with
`git fetch origin test-reports && git show origin/test-reports:reports/<date>/REPORT.md`.

## Set-up of the test machine (once)

1. **Software**: git, Docker with the compose plugin (https://docs.docker.com/engine/install/;
   add the user to the `docker` group: `sudo usermod -aG docker $USER`, then log in again), Claude
   Code (`curl -fsSL https://claude.ai/install.sh | bash`), `flock` and `timeout` (in util-linux and
   coreutils, present on Ubuntu).
2. **Git identity**, the one of the commits of the reports:
   ```bash
   git config --global user.name  "<name>"
   git config --global user.email "<email>"
   ```
   and push access to GitHub (SSH key or `gh auth login`).
3. **Repositories**:
   ```bash
   mkdir -p ~/PHD/repos && cd ~/PHD/repos
   git clone https://github.com/CORNERSTONE-2025/cornerstone_gui.git
   cd cornerstone_gui
   ```
4. **Data**: start the GUI once and download the published snapshot from the Data page
   (`docker compose up --build -d`, open http://localhost:8501, Data → Published snapshot, ~15
   min). The campaign uses the same data folder (`CORNERSTONE_DATA_DIR` in `.env`, or the `db`
   volume). Optional: copy the former `cornerstone.sqlite` into that folder for the P4.legacy test.
5. **Claude Code**: run `claude` once in the repository folder, log in with `/login` (same
   account) and exit. Check that `claude -p "say ok"` answers without asking anything.
6. **Trial run** (about one hour, results in `~/cornerstone_nightly/<date>_quick/`):
   ```bash
   scripts/nightly/run.sh --quick
   scripts/nightly/report.sh ~/cornerstone_nightly/$(date +%F)_quick
   ```
   Read `summary.md` and `REPORT.md`; if the push worked the branch `test-reports` exists on GitHub.
7. **Schedule** with `crontab -e` (replace `<user>`; `which claude` gives the folder to put in PATH):
   ```cron
   SHELL=/bin/bash
   PATH=/home/<user>/.local/bin:/usr/local/bin:/usr/bin:/bin
   0 21 * * *  /home/<user>/PHD/repos/cornerstone_gui/scripts/nightly/run.sh
   30 7 * * *  /home/<user>/PHD/repos/cornerstone_gui/scripts/nightly/report.sh
   ```
   Every night, or e.g. `0 21 * * 1-5` for Monday to Friday nights.
8. **No sleep at night**: cron runs without anybody logged in, but the machine must stay on:
   `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`
   (undo with `unmask`).

## Options

Environment variables, also in the crontab lines (`NIGHTLY_DEADLINE=06:00 /home/.../run.sh`):

| variable | default | |
|---|---|---|
| `NIGHTLY_DIR` | `~/cornerstone_nightly` | results folder |
| `NIGHTLY_DEADLINE` | `07:30` | no test starts if it would not end by then; hard stop 15 min later |
| `NIGHTLY_ONLY` | | only these tests or phases, e.g. `P5.def_xgb,P7.reproduce` |
| `NIGHTLY_KEEP` | `3` | artifact workspaces kept (results are always kept) |
| `NIGHTLY_PUSH` | `1` | `0`: the report stays on the machine |

Disk: a full night needs about 20–30 GB (datasets, splits, models, a copy of the store).

## Troubleshooting

- Nothing in the morning: `grep CRON /var/log/syslog`, then `~/cornerstone_nightly/<date>/run.log`.
- `REPORT.md` missing: `report.log` (Claude not in the PATH of cron, not logged in, usage limit);
  `summary.md` is written anyway.
- A test failed: its log is `logs/<id>.log`; run it alone with
  `NIGHTLY_ONLY=<id> scripts/nightly/run.sh --quick` (tests that need an earlier one, e.g. P7 after
  P5.def_xgb, need it in the list too).
