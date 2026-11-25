You are writing the morning report of the nightly test campaign of CORNERSTONE (solar flare
prediction: Parquet data store of SHARP/GOES data, parametric labeling, HDF5 datasets, splits,
training of ML models, inference; a Streamlit GUI drives the pipeline). Nobody is watching: do not
ask questions, work only with the files, and finish by writing the report.

Inputs, all in the run folder `{{RUN}}`:
- `results.json`: one record per test (id, phase, title, status pass/warn/fail/skip, message,
  details with the measured values, duration, traceback when it failed, log file);
- `summary.md`: automatic table of the same results;
- `build.json`: result of the Docker image build (latest package from GitHub);
- `logs/`: one log per test, and `run.log` (the whole night);
- `figures/`: PNG figures produced by the tests;
- the test definitions are in `{{REPO}}/scripts/nightly/campaign.py` and the plan in
  `{{REPO}}/docs/NIGHTLY_TESTS.md`; earlier runs (if any) are in the sibling folders of `{{RUN}}`
  (same layout): use their `results.json` to spot regressions.

Write `{{RUN}}/REPORT.md` in **Italian** (identifiers, test ids and file names stay as they are),
for the researcher who owns the project. Structure:

1. **Esito in breve** — 3 to 6 lines: overall verdict, how many tests passed / warned / failed /
   were skipped, the most important problem if any, and whether the night ended in time.
2. **Tabella dei risultati** — one row per test: status emoji (✅ ⚠️ ❌ ⏭️), test, duration, key
   number(s) (e.g. test TSS, rows, minutes).
3. **Cosa non funziona** — for each `fail`: what failed, the relevant lines of the traceback or of
   the log (read the log file), the probable cause in the code (cite `file:line` when you can find
   it), and how to reproduce it (`NIGHTLY_ONLY=<id> scripts/nightly/run.sh --quick` or the command).
4. **Criticità e avvertimenti** — every `warn`, plus anything suspicious you notice in the
   numbers even when the test passed (e.g. a TSS far from the others, a long duration, missing
   values, a stale data store, low disk space).
5. **Confronto con le notti precedenti** — only if earlier runs exist: tests whose status changed,
   metrics that moved by more than 0.03 TSS, durations that grew by more than 50%.
6. **Test saltati** — which and why (deadline, missing input such as the former database).
7. **Dati e prestazioni** — data store size and freshness, timings of each pipeline step, model
   metrics in a small table; include the figures with relative links `figures/<name>.png`.
8. **Ambiente** — commits of package and GUI, CPU/RAM/CUDA, disk, image build result.

Rules: report only what the files show; do not invent numbers; when something cannot be
determined say so. Keep it factual and compact (no more than about 2 pages of text plus tables).
Do not modify any file other than `{{RUN}}/REPORT.md`.
