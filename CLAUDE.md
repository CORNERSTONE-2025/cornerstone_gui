# Working rules for Claude Code in this repository

- Commits use the git identity configured on the machine (the owner's). Never add
  `Co-Authored-By: Claude` or any other AI attribution to commit messages or pull requests.
- Never commit local files of the owner: `gui_settings.json`, `.env`, `workspace/`, `data/`.
- Code comments and docs in English; answers to the owner in Italian.
- Small steps: one change, its tests, one commit and push per step.
- Do not publish or share anything outside the repository without asking.
- The pipeline logic lives in the package `cornerstone_db_package` (labeling, data store, training);
  this repository is the GUI and the runner. Do not change scientific logic without asking.
- Tests: `PYTHONPATH=../cornerstone_db_package pytest -m "not integration"` (unit), or inside
  the image: `docker compose run --rm gui pytest`. Nightly campaign: `docs/NIGHTLY_TESTS.md`.
