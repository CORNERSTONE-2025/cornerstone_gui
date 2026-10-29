#!/bin/sh
# Container entry point: prepares the workspace, falls back to demo mode when no database is
# mounted, then starts Streamlit. Any argument replaces the server command (e.g. `pytest`).
set -e

mkdir -p "$CS_GUI_WORKSPACE" "$CS_GUI_DEMO_WORKSPACE"

if [ ! -f "$CS_GUI_SETTINGS_FILE" ] && [ ! -f "$CS_GUI_DB_PATH" ]; then
    echo "[cornerstone-gui] No database at $CS_GUI_DB_PATH: starting in DEMO mode (synthetic data)."
    echo "[cornerstone-gui] Mount the real database to use real mode (see docs/DOCKER.md)."
    python - <<'EOF'
from cs_gui.demo import build_demo_db
from cs_gui.settings import load_settings, save_settings

s = load_settings()
s.mode = "demo"
save_settings(s)
if not s.demo_db_path.exists():
    print("[cornerstone-gui] Generating the demo database:", build_demo_db(s.demo_db_path))
EOF
fi

if [ "$#" -gt 0 ]; then
    exec "$@"
fi
exec streamlit run app/streamlit_app.py \
    --server.address 0.0.0.0 --server.port 8501 --server.fileWatcherType none
