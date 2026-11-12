#!/bin/sh
# Container entry point: prepares the workspace, falls back to demo mode on a first start without
# data, then starts Streamlit. Any argument replaces the server command (e.g. `pytest`).
set -e

mkdir -p "$CS_GUI_WORKSPACE" "$CS_GUI_DEMO_WORKSPACE"

if [ ! -f "$CS_GUI_SETTINGS_FILE" ] && [ ! -f "$CS_GUI_DATA_ROOT/manifest.json" ] && [ ! -f "$CS_GUI_LEGACY_DB" ]; then
    echo "[cornerstone-gui] No data in $CS_GUI_DATA_ROOT: starting in DEMO mode (synthetic data)."
    echo "[cornerstone-gui] Download the real data from the Data page, then switch to Real mode (Home)."
    PYTHONPATH="$CS_GUI_CORNERSTONE_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - <<'EOF'
from cs_gui.demo import build_demo_store
from cs_gui.settings import load_settings, save_settings

s = load_settings()
s.mode = "demo"
save_settings(s)
if not s.has_data():
    print("[cornerstone-gui] Generating the demo data:", build_demo_store(s.demo_data_root))
EOF
fi

if [ "$#" -gt 0 ]; then
    exec "$@"
fi
exec streamlit run app/streamlit_app.py \
    --server.address 0.0.0.0 --server.port 8501 --server.fileWatcherType none
