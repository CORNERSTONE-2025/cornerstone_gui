"""CORNERSTONE GUI — entry point. Run with: ``streamlit run app/streamlit_app.py``."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from cs_gui import ui  # noqa: E402

st.set_page_config(page_title="CORNERSTONE GUI", page_icon="☀️", layout="wide")

P = ROOT / "app" / "views"        # not "pages/": Streamlit would also run those files on their own
pages = {
    "Project": [
        st.Page(P / "home.py", title="Home and status", icon="🏠", default=True),
    ],
    "Pipeline": [
        st.Page(P / "labeling.py", title="Labeling", icon="🏷️"),
        st.Page(P / "dataset.py", title="Dataset", icon="🗂️"),
        st.Page(P / "models.py", title="Model configuration", icon="⚙️"),
        st.Page(P / "training.py", title="Training and validation", icon="🏋️"),
        st.Page(P / "inference.py", title="Inference", icon="🔮"),
    ],
    "Analysis": [
        st.Page(P / "active_regions.py", title="Active Regions", icon="🌞"),
        st.Page(P / "metrics.py", title="Global metrics", icon="📊"),
        st.Page(P / "compare.py", title="Algorithm comparison", icon="⚖️"),
    ],
    "Reproducibility": [
        st.Page(P / "experiments.py", title="Experiments and pipeline", icon="🧬"),
    ],
}

with st.sidebar:
    c = ui.ctx()
    mode = "DEMO (synthetic)" if c.settings.mode == "demo" else "Real"
    st.markdown(f"**Mode:** {mode}  \n**Workspace:** `{c.settings.active_workspace.name}`")
    active = [m for m in c.store.list() if m["status"] in ("pending", "running")]
    st.markdown(f"**Active jobs:** {len(active)}")
    for m in active[:5]:
        st.caption(f"{ui.badge(m['status'])} {m['type']} · {m['name']}")

st.navigation(pages).run()
