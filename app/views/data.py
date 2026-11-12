import streamlit as st

from cs_gui import ui

c = ui.page_header("Data",
                   "Local Parquet store of the data used by the pipeline, downloaded from the sources "
                   "and kept up to date (package module cornerstone.data).")

with st.expander("📖 What is in the store", expanded=False):
    st.markdown("""
| dataset | source | content |
|---|---|---|
| SHARP DEF | JSOC `hmi.sharp_720s` | definitive SHARP keywords, every 12 min per HARP (times in UTC) |
| SHARP NRT | JSOC `hmi.sharp_720s_nrt` | near-real-time SHARP keywords (own HARP numbers) |
| GOES flares | NOAA SWPC edited event reports | X-ray events 2010→today: class, peak, NOAA region, position |
| HARP ↔ NOAA | JSOC `all_harps_with_noaa_ars.txt` | map used to associate flares to DEF HARPs |

All records are kept as published (no filter): which records are used is decided by the
parameters of the *Labeling* and *Dataset* steps and saved with each artifact. GOES fluxes are
stored as reported; the calibration (science scale by default) is a labeling parameter.
""")

if c.settings.mode == "demo":
    st.info("Demo mode: the pipeline uses the synthetic store of the demo workspace "
            f"(`{c.settings.demo_data_root}`). The panel below manages the **real** store.")
ui.data_panel()
