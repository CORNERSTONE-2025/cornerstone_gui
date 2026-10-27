"""Integration layer executed inside the CORNERSTONE virtualenv (``python -m cs_gui.runner``).

Nothing here is imported by the Streamlit GUI: the GUI talks to this layer only through the
artifact store (manifests on disk) and subprocesses.
"""
