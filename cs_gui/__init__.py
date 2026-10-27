"""CORNERSTONE GUI: interactive front-end and experiment tracker for the CORNERSTONE DB package.

The package is split in layers:
- ``settings``, ``store``, ``jobs``, ``pipeline``: application logic (stdlib only, importable
  both from the GUI environment and from the CORNERSTONE virtualenv).
- ``runner``: integration layer, executed *inside* the CORNERSTONE virtualenv as a subprocess.
- ``ui`` and ``app/``: Streamlit front-end.
"""

__version__ = "0.1.0"
