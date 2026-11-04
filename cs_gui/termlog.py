"""Render a process log the way a terminal shows it.

The runner and the database download write stdout/stderr of the package unchanged to a file
(``dup2`` on the file descriptors, no buffering), so the file holds what a terminal would
receive: ``\\n`` ends a line, ``\\r`` moves back to the start of the line (tqdm redraws its bar
this way) and ANSI escape sequences move the cursor or clear lines (nested tqdm bars, colours).

``Terminal`` interprets that stream: a progress bar keeps a single, updated line, while every
line ended by ``\\n`` stays in the history. ``read_log`` reads a log file incrementally (only the
bytes added since the last call are parsed) so large logs refreshed every few seconds stay cheap.
"""
from __future__ import annotations

import codecs
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

# CSI sequences (ESC [ params final), other two-character ESC sequences, and the control
# characters that move the cursor. Everything else is text.
_TOKEN = re.compile(r"\x1b\[([0-9;?]*)([@-~])|\x1b[@-Z\\-_]|[\r\n\b]")

DISPLAY_MAX_LINES = 50_000      # lines sent to the browser; the full log stays downloadable from disk


class Terminal:
    """Minimal line-oriented terminal: text overwrites at the cursor like in a real terminal."""

    def __init__(self) -> None:
        self.lines: list[str] = [""]
        self.row = 0
        self.col = 0

    def feed(self, text: str) -> None:
        pos = 0
        for m in _TOKEN.finditer(text):
            if m.start() > pos:
                self._write(text[pos:m.start()])
            pos = m.end()
            tok = m.group(0)
            if tok == "\n":
                self.row += 1
                self.col = 0
                if self.row == len(self.lines):
                    self.lines.append("")
            elif tok == "\r":
                self.col = 0
            elif tok == "\b":
                self.col = max(0, self.col - 1)
            elif m.group(2):
                self._csi(m.group(1), m.group(2))
            # other ESC sequences (charset selection, ...) have no visible effect: dropped
        if pos < len(text):
            self._write(text[pos:])

    def _write(self, s: str) -> None:
        line = self.lines[self.row]
        if self.col == len(line):
            self.lines[self.row] = line + s
        else:
            if self.col > len(line):
                line += " " * (self.col - len(line))
            self.lines[self.row] = line[:self.col] + s + line[self.col + len(s):]
        self.col += len(s)

    def _csi(self, params: str, final: str) -> None:
        nums = [int(p) if p.isdigit() else 0 for p in params.lstrip("?").split(";")] if params else []
        n = nums[0] if nums else 0
        if final == "A":                       # cursor up (tqdm with several bars)
            self.row = max(0, self.row - max(n, 1))
        elif final == "B":                     # cursor down
            self.row += max(n, 1)
            while self.row >= len(self.lines):
                self.lines.append("")
        elif final == "C":                     # cursor forward
            self.col += max(n, 1)
        elif final == "D":                     # cursor back
            self.col = max(0, self.col - max(n, 1))
        elif final == "G":                     # cursor to column n
            self.col = max(n, 1) - 1
        elif final == "K":                     # erase in line
            line = self.lines[self.row]
            if n == 0:
                self.lines[self.row] = line[:self.col]
            elif n == 1:
                self.lines[self.row] = " " * min(self.col, len(line)) + line[self.col:]
            else:
                self.lines[self.row] = ""
        # colours (m), modes (h/l), screen erase (J), ...: no effect on the text

    def text(self, max_lines: int | None = None) -> str:
        lines = self.lines
        if len(lines) > 1 and lines[-1] == "":
            lines = lines[:-1]                 # the newline after the last line does not open a visible line
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[-max_lines:]
        return "\n".join(l.rstrip() for l in lines)

    def n_lines(self) -> int:
        return len(self.lines) - (1 if len(self.lines) > 1 and self.lines[-1] == "" else 0)


def render(text: str, max_lines: int | None = None) -> str:
    """Terminal view of ``text`` (convenience for whole strings)."""
    t = Terminal()
    t.feed(text)
    return t.text(max_lines)


class _LogState:
    def __init__(self, ino: int) -> None:
        self.ino = ino
        self.offset = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.term = Terminal()


_CACHE: OrderedDict[str, _LogState] = OrderedDict()
_CACHE_SIZE = 32
_LOCK = threading.Lock()


def read_log(path: Path | str, max_lines: int = DISPLAY_MAX_LINES) -> dict:
    """Terminal view of a log file, parsed incrementally.

    Returns ``text`` (at most ``max_lines`` lines, the most recent), ``n_lines`` (whole log),
    ``hidden_lines`` (older lines not included in ``text``), ``size`` (bytes on disk) and
    ``exists``. A file that shrank or was replaced is parsed again from the start.
    """
    path = str(path)
    try:
        st = os.stat(path)
    except OSError:
        return {"exists": False, "text": "", "n_lines": 0, "hidden_lines": 0, "size": 0}
    with _LOCK:
        state = _CACHE.get(path)
        if state is None or state.ino != st.st_ino or st.st_size < state.offset:
            state = _LogState(st.st_ino)
        _CACHE[path] = state
        _CACHE.move_to_end(path)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
        if st.st_size > state.offset:
            with open(path, "rb") as f:
                f.seek(state.offset)
                data = f.read(st.st_size - state.offset)
            state.offset += len(data)
            # a UTF-8 character split across two writes is completed at the next read
            state.term.feed(state.decoder.decode(data))
        n = state.term.n_lines()
        return {"exists": True, "text": state.term.text(max_lines), "n_lines": n,
                "hidden_lines": max(0, n - max_lines), "size": state.offset}
