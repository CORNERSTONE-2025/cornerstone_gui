"""Terminal-like rendering of the process logs (cs_gui.termlog)."""
import time

from cs_gui import termlog


def _tqdm(desc: str, total: int) -> str:
    """What tqdm writes to a non-tty stream: \\r + the redrawn bar, \\n when closed."""
    out = ""
    for i in range(total + 1):
        pct = int(100 * i / total)
        out += f"\r{desc}: {pct:3d}%|{'█' * (pct // 10):10s}| {i}/{total} [00:0{i % 10}<00:01,  1.37it/s]"
    return out + "\n"


def test_progress_bar_keeps_one_updated_line():
    text = termlog.render("Epoch 5/5:   0%|          | 0/162 [00:00<?, ?it/s]"
                          "\rEpoch 5/5:   1%|          | 1/162 [00:00<01:57,  1.37it/s]"
                          "\rEpoch 5/5:   2%|▏         | 3/162 [00:02<01:44,  1.52it/s]")
    assert text == "Epoch 5/5:   2%|▏         | 3/162 [00:02<01:44,  1.52it/s]"
    assert termlog.render(_tqdm("Epoch 1/5", 162)).splitlines() == [
        "Epoch 1/5: 100%|██████████| 162/162 [00:02<00:01,  1.37it/s]"]


def test_shorter_redraw_overwrites_like_a_terminal():
    # a shorter text after \r leaves the tail of the previous one, as a terminal does
    assert termlog.render("abcdef\rXY") == "XYcdef"
    # tqdm(leave=False) clears its line with spaces: the next print takes that line
    assert termlog.render("bar 50%\r       \rdone\n") == "done"
    assert termlog.render("bar 50%\r\x1b[Kdone\n") == "done"


def test_plain_lines_are_kept():
    lines = [f"line {i}" for i in range(50)] + ["", "  indented", "last"]
    assert termlog.render("\n".join(lines) + "\n").splitlines() == lines
    assert termlog.render("a\r\nb\r\n") == "a\nb"            # CRLF line endings


def test_mixed_output_order_and_content():
    tb = ('Traceback (most recent call last):\n  File "train.py", line 10, in <module>\n'
          '    main()\nValueError: bad shape\n')
    stream = ("📊 Debug - Pred pos: 884/2583 = 34.2%\n"
              "📊 Debug - Threshold: 0.4694\n"
              "Epoch 4/5 - Val Loss: 0.9331, TSS: 0.3973\n"
              + _tqdm("Epoch 5/5", 162)
              + "\x1b[31mERROR\x1b[0m something failed\n"
              + tb)
    assert termlog.render(stream).splitlines() == [
        "📊 Debug - Pred pos: 884/2583 = 34.2%",
        "📊 Debug - Threshold: 0.4694",
        "Epoch 4/5 - Val Loss: 0.9331, TSS: 0.3973",
        "Epoch 5/5: 100%|██████████| 162/162 [00:02<00:01,  1.37it/s]",
        "ERROR something failed",
        *tb.rstrip("\n").splitlines()]


def test_nested_bars_with_cursor_up():
    # tqdm with position=1 draws the inner bar on the next line and goes back up with ESC[A
    s = "outer 0%\n\rinner 10%\x1b[A\router 50%\n\rinner 90%\x1b[A\router 100%\n\n"
    assert termlog.render(s).splitlines() == ["outer 100%", "inner 90%"]


def test_read_log_incremental(tmp_path):
    p = tmp_path / "run.log"
    p.write_bytes(b"start\nEpoch 1: 10%")
    r = termlog.read_log(p)
    assert r["text"] == "start\nEpoch 1: 10%" and r["n_lines"] == 2
    with open(p, "ab") as f:
        f.write("\rEpoch 1: 60%".encode())
        f.write("\rEpoch 1: 100%\nend ".encode() + "é".encode()[:1])    # é split across two writes
    assert termlog.read_log(p)["text"] == "start\nEpoch 1: 100%\nend"
    with open(p, "ab") as f:
        f.write("é".encode()[1:] + b"\n")
    assert termlog.read_log(p)["text"] == "start\nEpoch 1: 100%\nend é"
    p.write_bytes(b"new\n")                                   # truncated/replaced: parsed again
    assert termlog.read_log(p)["text"] == "new"
    assert termlog.read_log(tmp_path / "missing.log")["exists"] is False


def test_long_log_nothing_lost_and_fast(tmp_path):
    p = tmp_path / "long.log"
    with open(p, "w") as f:
        for i in range(12_000):
            f.write(f"line {i:05d} " + "x" * 60 + "\n")
            if i % 1000 == 0:
                f.write(_tqdm(f"Epoch {i // 1000}", 162))
    t0 = time.perf_counter()
    r = termlog.read_log(p)
    first = time.perf_counter() - t0
    lines = r["text"].splitlines()
    assert r["n_lines"] == 12_012 and r["hidden_lines"] == 0 and len(lines) == 12_012
    assert [l for l in lines if l.startswith("line ")] == [f"line {i:05d} " + "x" * 60 for i in range(12_000)]
    assert sum(l.startswith("Epoch") for l in lines) == 12
    with open(p, "a") as f:
        f.write("one more\n")
    t0 = time.perf_counter()
    assert termlog.read_log(p)["text"].endswith("one more")
    again = time.perf_counter() - t0
    assert first < 2.0 and again < 0.5, (first, again)


def test_display_limit_reports_hidden_lines(tmp_path):
    p = tmp_path / "huge.log"
    p.write_text("".join(f"{i}\n" for i in range(100)))
    r = termlog.read_log(p, max_lines=30)
    assert r["hidden_lines"] == 70 and r["text"].splitlines() == [str(i) for i in range(70, 100)]
