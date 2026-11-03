"""Adapters that need only the standard library + HDF5 summaries."""
import sqlite3

import numpy as np
import pytest

from cs_gui import h5info
from cs_gui.runner import adapters
from conftest import make_h5


@pytest.fixture
def main_db(tmp_path):
    p = tmp_path / "main.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE DF_DEF_Cplus_delta24_three_classes (t_X TEXT, HARPn INT, label TEXT)")
    c.execute("INSERT INTO DF_DEF_Cplus_delta24_three_classes VALUES ('2012-01-01T00:00:00', 1, 'NO1')")
    c.execute("CREATE TABLE SHARPC_all_DEF (T_OBS TEXT, HARPNUM INT, USFLUX REAL)")
    c.executemany("INSERT INTO SHARPC_all_DEF VALUES (?, ?, ?)",
                  [("2012.01.01_00:00:00_TAI", 1, 1.0), ("2012.01.03_00:00:00_TAI", 1, 2.0),
                   ("2012.01.02_00:00:00_TAI", 1, 3.0), ("2012.02.01_00:00:00_TAI", 2, 4.0)])
    c.execute("CREATE TABLE NOAA_SHARPC_DEF_AR (tmax TEXT, fflux REAL, HARPNUM INT)")
    c.execute("INSERT INTO NOAA_SHARPC_DEF_AR VALUES ('2012-01-02T00:00:00', 2e-6, 1)")
    c.commit()
    c.close()
    return p


def test_readonly_shim_blocks_writes_and_shadows_labels(main_db, tmp_path):
    lab = tmp_path / "labels.sqlite"
    c = sqlite3.connect(lab)
    c.execute("CREATE TABLE DF_DEF_Cplus_delta12_three_classes (t_X TEXT, HARPn INT, label TEXT)")
    c.execute("INSERT INTO DF_DEF_Cplus_delta12_three_classes VALUES ('2013-01-01T00:00:00', 9, 'YES')")
    c.commit()
    c.close()
    shim = adapters.readonly_sqlite_module(str(main_db), str(lab),
                                           {"DF_DEF_Cplus_delta24_three_classes": "DF_DEF_Cplus_delta12_three_classes"})
    conn = shim.connect(str(main_db))
    assert conn.execute("SELECT HARPn, label FROM DF_DEF_Cplus_delta24_three_classes").fetchall() == [(9, "YES")]
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM SHARPC_all_DEF")
    assert shim.Row is sqlite3.Row


def test_labeling_db_reproduces_harp_bounds(main_db, tmp_path):
    out = tmp_path / "lab.sqlite"
    info = adapters.build_labeling_db(str(main_db), str(out), "DEF")
    assert info == {"n_flare_events": 1, "n_harps": 2}
    q = "SELECT HARPNUM, MIN(T_OBS), MAX(T_OBS) FROM SHARPC_all_DEF GROUP BY HARPNUM"
    a = sqlite3.connect(main_db).execute(q).fetchall()
    b = sqlite3.connect(out).execute(q).fetchall()
    assert a == b                                    # the query used by splitting_labeling gives the same answer


def test_snapshot_label_table(main_db, tmp_path):
    out = tmp_path / "snap.sqlite"
    assert adapters.snapshot_label_table(str(main_db), str(out), "DF_DEF_Cplus_delta24_three_classes") == 1
    with pytest.raises(RuntimeError, match="not found in the database"):
        adapters.snapshot_label_table(str(main_db), str(tmp_path / "x.sqlite"), "DF_NOPE")


def test_label_views_like_the_published_db(tmp_path):
    """Published DB: DF_*_three_classes are views on DF_*_three_classes_unified."""
    from cs_gui import db as dbq
    p = tmp_path / "pub.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE DF_NRT_Mplus_delta24_three_classes_unified (t_X TEXT, HARPn INTEGER, label TEXT)")
    c.execute("INSERT INTO DF_NRT_Mplus_delta24_three_classes_unified VALUES ('2013-01-01T00:00:00', 7, 'YES')")
    c.execute("CREATE VIEW DF_NRT_Mplus_delta24_three_classes AS SELECT * FROM DF_NRT_Mplus_delta24_three_classes_unified")
    c.commit()
    c.close()
    assert dbq.label_tables(str(p))["table"].tolist() == ["DF_NRT_Mplus_delta24_three_classes"]
    out = tmp_path / "snap.sqlite"
    assert adapters.snapshot_label_table(str(p), str(out), "DF_NRT_Mplus_delta24_three_classes") == 1
    o = sqlite3.connect(out)
    assert o.execute("SELECT type FROM sqlite_master WHERE name='DF_NRT_Mplus_delta24_three_classes'").fetchone() == ("table",)
    assert o.execute("SELECT * FROM DF_NRT_Mplus_delta24_three_classes").fetchall() == [("2013-01-01T00:00:00", 7, "YES")]


def test_h5_summary_and_keys(tmp_path):
    p = make_h5(tmp_path / "d.h5", n=12, t=5)
    s = h5info.summarize(str(p))
    assert s["n_samples"] == 12 and s["n_features"] == 3 and s["n_active_regions"] == 3
    assert s["class_counts"] == {"YES": 3, "NO1": 9} and s["empty_samples_in_sample"] == 1
    assert s["time_start"] < s["time_end"]
    keys = h5info.sample_keys(str(p))
    assert len(keys) == 12 and all(isinstance(h, int) for h, _ in keys)
    it = h5info.issue_times(np.array([[b"2012-01-01 00:00:00", b""], [b"", b""]]))
    assert it.tolist() == ["2012-01-01 00:00:00", ""]


def test_flares_of_the_published_db(tmp_path):
    """Published DB: one NOAA_SHARP_AR, associated to NRT HARPs; no flare table for DEF."""
    from cs_gui import db as dbq
    from cs_gui.pipeline import validate_labels
    from cs_gui.settings import Settings
    p = tmp_path / "pub.sqlite"
    c = sqlite3.connect(p)
    for t in ("SHARPC_all_DEF", "SHARPC_all_NRT"):
        c.execute(f"CREATE TABLE {t} (T_OBS TEXT, HARPNUM INTEGER)")
        c.execute(f"INSERT INTO {t} VALUES ('2013.01.01_00:00:00_TAI', 7)")
    c.execute("CREATE TABLE NOAA_SHARP_AR (tstart TEXT, tmax TEXT, tend TEXT, part1 TEXT, fclass TEXT, "
              "fflux REAL, HARPNUM INTEGER, NOAA_AR INTEGER)")
    c.execute("INSERT INTO NOAA_SHARP_AR VALUES ('2013-01-01T01:00:00', '2013-01-01T01:10:00', "
              "'2013-01-01T01:20:00', 'M1.0', 'M', 1e-5, 7, 11650)")
    c.commit()
    c.close()
    assert dbq.flare_table(str(p), "NRT") == "NOAA_SHARP_AR" and dbq.flare_table(str(p), "DEF") is None
    assert len(dbq.harp_flares(str(p), "NRT", 7)) == 1 and len(dbq.harp_flares(str(p), "DEF", 7)) == 0

    out = tmp_path / "lab.sqlite"
    assert adapters.build_labeling_db(str(p), str(out), "NRT")["n_flare_events"] == 1
    # copied under the name splitting_labeling reads
    assert sqlite3.connect(out).execute("SELECT part1, fflux FROM NOAA_SHARPC_NRT_AR").fetchall() == [("M1.0", 1e-5)]
    with pytest.raises(RuntimeError, match="No flare event table for DEF"):
        adapters.build_labeling_db(str(p), str(tmp_path / "lab2.sqlite"), "DEF")

    s = Settings(db_path=str(p), mode="real")
    cfg = {"source": "generate", "label_type": "Mplus", "delta_hours": 24}
    assert validate_labels({**cfg, "table_type": "NRT"}, s) == []
    assert any("No flare events for DEF" in e for e in validate_labels({**cfg, "table_type": "DEF"}, s))
