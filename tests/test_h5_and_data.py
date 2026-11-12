"""HDF5 summaries, GUI reads of the data store and labeling/dataset validation (synthetic data)."""
import numpy as np
import pytest

from cs_gui import h5info
from conftest import make_h5


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


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory):
    pytest.importorskip("cornerstone.data")
    from cs_gui.demo import build_demo_store
    root = tmp_path_factory.mktemp("demo") / "datastore"
    build_demo_store(root, n_harps=6, seed=1)
    return str(root)


def test_datastore_reads(demo_root):
    from cs_gui import datastore
    from cs_gui.demo import is_demo_store
    assert is_demo_store(demo_root)
    st = datastore.status(demo_root)
    assert set(st["key"]) == {"sharp/DEF", "sharp/NRT", "goes/swpc_events", "harp_noaa"}
    idx = datastore.harp_index(demo_root, "DEF")
    assert idx["HARPNUM"].tolist() == list(range(1000, 1006)) and idx["start"].dt.tz is None
    cols = datastore.sharp_columns(demo_root, "DEF")
    assert "USFLUX_N1" in cols and "MEANGBZ_N2" in cols and "T_OBS" not in cols
    s = datastore.harp_series(demo_root, "DEF", 1000, ["USFLUX_N1", "R_VALUE"])
    assert list(s.columns) == ["time", "USFLUX_N1", "R_VALUE"] and s["time"].is_monotonic_increasing
    raw = datastore.harp_series(demo_root, "DEF", 1000, ["USFLUX", "CMASK"])
    assert np.allclose(s["USFLUX_N1"], raw["USFLUX"] / raw["CMASK"], rtol=1e-5)


def test_label_and_dataset_validation(demo_root, tmp_path):
    from cs_gui.pipeline import default_label_config, label_key, quality_preset, validate_dataset, validate_labels
    from cs_gui.settings import Settings
    s = Settings(data_root=demo_root, workspace=str(tmp_path / "ws"), mode="real")
    cfg = default_label_config("NRT", "M")
    assert validate_labels(cfg, s) == [] and label_key(cfg) == "Mplus_delta24"
    assert validate_labels({**cfg, "threshold": "M5", "strat_threshold": 3e-6, "strat_window_hours": 24.0}, s) == []
    bad = validate_labels({**cfg, "threshold": "Q1", "flare_window_hours": 0, "series": "XYZ"}, s)
    assert any("threshold" in e for e in bad) and any("flare_window_hours" in e for e in bad) \
        and any("series" in e for e in bad)
    assert any("start must be before end" in e for e in validate_labels({**cfg, "start": "2015-01-01", "end": "2014-01-01"}, s))
    assert quality_preset("legacy_clean", "NRT")["quality_values"] == [0x400]
    empty = Settings(data_root=str(tmp_path / "none"), workspace=str(tmp_path / "ws2"), mode="real")
    assert any("No SHARP NRT data" in e for e in validate_labels(cfg, empty))
    lab = {"config": cfg, "summary": {"n_rows": 100}}
    ds = {"features": "clean", "quality_source": "feature_set", "window_hours": 24, "sampling_minutes": 12,
          "subsample": False, "subsample_size": 0, "random_seed": 1}
    assert validate_dataset(ds, lab, s) == []
    assert any("multiple" in e for e in validate_dataset({**ds, "sampling_minutes": 7}, lab, s))
    odd = {"config": {**cfg, "step_hours": 0.5}, "summary": {"n_rows": 100}}
    assert any("grid" in e for e in validate_dataset({**ds, "sampling_minutes": 20}, odd, s))
