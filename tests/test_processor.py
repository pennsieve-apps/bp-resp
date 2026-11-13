"""The synthetic fixture through the processor (no Docker): planted beats, pressures, breaths and
baroreflex gains are recovered, and the missing-signal and uncalibrated paths behave as conventions.md says."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from processor import main as M
from processor import rsa as RA
from tests import make_fixture

WRITERS = ["h5py"]
try:
    import pynwb  # noqa: F401
    WRITERS.insert(0, "pynwb")
except ImportError:
    pass


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(monkeypatch, input_dir: Path, output_dir: Path, **env) -> int:
    for k in M.DEFAULTS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session", params=WRITERS)
def fixture(request, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp(f"fixture-{request.param}")
    make_fixture.main(d, request.param)
    return d


@pytest.fixture(scope="session")
def truth(fixture) -> dict:
    return json.loads((fixture / "truth.json").read_text())


@pytest.fixture(scope="session")
def out(fixture, tmp_path_factory) -> Path:
    o = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture, o) == 0
    finally:
        mp.undo()
    return o / "bp-resp"


def brs(out, rid, method) -> dict:
    return next(r for r in rows(out / "brs.csv") if r["recording_id"] == rid and r["method"] == method)


def test_outputs_and_headers(out):
    for name, head in M.HEAD.items():
        with open(out / f"{name}.csv", newline="") as fh:
            assert next(csv.reader(fh)) == head
    prov = json.loads((out / "provenance.json").read_text())
    assert set(prov) >= {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                         "wall_seconds", "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"}
    assert prov["recordings"]["sub-H03_rec-ecgonly"]["status"] == "skipped"
    assert "no blood pressure or respiration" in prov["recordings"]["sub-H03_rec-ecgonly"]["reason"]
    assert prov["recordings"]["sub-R01_rec-rest"]["species"] == "rat"
    summ = json.loads((out / "summary.json").read_text())
    assert summ["figures"] and all((out / f["path"]).exists() and f["caption"] for f in summ["figures"])
    assert "sub-H04_rec-uncal" in summ["notice"]


def test_human_pressure_beats(out, truth):
    t = truth["sub-H01_rec-rest"]
    b = [r for r in rows(out / "bp_beats.csv") if r["recording_id"] == "sub-H01_rec-rest" and r["label"] == "N"]
    assert abs(len(b) - t["n_beats"]) <= 3
    sbp = np.array([float(r["sbp_mmhg"]) for r in b])
    dbp = np.array([float(r["dbp_mmhg"]) for r in b])
    mp = np.array([float(r["map_mmhg"]) for r in b])
    assert abs(sbp.mean() - t["sbp_mean"]) < 1.0
    assert abs(dbp.mean() - t["dbp_mean"]) < 1.0
    assert np.all((mp > dbp) & (mp < sbp))
    # the integral mean of this waveform is not the (2D+S)/3 rule
    assert abs(mp.mean() - (2 * dbp.mean() + sbp.mean()) / 3) > 0.5
    ibi = np.array([float(r["ibi_ms"]) for r in b])
    assert abs(60000.0 / np.median(ibi) - t["hr_bpm"]) < 2


def test_planted_baroreflex_is_recovered(out):
    for rid, gain in (("sub-H01_rec-rest", 12.0), ("sub-R01_rec-rest", 3.0)):
        r = brs(out, rid, "sequence-all")
        assert r["valid"] == "true" and int(r["n"]) >= 10
        assert abs(float(r["value_ms_per_mmhg"]) - gain) < 0.1 * gain, rid
        assert r["lag_beats"] == "0"
        a = brs(out, rid, "alpha-lf")
        assert a["valid"] == "true" and abs(float(a["value_ms_per_mmhg"]) - gain) < 0.15 * gain
        tf = brs(out, rid, "transfer-lf")
        assert tf["valid"] == "true" and abs(float(tf["value_ms_per_mmhg"]) - gain) < 0.2 * gain


def test_breaths_and_species_preset(out, truth):
    br = rows(out / "breaths.csv")
    for rid in ("sub-H01_rec-rest", "sub-R01_rec-rest", "sub-H02_rec-nobp"):
        rate = np.median([float(r["rate_bpm"]) for r in br if r["recording_id"] == rid and r["label"] == "N"])
        assert abs(rate - truth[rid]["resp_rate_bpm"]) < 0.03 * truth[rid]["resp_rate_bpm"], rid
    amp = np.median([float(r["amplitude"]) for r in br if r["recording_id"] == "sub-H01_rec-rest"])
    assert abs(amp - truth["sub-H01_rec-rest"]["resp_amplitude"]) < 0.05


def test_rsa(out):
    r = {(x["recording_id"], x["method"]): x for x in rows(out / "rsa.csv")}
    # H02: RR swings by +-30 ms with breathing (inspiration short): peak-to-trough about 60 ms
    assert 45 < float(r[("sub-H02_rec-nobp", "p2t")]["value"]) < 75
    assert "ecg-hrv" in r[("sub-H02_rec-nobp", "p2t")]["note"]       # beats from the hrv port
    assert r[("sub-H02_rec-nobp", "porges-bohrer")]["value"]
    # rat: no Porges-Bohrer (human band), peak-to-trough with a scaled delay
    assert r[("sub-R01_rec-rest", "porges-bohrer")]["value"] == ""
    assert "human" in r[("sub-R01_rec-rest", "porges-bohrer")]["note"]


def test_missing_and_uncalibrated(out):
    # no BP: BRS rows empty with the reason; breaths and RSA still written
    r = brs(out, "sub-H02_rec-nobp", "sequence-all")
    assert r["value_ms_per_mmhg"] == "" and r["valid"] == "false" and "no blood pressure" in r["note"]
    # uncalibrated: beats with timing only, no pressures, no mmHg series, no BRS
    b = [x for x in rows(out / "bp_beats.csv") if x["recording_id"] == "sub-H04_rec-uncal"]
    assert len(b) >= 110 and all(x["sbp_mmhg"] == "" and x["calibrated"] == "false" for x in b)
    assert all(x["time_s"] for x in b)
    r = brs(out, "sub-H04_rec-uncal", "alpha-lf")
    assert r["value_ms_per_mmhg"] == "" and "uncalibrated" in r["note"]
    s = rows(out / "series.csv")
    assert not any(x["recording_id"] == "sub-H04_rec-uncal" and x["variable"].endswith("mmhg") for x in s)
    assert any(x["recording_id"] == "sub-H04_rec-uncal" and x["variable"] == "resp_rate_bpm" for x in s)
    # skipped recording: nothing in the tables
    for name in ("bp_beats", "breaths", "brs", "rsa", "series"):
        assert not any(x["recording_id"] == "sub-H03_rec-ecgonly" for x in rows(out / f"{name}.csv"))


def test_series_grid(out):
    s = [x for x in rows(out / "series.csv") if x["recording_id"] == "sub-H01_rec-rest" and x["variable"] == "sbp_mmhg"]
    t = np.array([float(x["time_s"]) for x in s])
    assert np.allclose(np.diff(t), 0.25)          # human SERIES_HZ 4
    valid = [x for x in s if x["valid"] == "true"]
    assert len(valid) > 0.9 * len(s)
    r = [x for x in rows(out / "series.csv") if x["recording_id"] == "sub-R01_rec-rest" and x["variable"] == "map_mmhg"]
    assert np.allclose(np.diff([float(x["time_s"]) for x in r]), 0.1)   # rat SERIES_HZ 10
    assert {x["variable"] for x in rows(out / "series.csv")} == {"sbp_mmhg", "dbp_mmhg", "map_mmhg", "pp_mmhg",
                                                                 "resp_rate_bpm", "resp_amp"}


def test_stimulation_from_stim_events_only(out):
    qc = rows(out / "qc.csv")
    ex = [x for x in qc if x["recording_id"] == "sub-H01_rec-rest" and x["check"] == "stimulation_excluded"]
    assert len(ex) == 1 and ex[0]["value"] == "1"      # stim-events' one train, not physio-import's marker
    prov = json.loads((out / "provenance.json").read_text())
    assert "events/stim-events/events.tsv" in prov["inputs"]
    assert "physio/events.tsv" not in prov["inputs"]


def test_features(out):
    f = {(x["recording_id"], x["feature"]): x["value"] for x in rows(out / "features.csv")}
    assert abs(float(f[("sub-H01_rec-rest", "brs_sequence_all")]) - 12) < 1.2
    assert ("sub-H01_rec-rest", "bp_sbp_mean_mmhg") in f
    assert ("sub-H04_rec-uncal", "bp_sbp_mean_mmhg") not in f


def test_rr_source_bp_pulse(fixture, tmp_path, monkeypatch):
    """RR_SOURCE=bp: the baroreflex from the pulse interval of the pressure itself."""
    assert run(monkeypatch, fixture, tmp_path, RR_SOURCE="bp", FIGURES="false") == 0
    r = brs(tmp_path / "bp-resp", "sub-H01_rec-rest", "sequence-all")
    assert r["rr_source"] == "bp-pulse" and r["valid"] == "true"
    # pulse intervals of synthetic beats placed at a fixed fraction of the RR interval track RR closely
    assert abs(float(r["value_ms_per_mmhg"]) - 12) < 3


def test_species_override_and_bad_parameters(fixture, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture, tmp_path, SEQ_LAG="7") == 2
    assert run(monkeypatch, fixture, tmp_path, SPECIES="cat") == 2
    assert run(monkeypatch, fixture, tmp_path, LF_BAND="0.2-0.1") == 2
    assert run(monkeypatch, fixture, tmp_path / "x", SEQ_SBP_MMHG="0.5", SEQ_LAG="1", FIGURES="false") == 0
    r = brs(tmp_path / "x" / "bp-resp", "sub-H01_rec-rest", "sequence-all")
    assert r["lag_beats"] == "1"


def test_no_input(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out") == 1


def test_inhouse_p2t_matches_neurokit(fixture):
    """The in-house peak-to-trough (used for non-human species) is NeuroKit2's rule: with NeuroKit2's
    750 ms delay it gives NeuroKit2's value."""
    import neurokit2 as nk

    fs = 50.0
    n = int(120 * fs)
    t = np.arange(n) / fs
    rsp = np.sin(2 * np.pi * 0.2 * t - np.pi / 2)
    clean = nk.rsp_clean(rsp, sampling_rate=fs)
    _, info = nk.rsp_peaks(clean, sampling_rate=fs)
    rp, tb = [], 0.5
    while tb < 119:
        rp.append(int(round(tb * fs)))
        tb += 0.9 - 0.03 * np.sin(2 * np.pi * 0.2 * tb)
    rp = np.asarray(rp)
    seg = RA.RespSegment(0.0, fs, np.asarray(clean), np.asarray(info["RSP_Peaks"]), np.asarray(info["RSP_Troughs"]))
    ours = RA.stretch_rsa(seg, 0, n, rp / fs, False, 0.75)["p2t"][0]
    theirs = RA.stretch_rsa(seg, 0, n, rp / fs, True, 0.75)["p2t"][0]
    assert abs(ours - theirs) < 1e-6
