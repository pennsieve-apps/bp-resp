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


def _breathing(fs, dur, f_resp):
    import neurokit2 as nk

    t = np.arange(int(dur * fs)) / fs
    clean = np.asarray(nk.rsp_clean(np.sin(2 * np.pi * f_resp * t - np.pi / 2), sampling_rate=fs))
    _, info = nk.rsp_peaks(clean, sampling_rate=fs)
    return RA.RespSegment(0.0, fs, clean, np.asarray(info["RSP_Peaks"]), np.asarray(info["RSP_Troughs"]))


def test_rsa_matches_neurokit_on_a_fine_grid():
    """RSA here is NeuroKit2's hrv_rsa rule at exact R-peak times: on a 1 kHz grid (where NeuroKit2's
    rounding is negligible) NeuroKit2 gives the same peak-to-trough and Porges-Bohrer values."""
    import neurokit2 as nk
    import pandas as pd

    fs, dur = 1000.0, 300.0
    seg = _breathing(fs, dur, 0.25)
    r, tb = [], 0.5
    while tb < dur - 1:
        r.append(tb)
        tb += 0.85 - 0.04 * np.sin(2 * np.pi * 0.25 * tb) + 0.01 * np.sin(2 * np.pi * 0.07 * tb)
    r = np.asarray(r)
    n = seg.clean.size
    ecg = pd.DataFrame({"ECG_R_Peaks": np.zeros(n, np.int8)})
    ecg.loc[np.round(r * fs).astype(int), "ECG_R_Peaks"] = 1
    rsp = pd.concat([pd.DataFrame({"RSP_Clean": seg.clean}), nk.rsp_phase(seg.peaks, seg.troughs, desired_length=n)],
                    axis=1)
    rsp["RSP_Peaks"] = 0
    rsp["RSP_Troughs"] = 0
    rsp.loc[seg.peaks, "RSP_Peaks"] = 1
    rsp.loc[seg.troughs, "RSP_Troughs"] = 1
    theirs = nk.hrv_rsa(ecg, rsp, sampling_rate=fs, continuous=False)
    ours = RA.stretch_rsa(seg, 0, n, r, True, 0.75)
    assert abs(ours["p2t"][0] - theirs["RSA_P2T_Mean"]) < 0.5
    assert abs(ours["porges-bohrer"] - theirs["RSA_PorgesBohrer"]) < 0.1


def test_zero_rsa_rat_gives_zero_p2t():
    """A rat whose heart period does not change with breathing has no RSA: peak-to-trough must stay near
    0 ms (rounding R-peaks to a 100 Hz respiration grid gave about 5 ms)."""
    rng = np.random.default_rng(1)
    seg = _breathing(100.0, 120.0, 1.5)
    r = 0.3 + np.cumsum(np.full(700, 0.171) + 0.0002 * rng.standard_normal(700))
    v = RA.stretch_rsa(seg, 0, seg.clean.size, r, False, min(0.75, 0.25 / 1.5))["p2t"][0]
    assert v < 1.0


def test_level_steps_flagged():
    from processor import bp as B

    n = 400
    t = np.arange(n) * 0.8
    sbp = np.where(t < 160, 120.0, 140.0) + 0.5 * np.sin(t)
    bt = B.BpBeats(t - 0.2, t, sbp, sbp - 40, sbp - 25, np.full(n, 40.0), np.full(n, 0.8), np.full(n, "N"),
                   np.full(n, ""), np.zeros(n, int))
    steps = B.level_steps(bt)
    assert len(steps) == 1 and abs(steps[0][0] - 160) < 3 and abs(steps[0][1] - 20) < 2


def test_raw_channel_preferred_over_derived():
    from processor import inputs as I

    raw = I.ChannelInfo("bp", "bp", "mmHg", True, "noisy", "", "ABP", "name", "")
    der = I.ChannelInfo("bp-2", "bp", "mmHg", True, "good", "", "ABP filtered", "name", "bp")
    rec = I.Recording("r", Path("x.nwb"), channels=[der, raw])
    ch, note = I.pick_channel(rec, "bp")
    assert ch.name == "bp" and "processed copy" in note
    rec = I.Recording("r", Path("x.nwb"), channels=[der])
    ch, note = I.pick_channel(rec, "bp")
    assert ch.name == "bp-2" and "only a processed copy" in note


def test_no_notice_without_uncalibrated(fixture, tmp_path, monkeypatch):
    sub = tmp_path / "in"
    (sub / "physio").mkdir(parents=True)
    import shutil
    rows_ = rows(fixture / "physio" / "recordings.csv")
    keep = [r for r in rows_ if r["recording_id"] == "sub-H02_rec-nobp"]
    with open(sub / "physio" / "recordings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keep[0]))
        w.writeheader()
        w.writerows(keep)
    shutil.copytree(fixture / "physio" / "sub-H02", sub / "physio" / "sub-H02")
    assert run(monkeypatch, sub, tmp_path / "out", FIGURES="false") == 0
    assert "notice" not in json.loads((tmp_path / "out" / "bp-resp" / "summary.json").read_text())
