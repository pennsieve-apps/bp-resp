"""
A small synthetic input for the tests, `make run` and the release review (app.yml `review`), laid out
as the upstream steps write their ports (conventions.md), with known beats, pressures, breaths and
baroreflex gain:

  <out>/physio/   physio-import's port: recordings.csv, channels.csv, events.tsv (raw markers, which
                  bp-resp must NOT read), sub-*/<recording_id>.nwb
    sub-H01_rec-rest     human, 480 s: ECG (500 Hz) with known R-peaks; arterial pressure (mmHg, 250 Hz)
                         whose systolic pressure swings at 0.1 Hz (5 mmHg) and with breathing (3 mmHg),
                         every RR interval = 850 ms + 12 ms/mmHg x (SBP - 120) (a planted baroreflex
                         gain of 12 ms/mmHg at lag 0), DBP = SBP - 40; respiration at 15 breaths/min.
    sub-R01_rec-rest     rat, 150 s at 1 kHz: ECG, pressure (SBP 115 mmHg swinging 6 mmHg at 0.4 Hz,
                         RR = 171 ms + 3 ms/mmHg x (SBP - 115)), respiration at 90 breaths/min.
    sub-H02_rec-nobp     human, 120 s: ECG and respiration (12/min), no pressure ("missing signals").
    sub-H03_rec-ecgonly  human, 60 s: ECG only (skipped: no pressure and no respiration).
    sub-H04_rec-uncal    human, 120 s: uncalibrated pressure (a.u., calibrated=false) and respiration,
                         no ECG (pulse timing only, no pressures, no baroreflex, no RSA).
  <out>/events/stim-events/events.tsv   one stimulation train in sub-H01 at 200-210 s (stim-events' port)
  <out>/hrv/ecg-hrv/beats.csv           sub-H02's true R-peaks as ecg-hrv writes them (ecg-hrv's port)
  <out>/truth.json

The NWB files are written with pynwb when it is installed, else with h5py in the same layout.

Usage: python -m tests.make_fixture <out-dir> [--writer pynwb|h5py]
"""
from __future__ import annotations

import csv
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 42


def ecg(t: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """A simple ECG (V): a narrow R wave (8 ms) and a T wave at each beat."""
    x = np.zeros_like(t)
    for b in beats:
        i0, i1 = np.searchsorted(t, [b - 0.05, b + 0.4])
        tt = t[i0:i1] - b
        x[i0:i1] += 1e-3 * np.exp(-0.5 * (tt / 0.008) ** 2) + 2.5e-4 * np.exp(-0.5 * ((tt - 0.25) / 0.04) ** 2)
    return x


def pressure(t: np.ndarray, feet: np.ndarray, sys_t: np.ndarray, sbp: np.ndarray, dbp: np.ndarray) -> np.ndarray:
    """An arterial pressure wave: from each foot (DBP) a half-cosine rise to the systolic peak (SBP),
    then an exponential-like fall to the next foot's DBP."""
    x = np.full_like(t, dbp[0])
    for k in range(feet.size - 1):
        a, p, b = feet[k], sys_t[k], feet[k + 1]
        m1 = (t >= a) & (t < p)
        u = (t[m1] - a) / (p - a)
        x[m1] = dbp[k] + (sbp[k] - dbp[k]) * 0.5 * (1 - np.cos(np.pi * u))
        m2 = (t >= p) & (t < b)
        v = (t[m2] - p) / (b - p)
        x[m2] = dbp[k + 1] + (sbp[k] - dbp[k + 1]) * (np.exp(-3 * v) - np.exp(-3)) / (1 - np.exp(-3))
    return x



class Rec:
    def __init__(self, rid, subject, species, latin, source_path, source_format):
        self.rid, self.subject, self.species, self.latin = rid, subject, species, latin
        self.source_path, self.source_format = source_path, source_format
        self.series = []        # (name, data, unit, rate or None, timestamps or None, description, conversion)
        self.intervals = {}     # name -> (description, {col: list}) with start_time, stop_time
        self.channels = []      # channels.csv rows

    def add(self, name, data, unit, rate=None, timestamps=None, typ="other", original=None, site="",
            calibrated=True, conversion=1.0):
        self.series.append((name, data, unit, rate, timestamps, f"{original or name} (typing: name)", conversion))
        self.channels.append({"recording_id": self.rid, "channel": name, "type": typ, "original_name": original or name,
                              "unit": unit, "sampling_rate_hz": rate or round(1 / np.median(np.diff(timestamps))),
                              "n_samples": len(data), "calibrated": "true" if calibrated else "false",
                              "scale": 1, "site": site, "typing": "name", "status": "good",
                              "status_description": ""})

    def duration(self) -> float:
        end = 0.0
        for _, d, _, rate, ts, _, _ in self.series:
            end = max(end, (ts[-1] + 1.0 / 1000) if ts is not None else len(d) / rate)
        return end


def write_pynwb(rec: Rec, path: Path) -> None:
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    nwb = NWBFile(session_description=f"synthetic {rec.rid}", identifier=rec.rid,
                  session_start_time=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
                  subject=Subject(subject_id=rec.subject, species=rec.latin))
    for name, data, unit, rate, ts, desc, conv in rec.series:
        kw = dict(rate=float(rate), starting_time=0.0) if ts is None else dict(timestamps=H5DataIO(ts, compression="gzip"))
        nwb.add_acquisition(TimeSeries(name=name, data=H5DataIO(data, compression="gzip"), unit=unit,
                                       conversion=conv, description=desc, **kw))
    for name, (desc, cols) in rec.intervals.items():
        ti = TimeIntervals(name=name, description=desc)
        for c in cols:
            if c not in ("start_time", "stop_time"):
                ti.add_column(name=c, description=c)
        for i in range(len(cols["start_time"])):
            ti.add_row(**{c: v[i] for c, v in cols.items()})
        nwb.add_time_intervals(ti)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def write_h5py(rec: Rec, path: Path) -> None:
    """The same layout with h5py only (NWB 2.x group/attribute names; enough for readers of the layout)."""
    import h5py

    def attrs(obj, ntype, **kw):
        obj.attrs["neurodata_type"] = ntype
        obj.attrs["namespace"] = "core"
        obj.attrs["object_id"] = str(uuid.uuid4())
        for k, v in kw.items():
            obj.attrs[k] = v

    with h5py.File(path, "w") as f:
        attrs(f, "NWBFile", nwb_version="2.9.0")
        f["identifier"] = rec.rid
        f["session_description"] = f"synthetic {rec.rid}"
        f["session_start_time"] = "2026-01-01T09:00:00+00:00"
        f["timestamps_reference_time"] = "2026-01-01T09:00:00+00:00"
        f.create_dataset("file_create_date", data=[datetime.now(timezone.utc).isoformat()],
                         dtype=h5py.string_dtype())
        for g in ("acquisition", "analysis", "processing", "stimulus/presentation", "stimulus/templates", "intervals"):
            f.require_group(g)
        sub = f.require_group("general/subject")
        attrs(sub, "Subject")
        sub["subject_id"] = rec.subject
        sub["species"] = rec.latin
        for name, data, unit, rate, ts, desc, conv in rec.series:
            g = f["acquisition"].create_group(name)
            attrs(g, "TimeSeries", description=desc, comments="no comments")
            d = g.create_dataset("data", data=data, compression="gzip")
            d.attrs.update(unit=unit, conversion=conv, offset=0.0, resolution=-1.0)
            if ts is None:
                s = g.create_dataset("starting_time", data=0.0)
                s.attrs.update(rate=float(rate), unit="seconds")
            else:
                s = g.create_dataset("timestamps", data=ts, compression="gzip")
                s.attrs.update(interval=1, unit="seconds")
        for name, (desc, cols) in rec.intervals.items():
            g = f["intervals"].create_group(name)
            attrs(g, "TimeIntervals", description=desc, colnames=[c for c in cols])
            n = len(cols["start_time"])
            g.create_dataset("id", data=np.arange(n))
            for c, v in cols.items():
                if isinstance(v[0], str):
                    g.create_dataset(c, data=v, dtype=h5py.string_dtype())
                else:
                    g.create_dataset(c, data=np.asarray(v))




# --------------------------------------------------------------------------- recordings

def baro_beats(dur: float, rr0: float, gain: float, sbp0: float, comps: list, rng, t0: float = 0.4,
               noise_sbp: float = 0.3, noise_rr: float = 0.002) -> dict:
    """R-peaks r_k whose interval r_k -> r_k+1 = rr0 + gain (SBP_k - sbp0) [s, s/mmHg], with SBP_k a sum
    of sines (amplitude mmHg, frequency Hz) at the systolic time r_k + 0.25 rr0."""
    r, sbp = [t0], []
    while r[-1] < dur:
        ts = r[-1] + 0.3 * rr0
        s = sbp0 + sum(a * np.sin(2 * np.pi * f * ts) for a, f in comps) + noise_sbp * rng.standard_normal()
        sbp.append(s)
        r.append(r[-1] + rr0 + gain * (s - sbp0) + noise_rr * rng.standard_normal())
    r = np.asarray(r)
    sbp = np.asarray(sbp)
    return {"r": r, "sbp": sbp, "sys_t": r[:-1] + 0.3 * rr0, "feet": r[:-1] + 0.12 * rr0}


def add_bp(rec: Rec, b: dict, fs: float, pp: float, dur: float, unit="mmHg", calibrated=True, scale=1.0):
    t = np.arange(int(dur * fs)) / fs
    dbp = b["sbp"] - pp
    x = pressure(t, b["feet"], b["sys_t"], b["sbp"], dbp)
    rec.add("bp", (x * scale).astype(np.float32), unit, fs, typ="bp", original="ABP" if calibrated else "BP",
            calibrated=calibrated)
    return t, x


def rec_h01(truth: dict, rng) -> Rec:
    dur = 480.0
    b = baro_beats(dur, 0.85, 0.012, 120.0, [(5.0, 0.1), (3.0, 0.25)], rng)
    r = Rec("sub-H01_rec-rest", "H01", "human", "Homo sapiens", "raw/H01.edf", "edf")
    fs = 500.0
    t = np.arange(int(dur * fs)) / fs
    r.add("ecg", (ecg(t, b["r"]) + 2e-5 * rng.standard_normal(t.size)).astype(np.float32), "V", fs, typ="ecg",
          original="ECG")
    add_bp(r, b, 250.0, 40.0, dur)
    tr = np.arange(int(dur * 50)) / 50.0
    r.add("resp", (0.5 * np.sin(2 * np.pi * 0.25 * tr - np.pi / 2)).astype(np.float32), "a.u.", 50.0, typ="resp",
          original="Resp belt")
    m = b["sys_t"] < dur - 1
    truth["sub-H01_rec-rest"] = {"brs_ms_per_mmhg": 12.0, "sbp_mean": float(np.mean(b["sbp"][m])),
                                 "dbp_mean": float(np.mean(b["sbp"][m] - 40.0)), "n_beats": int(m.sum()),
                                 "hr_bpm": 60.0 / 0.85, "resp_rate_bpm": 15.0, "resp_amplitude": 1.0,
                                 "stimulation": [200.0, 10.0]}
    return r


def rec_r01(truth: dict, rng) -> Rec:
    dur = 150.0
    b = baro_beats(dur, 0.171, 0.003, 115.0, [(6.0, 0.4)], rng, noise_sbp=0.2, noise_rr=0.0003)
    r = Rec("sub-R01_rec-rest", "R01", "rat", "Rattus norvegicus", "raw/R01.abf", "abf")
    fs = 1000.0
    t = np.arange(int(dur * fs)) / fs
    beats = b["r"]
    x = np.zeros_like(t)
    for bb in beats:      # a narrower rat QRS (3 ms)
        i0, i1 = np.searchsorted(t, [bb - 0.02, bb + 0.1])
        tt = t[i0:i1] - bb
        x[i0:i1] += 1e-3 * np.exp(-0.5 * (tt / 0.003) ** 2) + 2e-4 * np.exp(-0.5 * ((tt - 0.05) / 0.01) ** 2)
    r.add("ecg", (x + 2e-5 * rng.standard_normal(t.size)).astype(np.float32), "V", fs, typ="ecg", original="ECG")
    add_bp(r, b, fs, 35.0, dur)
    r.add("resp", (0.3 * np.sin(2 * np.pi * 1.5 * t)).astype(np.float32), "a.u.", fs, typ="resp", original="Pleth")
    m = b["sys_t"] < dur - 1
    truth["sub-R01_rec-rest"] = {"brs_ms_per_mmhg": 3.0, "sbp_mean": float(np.mean(b["sbp"][m])),
                                 "hr_bpm": 60.0 / 0.171, "resp_rate_bpm": 90.0}
    return r


def rec_h02(truth: dict, rng) -> Rec:
    dur = 120.0
    fs = 500.0
    t = np.arange(int(dur * fs)) / fs
    # RR shortens in inspiration and lengthens in expiration (RSA, 60 ms swing) with breathing at 0.2 Hz
    beats, tb = [], 0.5
    while tb < dur:
        beats.append(tb)
        tb += 0.9 - 0.03 * np.sin(2 * np.pi * 0.2 * tb)
    beats = np.asarray(beats)
    r = Rec("sub-H02_rec-nobp", "H02", "human", "Homo sapiens", "raw/H02.edf", "edf")
    r.add("ecg", (ecg(t, beats) + 2e-5 * rng.standard_normal(t.size)).astype(np.float32), "V", fs, typ="ecg",
          original="ECG")
    r.add("resp", (np.sin(2 * np.pi * 0.2 * t - np.pi / 2)).astype(np.float32), "a.u.", fs, typ="resp",
          original="Thorax")
    truth["sub-H02_rec-nobp"] = {"resp_rate_bpm": 12.0, "r_peaks": beats.tolist()}
    return r


def rec_h03(truth: dict, rng) -> Rec:
    dur, fs = 60.0, 500.0
    t = np.arange(int(dur * fs)) / fs
    r = Rec("sub-H03_rec-ecgonly", "H03", "human", "Homo sapiens", "raw/H03.edf", "edf")
    r.add("ecg", ecg(t, np.arange(0.5, dur, 0.8)).astype(np.float32), "V", fs, typ="ecg", original="ECG")
    truth["sub-H03_rec-ecgonly"] = {"status": "skipped"}
    return r


def rec_h04(truth: dict, rng) -> Rec:
    dur = 120.0
    b = baro_beats(dur, 1.0, 0.0, 120.0, [(4.0, 0.1)], rng)
    r = Rec("sub-H04_rec-uncal", "H04", "human", "Homo sapiens", "raw/H04.dat", "wfdb")
    add_bp(r, b, 250.0, 40.0, dur, unit="a.u.", calibrated=False, scale=0.01)
    t = np.arange(int(dur * 250)) / 250.0
    r.add("resp", (np.sin(2 * np.pi * 0.25 * t)).astype(np.float32), "a.u.", 250.0, typ="resp", original="RESP")
    m = b["sys_t"] < dur - 1
    truth["sub-H04_rec-uncal"] = {"n_beats": int(m.sum()), "hr_bpm": 60.0, "calibrated": False}
    return r


def main(out, writer: str | None = None) -> Path:
    out = Path(out)
    rng = np.random.default_rng(SEED)
    if writer is None:
        try:
            import pynwb  # noqa: F401
            writer = "pynwb"
        except ImportError:
            writer = "h5py"
    write = write_pynwb if writer == "pynwb" else write_h5py
    physio = out / "physio"
    physio.mkdir(parents=True, exist_ok=True)
    truth: dict = {"writer": writer}
    recs = [rec_h01(truth, rng), rec_r01(truth, rng), rec_h02(truth, rng), rec_h03(truth, rng), rec_h04(truth, rng)]
    rec_rows, ch_rows = [], []
    for r in recs:
        rel = Path(f"sub-{r.subject}") / f"{r.rid}.nwb"
        (physio / rel.parent).mkdir(parents=True, exist_ok=True)
        write(r, physio / rel)
        types = sorted({c["type"] for c in r.channels})
        rec_rows.append({"recording_id": r.rid, "subject": r.subject, "session": "", "label": r.rid.split("rec-")[1],
                         "path": str(rel), "species": r.species, "start_time": "2026-01-01T09:00:00",
                         "duration_s": round(r.duration(), 3), "recorded_s": round(r.duration(), 3),
                         "n_segments": 1, "channel_types": "|".join(types), "source_format": r.source_format,
                         "source_path": r.source_path, "metadata_source": "file", "age": "", "sex": ""})
        ch_rows.extend(r.channels)
    for name, rows in (("recordings.csv", rec_rows), ("channels.csv", ch_rows)):
        with open(physio / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    cols = ["recording_id", "onset", "duration", "trial_type", "event_id", "amplitude", "amplitude_unit",
            "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s", "train_off_s", "n_pulses", "polarity",
            "waveform", "site", "contact", "condition", "label", "source"]
    # physio-import's raw markers: a "stimulation" row bp-resp must ignore (it reads stim-events' file only)
    with open(physio / "events.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        w.writerow(["sub-H01_rec-rest", 50, 10, "stimulation", "m-0001"] + ["n/a"] * 13 + ["VNS on", "marker"])
    (physio / "provenance.json").write_text(json.dumps({"app": "physio-import", "version": "synthetic"}))
    ev = out / "events" / "stim-events"
    ev.mkdir(parents=True, exist_ok=True)
    with open(ev / "events.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        w.writerow(["sub-H01_rec-rest", 200, 10, "stimulation", "stim-0001", 1.0, "mA", 250, 20, 10, "n/a", "n/a",
                    200, "cathodic", "biphasic", "n/a", "n/a", "n/a", "n/a", "trigger"])
    hv = out / "hrv" / "ecg-hrv"
    hv.mkdir(parents=True, exist_ok=True)
    with open(hv / "beats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "beat", "time_s", "rr_ms", "label", "detector"])
        prev = None
        for i, tb in enumerate(truth["sub-H02_rec-nobp"]["r_peaks"], 1):
            w.writerow(["sub-H02_rec-nobp", i, round(tb, 4), "" if prev is None else round((tb - prev) * 1000, 2),
                        "N", "neurokit"])
            prev = tb
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    w = None
    if "--writer" in args:
        w = args[args.index("--writer") + 1]
    main(args[0], w)
    print(f"wrote {args[0]}")
