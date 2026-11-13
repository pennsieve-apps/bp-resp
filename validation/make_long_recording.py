"""
A long synthetic rodent recording for measuring memory and time: rat, ECG, arterial pressure and
respiration at 1 kHz for HOURS hours (default 6), written a 10-minute chunk at a time with h5py in
physio-import's layout (recordings.csv, channels.csv, sub-L01/<recording_id>.nwb).

  python -m validation.make_long_recording <out-dir> [hours]
"""
from __future__ import annotations

import csv
import sys
import uuid
from pathlib import Path

import h5py
import numpy as np

from tests.make_fixture import pressure

FS = 1000.0


def main(out: str, hours: float = 6.0) -> None:
    out = Path(out)
    rid = "sub-L01_rec-long"
    dur = hours * 3600.0
    n = int(dur * FS)
    rng = np.random.default_rng(7)
    # beats: RR = 171 ms + 3 ms/mmHg (SBP - 115), SBP swinging at 0.4 Hz
    r, sbp = [0.3], []
    while r[-1] < dur + 1:
        ts = r[-1] + 0.05
        s = 115 + 6 * np.sin(2 * np.pi * 0.4 * ts) + 0.2 * rng.standard_normal()
        sbp.append(s)
        r.append(r[-1] + 0.171 + 0.003 * (s - 115))
    r, sbp = np.asarray(r), np.asarray(sbp)
    feet, sys_t = r[:-1] + 0.02, r[:-1] + 0.05
    (out / "physio" / "sub-L01").mkdir(parents=True, exist_ok=True)
    path = out / "physio" / "sub-L01" / f"{rid}.nwb"
    with h5py.File(path, "w") as f:
        f.attrs.update(neurodata_type="NWBFile", namespace="core", object_id=str(uuid.uuid4()), nwb_version="2.9.0")
        f["identifier"] = rid
        f["session_description"] = "synthetic long rat recording"
        f["session_start_time"] = "2026-01-01T09:00:00+00:00"
        f["timestamps_reference_time"] = "2026-01-01T09:00:00+00:00"
        f.require_group("acquisition")
        sub = f.require_group("general/subject")
        sub["subject_id"] = "L01"
        sub["species"] = "Rattus norvegicus"
        ds = {}
        for name, unit in (("ecg", "V"), ("bp", "mmHg"), ("resp", "a.u.")):
            g = f["acquisition"].create_group(name)
            g.attrs.update(neurodata_type="TimeSeries", namespace="core", object_id=str(uuid.uuid4()),
                           description=name, comments="no comments")
            d = g.create_dataset("data", shape=(n,), dtype="float32", chunks=(65536,), compression="gzip")
            d.attrs.update(unit=unit, conversion=1.0, offset=0.0, resolution=-1.0)
            st = g.create_dataset("starting_time", data=0.0)
            st.attrs.update(rate=FS, unit="seconds")
            ds[name] = d
        step = int(600 * FS)
        for i in range(0, n, step):
            t = np.arange(i, min(n, i + step)) / FS
            a, b = t[0] - 1, t[-1] + 1
            m = (r >= a) & (r <= b)
            x = np.zeros_like(t)
            for bb in r[m]:
                j0, j1 = np.searchsorted(t, [bb - 0.02, bb + 0.1])
                tt = t[j0:j1] - bb
                x[j0:j1] += 1e-3 * np.exp(-0.5 * (tt / 0.003) ** 2) + 2e-4 * np.exp(-0.5 * ((tt - 0.05) / 0.01) ** 2)
            ds["ecg"][i:i + t.size] = (x + 2e-5 * rng.standard_normal(t.size)).astype(np.float32)
            k = np.flatnonzero((feet >= a - 1) & (feet <= b + 1))
            if k.size > 1:
                ds["bp"][i:i + t.size] = pressure(t, feet[k], sys_t[k], sbp[k], sbp[k] - 35).astype(np.float32)
            ds["resp"][i:i + t.size] = (0.3 * np.sin(2 * np.pi * 1.5 * t)).astype(np.float32)
            print(f"{t[-1] / 3600:.2f} h", flush=True)
    with open(out / "physio" / "recordings.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recording_id", "subject", "session", "label", "path", "species", "duration_s"])
        w.writerow([rid, "L01", "", "long", f"sub-L01/{rid}.nwb", "rat", dur])
    with open(out / "physio" / "channels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recording_id", "channel", "type", "original_name", "unit", "sampling_rate_hz", "n_samples",
                    "calibrated", "scale", "site", "typing", "status", "status_description", "derived_from"])
        for name, typ, unit, cal in (("ecg", "ecg", "V", "true"), ("bp", "bp", "mmHg", "true"),
                                     ("resp", "resp", "a.u.", "false")):
            w.writerow([rid, name, typ, name.upper(), unit, FS, n, cal, 1, "", "name", "good", "", ""])


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6.0)
