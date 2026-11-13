"""
Validation of bp-resp on the demo data (run after the app has written its outputs).

  python validation/validate.py <fantasia bp-resp out> <fantasia wfdb folder> <aging bp-resp out> \
      <results.csv> [<fantasia bp-resp out without the hrv port>]

1. Fantasia f2 records (uncalibrated BP, reference beat annotations by PhysioNet): every reference
   QRS (`N`) should be followed by exactly one detected systolic peak within 60-500 ms (the pulse
   arrival time); sensitivity = reference beats with a matched pressure beat, positive predictive
   value = pressure beats matched to a reference beat; median pulse arrival time.
2. RSA from ecg-hrv's beats (hrv port) against RSA from the R-peaks bp-resp detects itself (a run
   without the hrv port), Fantasia.
3. Autonomic Aging: BRS per method next to the age-expected value from Kardos et al. 2001 (lnBRS =
   3.24 - 0.03 x age, sequence method, Finapres, 1134 healthy adults; the regression in the abstract)
   at the middle of each subject's age group.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np


def rows(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def main(fan_out: str, fan_wfdb: str, aging_out: str, results: str, nohrv_out: str = "") -> None:
    import wfdb

    out = []
    bp = rows(Path(fan_out) / "bp_beats.csv")
    for rec in ("f2o01", "f2o02", "f2y01", "f2y02"):
        rid = f"sub-{rec}_rec-wfdb"
        sys_t = np.array([float(r["time_s"]) for r in bp if r["recording_id"] == rid])
        ann = wfdb.rdann(str(Path(fan_wfdb) / rec), "ecg")
        ref = np.array([s / ann.fs for s, sym in zip(ann.sample, ann.symbol) if sym == "N"])
        j = np.searchsorted(sys_t, ref + 0.06)
        ok = j < sys_t.size
        jj = np.clip(j, 0, sys_t.size - 1)
        lat = sys_t[jj] - ref
        nxt = np.r_[ref[1:], np.inf]
        hit = ok & (lat <= 0.5) & (sys_t[jj] < nxt + 0.06)
        sens = float(hit.mean())
        matched = np.unique(jj[hit])
        ppv = matched.size / sys_t.size
        out += [[rec, "bp_pulse_sensitivity_vs_reference_qrs", round(sens, 4), f"{hit.sum()}/{ref.size} N beats"],
                [rec, "bp_pulse_ppv_vs_reference_qrs", round(ppv, 4), f"{matched.size}/{sys_t.size} pressure beats"],
                [rec, "median_pulse_arrival_s", round(float(np.median(lat[hit])), 3), "R to systolic peak"]]
    # RSA with R-peaks detected inside bp-resp (run without the hrv port, in <fan_out>-nohrv) vs ecg-hrv's beats
    det = Path(nohrv_out) if nohrv_out else Path("/nonexistent")
    if det.exists():
        own = {(r["recording_id"], r["method"]): r["value"] for r in rows(det / "rsa.csv")}
        for r in rows(Path(fan_out) / "rsa.csv"):
            out.append([r["recording_id"], f"rsa_{r['method']}_ecg_hrv_vs_detected", r["value"],
                        f"with R-peaks detected in bp-resp: {own.get((r['recording_id'], r['method']), '')}"])
    # aging
    info = {r["ID"]: r for r in rows(Path(fan_wfdb).parent / "autonomic-aging" / "subject-info.csv")}
    groups = {1: (18, 19), 2: (20, 24), 3: (25, 29), 4: (30, 34), 5: (35, 39), 6: (40, 44), 7: (45, 49),
              8: (50, 54), 9: (55, 59), 10: (60, 64), 11: (65, 69), 12: (70, 74), 13: (75, 79), 14: (80, 84),
              15: (85, 92)}
    for r in rows(Path(aging_out) / "brs.csv"):
        sid = r["recording_id"].split("_")[0].replace("sub-", "")
        g = groups[int(info[sid]["Age_group"])]
        age = (g[0] + g[1]) / 2
        exp_ = math.exp(3.24 - 0.03 * age)
        out.append([r["recording_id"], f"brs_{r['method']}", r["value_ms_per_mmhg"],
                    f"n={r['n']} valid={r['valid']} coherence={r['coherence']}; age group {g[0]}-{g[1]} y; "
                    f"Kardos 2001 regression at {age:g} y: {exp_:.1f} ms/mmHg"])
    with open(results, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recording", "check", "value", "note"])
        w.writerows(out)
    for o in out:
        print(*o, sep=" | ")


if __name__ == "__main__":
    main(*sys.argv[1:6])
