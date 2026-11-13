"""
bp-resp: beat-to-beat arterial pressure, breathing, spontaneous baroreflex sensitivity and
respiratory sinus arrhythmia, with species presets.

  INPUT_DIR   physio/  physio-import's output (recordings.csv, channels.csv, NWB per recording) or
                       any NWB files
              events/  optional: stim-events' stim-events/events.tsv (stimulation epochs are left out
                       of the resting baroreflex and RSA estimates and shaded in the figures)
              hrv/     optional: ecg-hrv's ecg-hrv/beats.csv (R-peaks for the RR intervals)
              (all found by name and content, recursively, also when merged side by side)
  OUTPUT_DIR/bp-resp/
      bp_beats.csv  breaths.csv  series.csv  brs.csv  rsa.csv  qc.csv  features.csv
      figures/<recording_id>_{bp,resp,brs}.png  summary.json  provenance.json

One recording at a time; a recording that fails is logged and skipped; a recording without blood
pressure and respiration is listed as skipped (not a failure).
Exit codes: 0 ok, 1 no usable input / every recording with a signal failed, 2 bad parameters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_environment() -> None:
    """Lambda-safe start: HOME and TMPDIR may be fresh, empty folders and everything else read-only."""
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    for var, default in (("TMPDIR", tmp), ("HOME", tmp / "home")):
        p = Path(os.environ.get(var) or default)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        os.environ[var] = str(p)
    mpl = Path(os.environ["TMPDIR"]) / "mpl"
    baked = Path(os.environ.get("BP_RESP_MPL_CACHE", "/opt/bp-resp/matplotlib"))
    if not os.environ.get("MPLCONFIGDIR") or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
        mpl.mkdir(parents=True, exist_ok=True)
        if baked.is_dir():
            for f in baked.glob("fontlist-*.json"):
                target = mpl / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
        os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("MPLBACKEND", "Agg")


_prepare_environment()

from processor import resources as R  # noqa: E402

_requested = os.environ.get("THREADS", "0").strip() or "0"
try:
    _THREADS, _THREADS_WHY = R.resolve_workers(int(_requested), per_worker_mb=300, reserve_mb=400)
except ValueError:
    _THREADS, _THREADS_WHY = 1, f"THREADS={_requested!r} is not an integer"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import numpy as np  # noqa: E402

from processor import bp as B  # noqa: E402
from processor import brs as S  # noqa: E402
from processor import inputs as I  # noqa: E402
from processor import presets as PR  # noqa: E402
from processor import resp as RS  # noqa: E402
from processor import rsa as RA  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("bp-resp")
log.setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="neurokit2")

APP_NAME = "bp-resp"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE, FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
RESEARCH = "Research output, not a clinical finding."

# DOIs a report should cite for this step (verified; full records in CITATION.cff)
CITATIONS = ["10.3758/s13428-020-01516-y", "10.1109/CIC.2003.1291140", "10.1161/01.HYP.12.2.214",
             "10.1152/ajpregu.00709.2002", "10.1088/1361-6579/aad7e6", "10.1016/j.biopsycho.2011.11.009",
             "10.7554/eLife.78362"]

DEFAULTS = {
    "SPECIES": "auto", "SERIES_HZ": "auto", "MAX_GAP_S": "auto",
    "BP_LOWPASS_HZ": "auto", "SSF_WINDOW_S": "auto", "BP_REFRACTORY_S": "auto", "SSF_THRESHOLD": "0.5",
    "SSF_ONSET": "0.02", "BP_JUMP": "0.5", "ECTOPIC_FRAC": "0.3",
    "RESP_BAND": "auto", "RESP_AMPLITUDE_MIN": "0.3", "RESP_POLARITY": "normal",
    "RR_SOURCE": "auto", "SEQ_SBP_MMHG": "auto", "SEQ_RR_MS": "auto", "SEQ_LAG": "auto", "SEQ_MIN_BEATS": "3",
    "SEQ_MIN_R": "0.8", "SEQ_MIN_N": "3", "LF_BAND": "auto", "HF_BAND": "auto", "COHERENCE_MIN": "0.5",
    "SPECTRAL_WINDOW_S": "auto", "EXCLUDE_STIM": "true", "AFTER_S": "auto", "FIGURES": "true", "THREADS": "0",
}

HEAD = {
    "bp_beats": ["recording_id", "beat", "time_s", "sbp_mmhg", "dbp_mmhg", "map_mmhg", "pp_mmhg", "calibrated",
                 "onset_s", "ibi_ms", "label"],
    "breaths": ["recording_id", "breath", "onset_s", "duration_s", "rate_bpm", "amplitude", "inspiration_s",
                "expiration_s", "label"],
    "series": ["recording_id", "time_s", "variable", "value", "valid"],
    "brs": ["recording_id", "method", "value_ms_per_mmhg", "n", "coherence", "valid", "lag_beats", "rr_source",
            "note"],
    "rsa": ["recording_id", "method", "value", "unit", "n_breaths", "note"],
    "qc": ["recording_id", "check", "value", "note"],
    "features": ["recording_id", "channel", "feature", "value"],
}
BRS_METHODS = ["sequence-up", "sequence-down", "sequence-all", "alpha-lf", "alpha-hf", "transfer-lf"]


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("neurokit2", "numpy", "scipy", "pandas", "h5py", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


# --------------------------------------------------------------------------- parameters

def _bool(name: str, v: str) -> bool:
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {v!r}")


def _num(name: str, v: str, lo: float | None = 0.0, hi: float | None = None, auto: bool = False,
         strict_lo: bool = True) -> float | None:
    s = str(v).strip().lower()
    if auto and s in ("auto", ""):
        return None
    try:
        f = float(s)
    except ValueError:
        raise ValueError(f"{name} must be {'auto or ' if auto else ''}a number, got {v!r}") from None
    if not math.isfinite(f) or (lo is not None and (f <= lo if strict_lo else f < lo)) or (hi is not None and f > hi):
        rng = f"{'>' if strict_lo else '>='} {lo:g}" if lo is not None else ""
        if hi is not None:
            rng += f" and <= {hi:g}"
        raise ValueError(f"{name} must be {rng}, got {v!r}")
    return f


def _band(name: str, v: str) -> tuple | None:
    s = str(v).strip().lower()
    if s in ("auto", ""):
        return None
    try:
        a, b = (float(x) for x in s.replace(",", "-").split("-", 1))
    except ValueError:
        raise ValueError(f"{name} must be auto or low-high in Hz, e.g. 0.04-0.15, got {v!r}") from None
    if not (0 <= a < b):
        raise ValueError(f"{name}: the low edge must be >= 0 and below the high edge, got {v!r}")
    return (a, b)


def read_params(env=os.environ) -> tuple[dict, dict]:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    sp = raw["SPECIES"].strip().lower()
    if sp != "auto" and PR.resolve(sp) is None:
        raise ValueError(f"SPECIES must be auto or one of {', '.join(PR.SPECIES)}, got {raw['SPECIES']!r}")
    lag = raw["SEQ_LAG"].strip().lower()
    if lag not in ("auto", "0", "1", "2", "3"):
        raise ValueError(f"SEQ_LAG must be auto, 0, 1, 2 or 3, got {raw['SEQ_LAG']!r}")
    rr_src = raw["RR_SOURCE"].strip().lower()
    if rr_src not in ("auto", "hrv", "ecg", "bp"):
        raise ValueError(f"RR_SOURCE must be auto, hrv, ecg or bp, got {raw['RR_SOURCE']!r}")
    pol = raw["RESP_POLARITY"].strip().lower()
    if pol not in ("normal", "inverted"):
        raise ValueError(f"RESP_POLARITY must be normal or inverted, got {raw['RESP_POLARITY']!r}")
    p = {
        "species": None if sp == "auto" else PR.resolve(sp),
        "series_hz": _num("SERIES_HZ", raw["SERIES_HZ"], auto=True, hi=100),
        "max_gap_s": _num("MAX_GAP_S", raw["MAX_GAP_S"], auto=True),
        "bp_lowpass_hz": _num("BP_LOWPASS_HZ", raw["BP_LOWPASS_HZ"], auto=True),
        "ssf_window_s": _num("SSF_WINDOW_S", raw["SSF_WINDOW_S"], auto=True, hi=1),
        "bp_refractory_s": _num("BP_REFRACTORY_S", raw["BP_REFRACTORY_S"], auto=True, hi=5),
        "ssf_threshold": _num("SSF_THRESHOLD", raw["SSF_THRESHOLD"], hi=1),
        "ssf_onset": _num("SSF_ONSET", raw["SSF_ONSET"], hi=0.5),
        "bp_jump": _num("BP_JUMP", raw["BP_JUMP"]),
        "ectopic_frac": _num("ECTOPIC_FRAC", raw["ECTOPIC_FRAC"], hi=1),
        "resp_band": _band("RESP_BAND", raw["RESP_BAND"]),
        "resp_amplitude_min": _num("RESP_AMPLITUDE_MIN", raw["RESP_AMPLITUDE_MIN"], lo=0, strict_lo=False, hi=1),
        "resp_inverted": pol == "inverted",
        "rr_source": rr_src,
        "seq_sbp": _num("SEQ_SBP_MMHG", raw["SEQ_SBP_MMHG"], auto=True, lo=0, strict_lo=False),
        "seq_rr": _num("SEQ_RR_MS", raw["SEQ_RR_MS"], auto=True, lo=0, strict_lo=False),
        "seq_lag": lag,
        "seq_min_beats": int(_num("SEQ_MIN_BEATS", raw["SEQ_MIN_BEATS"], lo=3, strict_lo=False, hi=20)),
        "seq_min_r": _num("SEQ_MIN_R", raw["SEQ_MIN_R"], lo=0, strict_lo=False, hi=1),
        "seq_min_n": int(_num("SEQ_MIN_N", raw["SEQ_MIN_N"], lo=1, strict_lo=False)),
        "lf": _band("LF_BAND", raw["LF_BAND"]),
        "hf": _band("HF_BAND", raw["HF_BAND"]),
        "coh_min": _num("COHERENCE_MIN", raw["COHERENCE_MIN"], lo=0, strict_lo=False, hi=1),
        "spectral_window_s": _num("SPECTRAL_WINDOW_S", raw["SPECTRAL_WINDOW_S"], auto=True),
        "exclude_stim": _bool("EXCLUDE_STIM", raw["EXCLUDE_STIM"]),
        "after_s": _num("AFTER_S", raw["AFTER_S"], auto=True, lo=0, strict_lo=False),
        "figures": _bool("FIGURES", raw["FIGURES"]),
    }
    try:
        int(raw["THREADS"])
    except ValueError:
        raise ValueError("THREADS must be an integer") from None
    return raw, p


def preset_for(name: str, prm: dict) -> PR.Preset:
    """The species preset with every parameter the user set (not auto) applied over it."""
    pre = PR.PRESETS[name]
    over = {"series_hz": prm["series_hz"], "max_gap_s": prm["max_gap_s"], "bp_lowpass_hz": prm["bp_lowpass_hz"],
            "ssf_window_s": prm["ssf_window_s"], "bp_refractory_s": prm["bp_refractory_s"],
            "resp_band_hz": prm["resp_band"], "seq_sbp_mmhg": prm["seq_sbp"], "seq_rr_ms": prm["seq_rr"],
            "lf_hz": prm["lf"], "hf_hz": prm["hf"], "spectral_window_s": prm["spectral_window_s"],
            "after_s": prm["after_s"]}
    over = {k: v for k, v in over.items() if v is not None}
    if prm["seq_lag"] != "auto":
        over["seq_lag"] = prm["seq_lag"]
    return replace(pre, **over) if over else pre


def num(x, nd: int = 6):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if math.isfinite(xf) else None


def fmt(v, sig: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return ""
        return f"{float(v):.{sig}g}"
    return str(v)


# --------------------------------------------------------------------------- series

def grid_rows(rid: str, segments: list, t: np.ndarray, vals: dict[str, np.ndarray], hz: float,
              max_gap_s: float) -> list:
    """series.csv rows: each variable linearly interpolated between its event times onto the regular
    grid (multiples of 1/hz) inside each signal segment; valid=false outside the first-last event,
    and where the neighbouring events are more than max_gap_s apart."""
    rows = []
    for seg in segments:
        g = np.arange(math.ceil(seg.t0 * hz) / hz, seg.t1 + 1e-9, 1.0 / hz)
        if g.size == 0:
            continue
        m = (t >= seg.t0) & (t <= seg.t1)
        tt = t[m]
        if tt.size >= 2:
            j = np.clip(np.searchsorted(tt, g), 1, tt.size - 1)
            inside = (g >= tt[0]) & (g <= tt[-1])
            valid = inside & ((tt[j] - tt[j - 1]) <= max_gap_s)
        else:
            inside = valid = np.zeros(g.size, bool)
        for var, v in vals.items():
            if tt.size >= 2:
                y = np.interp(g, tt, v[m])
            else:
                y = np.full(g.size, np.nan)
            for gt, yy, ins, ok in zip(g, y, inside, valid):
                rows.append([rid, f"{gt:.4f}".rstrip("0").rstrip("."), var, fmt(yy) if ins else "",
                             "true" if ok else "false"])
    return rows


def in_epochs(t: np.ndarray, epochs: list) -> np.ndarray:
    m = np.zeros(t.size, bool)
    for a, b, _ in epochs:
        m |= (t >= a) & (t <= b)
    return m


# --------------------------------------------------------------------------- one recording

def species_of(rec: I.Recording, prm: dict) -> tuple[str, str]:
    if prm["species"]:
        return prm["species"], "SPECIES parameter"
    sp = PR.resolve(rec.species)
    if sp:
        return sp, "recordings.csv / NWB Subject"
    if rec.species:
        return "human", f"species {rec.species!r} has no preset: human preset used"
    return "human", "species unknown: human preset used (set SPECIES)"


def process(rec: I.Recording, prm: dict, epochs_rows: list[dict], hrv_beats: I.EcgBeats | None, out: dict) -> dict:
    t0 = time.monotonic()
    rid = rec.recording_id
    sp, sp_why = species_of(rec, prm)
    pre = preset_for(sp, prm)
    qc = out["qc"]
    if "used" in sp_why or "unknown" in sp_why:
        qc.append([rid, "species", sp, sp_why])
    bp_ch, resp_ch, ecg_ch = I.pick_channel(rec, "bp"), I.pick_channel(rec, "resp"), I.pick_channel(rec, "ecg")
    info = {"species": sp, "species_source": sp_why, "preset": pre.name,
            "channels": {k: (c.name if c else None) for k, c in (("bp", bp_ch), ("resp", resp_ch), ("ecg", ecg_ch))}}
    flat_types = [c.type for c in rec.channels if c.status == "flat" and c.type in ("bp", "resp")]
    if bp_ch is None and resp_ch is None:
        why = "no blood pressure or respiration channel"
        if flat_types:
            why += f" ({' and '.join(sorted(set(flat_types)))} channel flat)"
        return {"status": "skipped", "reason": why, "seconds": round(time.monotonic() - t0, 2), **info}
    epochs = I.stim_epochs(epochs_rows, pre.after_s) if prm["exclude_stim"] else []
    if epochs:
        qc.append([rid, "stimulation_excluded", len(epochs),
                   f"{len(epochs)} stimulation/block epoch(s) (+{pre.after_s:g} s after each) left out of the "
                   "resting baroreflex and RSA estimates"])
    head: dict = {"species": sp, "calibrated": None}
    feats = out["features"]
    res = {"bp": None, "breaths": None, "ecg": None, "brs": None, "rsa": None, "epochs": epochs, "pre": pre}

    # ---------------- blood pressure
    bp_segs = []
    calibrated = False
    if bp_ch is not None:
        calibrated = bool(bp_ch.calibrated) and bp_ch.unit.strip().lower() in ("mmhg", "mm hg")
        if bp_ch.calibrated and not calibrated:
            qc.append([rid, "bp_unit", bp_ch.unit, "calibrated, but the unit is not mmHg: treated as uncalibrated"])
        head["calibrated"] = calibrated
        bp_segs = I.read_channel(rec.nwb, bp_ch.name, pre.bp_hz)
        s = B.BpSettings(pre.bp_lowpass_hz, pre.ssf_window_s, pre.bp_refractory_s, prm["ssf_threshold"],
                         prm["ssf_onset"], pre.rr_min_s, pre.rr_max_s, pre.sbp_range, pre.dbp_range, pre.pp_min_mmhg,
                         prm["bp_jump"], prm["ectopic_frac"])
        bt = B.beats(bp_segs, s, calibrated)
        res["bp"] = bt
        g = bt.good
        n_art, n_ect = int(np.sum(bt.label == "artifact")), int(np.sum(bt.label == "ectopic"))
        pct_art = 100.0 * n_art / bt.n if bt.n else float("nan")
        qc.append([rid, "bp_beats", bt.n, f"{bt.n} pressure beats on '{bp_ch.name}' ({bp_ch.original_name}) at "
                   f"{bp_segs[0].fs if bp_segs else 0:g} Hz; {n_art} artifact, {n_ect} ectopic"
                   + ("" if calibrated else "; uncalibrated: timing only, no pressures written")])
        why_counts = {}
        for w in bt.why[~g]:
            why_counts[str(w)] = why_counts.get(str(w), 0) + 1
        for w, c in sorted(why_counts.items(), key=lambda x: -x[1]):
            qc.append([rid, "bp_beat_label", c, w])
        if bt.n and pct_art > 20:
            out["warnings"].append(f"{rid}: {pct_art:.0f} % of pressure beats are artifact")
        if bp_ch.status and bp_ch.status != "good":
            qc.append([rid, "bp_channel_status", bp_ch.status, bp_ch.status_description])
        head["bp_beats"] = bt.n
        head["bp_pct_artifact"] = num(pct_art, 2)
        if g.sum():
            head["pulse_rate_bpm"] = num(60.0 / np.median(bt.ibi_s[g]), 2)
            feats.append([rid, bp_ch.name, "bp_pulse_rate_bpm", fmt(head["pulse_rate_bpm"])])
        if calibrated and g.sum():
            for k, v in (("sbp", bt.sbp), ("dbp", bt.dbp), ("map", bt.map), ("pp", bt.pp)):
                m_, sd_ = float(np.mean(v[g])), float(np.std(v[g], ddof=1)) if g.sum() > 1 else float("nan")
                head[f"{k}_mmhg"] = num(m_, 2)
                feats.append([rid, bp_ch.name, f"bp_{k}_mean_mmhg", fmt(m_)])
                feats.append([rid, bp_ch.name, f"bp_{k}_sd_mmhg", fmt(sd_)])
            if not (20 <= head["map_mmhg"] <= 200):
                qc.append([rid, "bp_calibration_doubtful", head["map_mmhg"],
                           "mean arterial pressure outside 20-200 mmHg: check the channel's unit and scale"])
                out["warnings"].append(f"{rid}: mean pressure {head['map_mmhg']} mmHg looks wrong: check calibration")
        if not calibrated:
            out["uncalibrated"].append(rid)
        feats.append([rid, bp_ch.name, "bp_n_beats", str(bt.n)])
        feats.append([rid, bp_ch.name, "bp_pct_artifact", fmt(pct_art)])
        for i in range(bt.n):
            out["bp_beats"].append([rid, i + 1, fmt(bt.sys_t[i], 7),
                                    fmt(bt.sbp[i]) if calibrated else "", fmt(bt.dbp[i]) if calibrated else "",
                                    fmt(bt.map[i]) if calibrated else "", fmt(bt.pp[i]) if calibrated else "",
                                    "true" if calibrated else "false", fmt(bt.onset_t[i], 7),
                                    fmt(bt.ibi_s[i] * 1000.0), bt.label[i]])
        if calibrated:
            gt = bt.sys_t[g]
            out["series"].extend(grid_rows(rid, bp_segs, gt, {"sbp_mmhg": bt.sbp[g], "dbp_mmhg": bt.dbp[g],
                                                              "map_mmhg": bt.map[g], "pp_mmhg": bt.pp[g]},
                                           pre.series_hz, pre.max_gap_s))

    # ---------------- respiration
    resp_segs = []
    if resp_ch is not None:
        resp_segs = I.read_channel(rec.nwb, resp_ch.name, pre.resp_hz)
        br = RS.breaths(resp_segs, pre.resp_band_hz, pre.resp_rate_range, prm["resp_amplitude_min"],
                        prm["resp_inverted"])
        res["breaths"] = br
        g = br.good
        qc.append([rid, "breaths", br.n, f"{br.n} breaths on '{resp_ch.name}' ({resp_ch.original_name}, "
                   f"{resp_ch.unit or 'no unit'}); {int((~g).sum())} outside {pre.resp_rate_range[0]:g}-"
                   f"{pre.resp_rate_range[1]:g} breaths/min"])
        if resp_ch.status and resp_ch.status != "good":
            qc.append([rid, "resp_channel_status", resp_ch.status, resp_ch.status_description])
        if g.sum() >= 5:
            ie = float(np.median(br.insp_s[g]) / np.median(br.exp_s[g]))
            qc.append([rid, "resp_inspiration_expiration_ratio", num(ie, 3),
                       "median inspiration / expiration time" + (
                           "; inspiration longer than expiration: if the trace is a volume or belt signal it may be "
                           "upside down (set RESP_POLARITY=inverted)" if ie > 1.2 else "")])
            head["resp_rate_bpm"] = num(float(np.median(br.rate_bpm[g])), 2)
            head["resp_amplitude"] = num(float(np.median(br.amplitude[g])), 6)
            head["resp_amplitude_unit"] = resp_ch.unit or "a.u."
            feats.append([rid, resp_ch.name, "resp_rate_median_bpm", fmt(head["resp_rate_bpm"])])
            feats.append([rid, resp_ch.name, "resp_rate_sd_bpm", fmt(float(np.std(br.rate_bpm[g], ddof=1)))])
            feats.append([rid, resp_ch.name, "resp_amplitude_median", fmt(head["resp_amplitude"])])
            feats.append([rid, resp_ch.name, "resp_n_breaths", str(br.n)])
            mid = br.onset_s[g] + br.duration_s[g] / 2.0
            gap = max(pre.max_gap_s, 3.0 * float(np.median(br.duration_s[g])))
            out["series"].extend(grid_rows(rid, resp_segs, mid, {"resp_rate_bpm": br.rate_bpm[g],
                                                                 "resp_amp": br.amplitude[g]},
                                           pre.series_hz, gap))
        else:
            out["warnings"].append(f"{rid}: fewer than 5 breaths found on {resp_ch.name}")
        for i in range(br.n):
            out["breaths"].append([rid, i + 1, fmt(br.onset_s[i], 7), fmt(br.duration_s[i]), fmt(br.rate_bpm[i]),
                                   fmt(br.amplitude[i]), fmt(br.insp_s[i]), fmt(br.exp_s[i]), br.label[i]])

    # ---------------- R-peaks (for the baroreflex and RSA)
    need_rr = (res["bp"] is not None and calibrated and res["bp"].good.sum() > 10) or \
        (res["breaths"] is not None and res["breaths"].good.sum() > 5)
    ecg = None
    if need_rr:
        src = prm["rr_source"]
        if src in ("auto", "hrv") and hrv_beats is not None:
            ecg = hrv_beats
        elif src in ("auto", "ecg") and ecg_ch is not None:
            from processor import ecgbeats as EB  # noqa: PLC0415
            ecg_hz = 500.0 if pre.typical_hr_bpm < 200 else 1000.0
            ecg = EB.detect(I.read_channel(rec.nwb, ecg_ch.name, ecg_hz), pre, 0.2)
            ecg.note += f" on '{ecg_ch.name}' ({ecg_ch.original_name})"
        elif src == "hrv":
            qc.append([rid, "rr_source", "none", "RR_SOURCE=hrv but ecg-hrv's beats.csv has no rows for this recording"])
        elif src == "ecg":
            qc.append([rid, "rr_source", "none", "RR_SOURCE=ecg but the recording has no ECG channel"])
        if ecg is not None and ecg.times.size < 10:
            qc.append([rid, "rr_source", ecg.source, f"too few R-peaks ({ecg.times.size}): {ecg.note}"])
            ecg = None
        if ecg is not None:
            qc.append([rid, "rr_source", ecg.source, f"{ecg.times.size} R-peaks, {100 * ecg.good.mean():.1f} % usable; "
                       + ecg.note])
        elif res["bp"] is not None and calibrated:
            qc.append([rid, "rr_source", "bp-pulse", "no R-peaks: the baroreflex uses the pulse interval (systolic "
                       "peak to systolic peak) of the pressure instead of the RR interval"])
    res["ecg"] = ecg
    head["rr_source"] = (ecg.source if ecg is not None else ("bp-pulse" if res["bp"] is not None and calibrated
                                                            else None))

    # ---------------- baroreflex
    brs_rows = out["brs"]
    if res["bp"] is None:
        for m in BRS_METHODS:
            brs_rows.append([rid, m, "", "", "", "false", "", "", "no blood pressure channel"])
    elif not calibrated:
        for m in BRS_METHODS:
            brs_rows.append([rid, m, "", "", "", "false", "", "",
                             "blood pressure uncalibrated (no mmHg): baroreflex slopes in ms/mmHg cannot be computed"])
    else:
        res["brs"] = baroreflex(rid, res["bp"], ecg, pre, prm, epochs, brs_rows, qc, feats, head)

    # ---------------- RSA
    rsa_rows = out["rsa"]
    br = res["breaths"]
    if br is None or not br.segments:
        why = "no respiration channel" if resp_ch is None else "no breaths found"
        rsa_rows += [[rid, "p2t", "", "ms", "", why], [rid, "porges-bohrer", "", "ln(ms^2)", "", why]]
    elif ecg is None:
        why = "no R-peaks (no ECG channel and no ecg-hrv beats linked)"
        rsa_rows += [[rid, "p2t", "", "ms", "", why], [rid, "porges-bohrer", "", "ln(ms^2)", "", why]]
    else:
        res["rsa"] = rsa_estimate(rid, br, ecg, pre, epochs, rsa_rows, feats, head)

    out["headlines"][rid] = head
    res.update({"status": "ok", "seconds": round(time.monotonic() - t0, 2), "bp_segs": bp_segs,
                "resp_segs": resp_segs, "calibrated": calibrated, **info})
    return res


def baroreflex(rid, bt, ecg, pre, prm, epochs, rows, qc, feats, head) -> dict:
    rr_range = (pre.rr_min_s, pre.rr_max_s)
    use = ~in_epochs(bt.sys_t, epochs)
    lags = list(pre.seq_lags) if pre.seq_lag == "auto" else [int(pre.seq_lag)]
    best = None
    per_lag = {}
    for lag in lags:
        iv = S.pair_rr(bt, ecg, lag, rr_range)
        seqs, ramps = S.sequences(bt.sbp, iv, use, pre.seq_sbp_mmhg, pre.seq_rr_ms, prm["seq_min_beats"],
                                  prm["seq_min_r"])
        per_lag[lag] = len(seqs)
        if best is None or len(seqs) > len(best[1]):
            best = (lag, seqs, ramps, iv)
    lag, seqs, ramps, iv = best
    src = iv.source
    if len(lags) > 1:
        qc.append([rid, "brs_sequence_lag", lag, "sequences per lag: " + ", ".join(f"{k}: {v}" for k, v in per_lag.items())
                   + f"; the lag with the most sequences is used (SEQ_LAG=auto)"])
    n_beats_used = int((iv.ok & use).sum())
    qc.append([rid, "brs_sequences", len(seqs), f"{ramps} SBP/RR ramps of >= {prm['seq_min_beats']} beats "
               f"(thresholds {pre.seq_sbp_mmhg:g} mmHg, {pre.seq_rr_ms:g} ms), {len(seqs)} with r >= {prm['seq_min_r']:g}; "
               f"{n_beats_used} beats paired with an RR interval"])
    out = {"lag": lag, "seqs": seqs, "iv": iv, "spec": None}
    for kind, sel in (("sequence-up", [q for q in seqs if q.kind == "up"]),
                      ("sequence-down", [q for q in seqs if q.kind == "down"]), ("sequence-all", seqs)):
        n = len(sel)
        val = float(np.mean([q.slope for q in sel])) if n else float("nan")
        ok = n >= prm["seq_min_n"]
        note = "" if ok else (f"only {n} sequence(s) (SEQ_MIN_N {prm['seq_min_n']})" if n else "no sequences")
        rows.append([rid, kind, fmt(val), n, "", "true" if ok else "false", lag, src, note])
        if ok:
            feats.append([rid, "", "brs_" + kind.replace("-", "_"), fmt(val)])
    head["brs_sequence_all"] = num(np.mean([q.slope for q in seqs]), 3) if len(seqs) >= prm["seq_min_n"] else None
    head["brs_n_sequences"] = len(seqs)
    # spectral, on lag-0 pairs
    iv0 = S.pair_rr(bt, ecg, 0, rr_range)
    ok = iv0.ok & np.isfinite(bt.sbp)
    sp = S.spectral(bt.sys_t, bt.sbp, iv0.rr_ms, ok, pre.spectral_hz, pre.spectral_window_s,
                    max(pre.max_gap_s, 2.5 * pre.rr_max_s), [(a, b) for a, b, _ in epochs])
    out["spec"] = sp
    if sp is None or sp.n_windows < 3:
        why = (f"fewer than 3 clean {pre.spectral_window_s:g} s windows (coherence needs several windows)"
               if sp is not None else f"no clean {pre.spectral_window_s:g} s window of SBP and RR")
        for m in ("alpha-lf", "alpha-hf", "transfer-lf"):
            rows.append([rid, m, "", sp.n_windows if sp else 0, "", "false", 0, src, why])
        return out
    for m, band in (("alpha-lf", pre.lf_hz), ("alpha-hf", pre.hf_hz)):
        a, c, nb = S.band_alpha(sp, band)
        valid = bool(np.isfinite(a) and np.isfinite(c) and c >= prm["coh_min"])
        note = f"{band[0]:g}-{band[1]:g} Hz, {sp.n_windows} windows of {sp.window_s:g} s"
        if not valid:
            note += f"; mean coherence {c:.2f} < {prm['coh_min']:g}: not a valid estimate"
        rows.append([rid, m, fmt(a), sp.n_windows, fmt(c, 3), "true" if valid else "false", 0, src, note])
        if valid:
            feats.append([rid, "", "brs_" + m.replace("-", "_"), fmt(a)])
            head["brs_" + m.replace("-", "_")] = num(a, 3)
    g_, c_, nb = S.band_transfer(sp, pre.lf_hz, prm["coh_min"])
    valid = nb > 0
    note = (f"{pre.lf_hz[0]:g}-{pre.lf_hz[1]:g} Hz, mean gain over the {nb} frequencies with coherence >= "
            f"{prm['coh_min']:g}" if valid else f"no LF frequency with coherence >= {prm['coh_min']:g}")
    rows.append([rid, "transfer-lf", fmt(g_), nb, fmt(c_, 3), "true" if valid else "false", 0, src, note])
    if valid:
        feats.append([rid, "", "brs_transfer_lf", fmt(g_)])
        head["brs_transfer_lf"] = num(g_, 3)
    return out


def rsa_estimate(rid, br, ecg, pre, epochs, rows, feats, head) -> dict | None:
    human = pre.name == "human"
    r_t = RA.fill_beats(ecg.times, ecg.good)
    gb = br.good
    med_breath = float(np.median(br.duration_s[gb])) if gb.any() else 4.0
    offset = 0.75 if human else min(0.75, 0.25 * med_breath)
    min_len = max(60.0, 10 * med_breath)
    acc = {"p2t": [], "pb": [], "w": [], "why_pb": ""}
    for seg in br.segments:
        n = seg.clean.size
        t = seg.t0 + np.arange(n) / seg.fs
        free = ~in_epochs(t, epochs) if epochs else np.ones(n, bool)
        edges = np.diff(np.concatenate([[0], free.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if (b - a) / seg.fs < min_len:
                continue
            r = RA.stretch_rsa(seg, int(a), int(b), r_t, human, offset)
            if not r:
                continue
            val, nbr = r["p2t"]
            if np.isfinite(val) and nbr:
                acc["p2t"].append(val)
                acc["w"].append(nbr)
                if r["porges-bohrer"] is not None and np.isfinite(r["porges-bohrer"]):
                    acc["pb"].append((r["porges-bohrer"], (b - a) / seg.fs))
            acc["why_pb"] = r["why_pb"]
    if not acc["p2t"]:
        why = f"no stretch of >= {min_len:g} s with breaths and R-peaks" + (" outside stimulation" if epochs else "")
        rows += [[rid, "p2t", "", "ms", "", why], [rid, "porges-bohrer", "", "ln(ms^2)", "", why]]
        return None
    w = np.asarray(acc["w"], float)
    p2t = float(np.sum(np.asarray(acc["p2t"]) * w) / np.sum(w))
    note = f"R-peaks from {ecg.source}" + ("" if human else f"; vagal delay {offset:.3g} s (scaled to the breath)")
    rows.append([rid, "p2t", fmt(p2t), "ms", int(w.sum()), note])
    feats.append([rid, "", "rsa_p2t_ms", fmt(p2t)])
    head["rsa_p2t_ms"] = num(p2t, 2)
    if acc["pb"]:
        v = np.asarray([x[0] for x in acc["pb"]])
        d = np.asarray([x[1] for x in acc["pb"]])
        pb = float(np.sum(v * d) / np.sum(d))
        rows.append([rid, "porges-bohrer", fmt(pb), "ln(ms^2)", int(w.sum()), f"R-peaks from {ecg.source}"])
        feats.append([rid, "", "rsa_porges_bohrer", fmt(pb)])
        head["rsa_porges_bohrer"] = num(pb, 3)
    else:
        rows.append([rid, "porges-bohrer", "", "ln(ms^2)", "", acc["why_pb"] or "not computed"])
    return {"p2t": p2t, "offset": offset, "r_t": r_t}


# --------------------------------------------------------------------------- run

def peak_rss_mb() -> float:
    for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(Path(f).read_text().strip()) / 2**20, 1)
        except (OSError, ValueError):
            continue
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024, 1)


def write_csv(path: Path, header: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (fmt(v) if isinstance(v, (float, np.floating)) else v) for v in r])


def main() -> int:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc)
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    try:
        raw, prm = read_params()
    except ValueError as e:
        log.error("parameters: %s", e)
        return 2
    out_dir = output_dir / APP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s threads=%d (%s)", APP_NAME, app_version(), input_dir, out_dir,
             _THREADS, _THREADS_WHY)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s does not exist", input_dir)
        return 1
    physio = I.find_physio(input_dir)
    events, ev_files = I.find_events(input_dir)
    hrv, hrv_files = I.find_hrv_beats(input_dir)
    if ev_files:
        log.info("stimulation events: %s (%d recording(s) with epochs)", ", ".join(ev_files), len(events))
    if hrv_files:
        log.info("ecg-hrv beats: %s (%d recording(s))", ", ".join(hrv_files), len(hrv))
    if not physio.recordings:
        log.error("no recording found under %s (recordings.csv or *.nwb)", input_dir)
        return 1
    log.info("%d recording(s) found", len(physio.recordings))

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(input_dir))
        except ValueError:
            return str(p)

    out = {k: [] for k in HEAD}
    out.update({"warnings": [], "headlines": {}, "uncalibrated": []})
    per_rec, figures = {}, []
    n_ok = n_failed = 0
    figdir = out_dir / "figures"
    presets_used = {}
    for i, rec in enumerate(physio.recordings, 1):
        rid = rec.recording_id
        log.info("[%d/%d] %s (%s)", i, len(physio.recordings), rid, rel(rec.nwb))
        try:
            res = process(rec, prm, events.get(rid, []), hrv.get(rid), out)
        except Exception as e:  # noqa: BLE001
            log.error("%s: failed: %s", rid, e)
            log.info(traceback.format_exc())
            per_rec[rid] = {"status": "failed", "error": str(e), "path": rel(rec.nwb)}
            out["warnings"].append(f"{rid}: failed: {e}")
            n_failed += 1
            continue
        entry = {"status": res["status"], "seconds": res["seconds"], "path": rel(rec.nwb), "species": res["species"],
                 "species_source": res["species_source"], "channels": res["channels"]}
        if res["status"] == "skipped":
            entry["reason"] = res["reason"]
            out["warnings"].append(f"{rid}: skipped: {res['reason']}")
            log.info("%s: skipped: %s", rid, res["reason"])
            per_rec[rid] = entry
            continue
        n_ok += 1
        presets_used[res["pre"].name] = res["pre"].as_dict()
        entry["calibrated"] = res["calibrated"] if res["bp"] is not None else None
        entry["rr_source"] = out["headlines"][rid].get("rr_source")
        per_rec[rid] = entry
        h = out["headlines"][rid]
        log.info("%s: %s beats (%s), %s breaths, BRS seq %s ms/mmHg, RSA p2t %s ms (%.1fs)", rid,
                 res["bp"].n if res["bp"] is not None else "no BP",
                 "calibrated" if res["calibrated"] else "uncalibrated" if res["bp"] is not None else "-",
                 res["breaths"].n if res["breaths"] is not None else "no resp", h.get("brs_sequence_all"),
                 h.get("rsa_p2t_ms"), res["seconds"])
        if prm["figures"]:
            try:
                from processor import figures as F  # noqa: PLC0415
                figdir.mkdir(exist_ok=True)
                figures.extend(F.recording_figures(figdir, rid, res, prm["coh_min"]))
            except Exception as e:  # noqa: BLE001
                log.warning("%s: figure failed: %s", rid, e)
                log.info(traceback.format_exc())

    for name in ("bp_beats", "breaths", "series", "brs", "rsa", "qc", "features"):
        write_csv(out_dir / f"{name}.csv", HEAD[name], out[name])

    finished = datetime.now(timezone.utc)
    provenance = {
        "app": APP_NAME, "version": app_version(), "variant": None, "tools": tool_versions(),
        "parameters": raw, "inputs": sorted(set(physio.sources + ev_files + hrv_files)),
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2), "peak_rss_mb": peak_rss_mb(),
        "threads": {"requested": int(raw["THREADS"]), "used": _THREADS, "why": _THREADS_WHY},
        "recordings": per_rec, "presets": presets_used, "citations": CITATIONS, "licence_notice": None,
    }
    with open(out_dir / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False, default=str)
    warnings_out = out["warnings"]
    if n_ok == 0 and n_failed == 0:
        warnings_out.append("no recording has blood pressure or respiration: every table has the header only")
    summary = {
        "app": APP_NAME, "version": app_version(), "recordings": out["headlines"], "figures": figures,
        "tables": [
            {"path": "bp_beats.csv", "caption": "One row per arterial pressure beat: systolic peak time, systolic, "
             "diastolic, beat-mean (integral) and pulse pressure, the inter-beat interval and a label (N, ectopic, "
             "artifact). Pressures are empty when the channel is uncalibrated."},
            {"path": "breaths.csv", "caption": "One row per breath (inspiration onset to the next): duration, rate, "
             "amplitude in the channel's unit, inspiration and expiration times, label."},
            {"path": "series.csv", "caption": "Beat- and breath-derived series (sbp/dbp/map/pp in mmHg, breathing "
             "rate and amplitude) on a regular time grid, the shape stim-dose-response reads."},
            {"path": "brs.csv", "caption": "Spontaneous baroreflex sensitivity in ms/mmHg: sequence method (up, down, "
             "all) and spectral alpha (LF, HF) and transfer-function gain (LF), with the number of sequences or "
             "windows, coherence and whether the estimate is valid. " + RESEARCH},
            {"path": "rsa.csv", "caption": "Respiratory sinus arrhythmia: peak-to-trough (ms) and Porges-Bohrer "
             "(ln ms^2). " + RESEARCH},
            {"path": "qc.csv", "caption": "Checks per recording: beat and breath counts, why beats were labelled, "
             "where the RR intervals came from, the sequence lag, stimulation left out."},
            {"path": "features.csv", "caption": "One number per recording (means, BRS, RSA) in the shared long shape."},
        ],
        "warnings": warnings_out,
    }
    if out["uncalibrated"]:
        summary["notice"] = ("Blood pressure is uncalibrated in " + ", ".join(out["uncalibrated"]) +
                             ": pulse timing only; no pressures in mmHg and no baroreflex sensitivity for these "
                             "recordings.")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    log.info("%d recording(s) analysed, %d skipped, %d failed; %.1fs, peak memory %.0f MB; wrote %s", n_ok,
             len(physio.recordings) - n_ok - n_failed, n_failed, time.monotonic() - t_start, provenance["peak_rss_mb"],
             out_dir)
    if n_failed and n_ok == 0:
        log.error("every recording with a signal failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
