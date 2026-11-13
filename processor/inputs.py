"""
Finding and reading the inputs.

physio port (physio-import's output, or any NWB files): recordings.csv, channels.csv and one NWB file
per recording, found by name and content, recursively, following symlinks (INPUT_DIR entries are
symlinks). physio-import's own events.tsv (raw markers, next to recordings.csv) is NOT read.

events port (optional): stim-events' `stim-events/events.tsv`: stimulation epochs, left out of the
resting baroreflex and RSA estimates and shaded in the figures.

hrv port (optional): ecg-hrv's `ecg-hrv/beats.csv` (R-peak times and beat labels), used for the RR
intervals of the baroreflex and RSA instead of detecting R-peaks here.

NWB files are read with h5py directly (NWB 2.x is HDF5 with a fixed layout), a slice at a time, so a
long or high-rate channel is never in memory at its native rate: it is low-pass filtered and
decimated chunk by chunk to the working rate the analysis needs (see `read_channel`).
"""
from __future__ import annotations

import csv
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger("bp-resp")

GAP_FACTOR = 1.5          # a step between timestamps longer than 1.5 sample periods starts a new segment
NAN_GAP_S = 1.0           # NaN runs longer than this split a segment; shorter ones are interpolated
CHUNK = 1_000_000         # samples read at a time


# --------------------------------------------------------------------------- file discovery

def walk(root: Path):
    """Every file under root, recursively, following symlinks; no loops."""
    seen: set = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
            key = (st.st_dev, st.st_ino)
        except OSError:
            continue
        if key in seen:
            dirnames[:] = []
            continue
        seen.add(key)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith((".", "~$")):
                yield Path(dirpath) / f


def read_delimited(path: Path) -> list[dict]:
    """CSV or TSV (by extension, else sniffed) as a list of row dicts; UTF-8 with or without BOM."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    if not text.strip():
        return []
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    if path.suffix.lower() != ".tsv":
        first = text.splitlines()[0]
        if first.count("\t") > first.count(","):
            delim = "\t"
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    return [{(k or "").strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k} for r in rows]


def _s(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _s(v[()])
    return str(v)


def fnum(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- physio dataset

@dataclass
class ChannelInfo:
    name: str                 # NWB TimeSeries name (`bp`, `resp-2`, ...)
    type: str                 # ecg / bp / resp / ...
    unit: str
    calibrated: bool
    status: str               # good / flat / saturated / noisy (channels.csv) or "" when unknown
    status_description: str
    original_name: str
    typing: str
    derived_from: str = ""    # empty for a raw channel; else the channel it was derived from (or `unknown`)


@dataclass
class Recording:
    recording_id: str
    nwb: Path
    subject: str = ""
    session: str = ""
    species: str = ""
    duration_s: float | None = None
    age: str = ""
    sex: str = ""
    channels: list[ChannelInfo] = field(default_factory=list)
    from_index: bool = False  # described by channels.csv (else typed here from names)


@dataclass
class PhysioInput:
    recordings: list[Recording]
    sources: list[str]


TYPE_BY_NAME = [
    ("ecg", re.compile(r"(?i)^(ecg|ekg)(-\d+)?$|\becg\b|\bekg\b|lead ?(i|ii|iii)\b")),
    ("bp", re.compile(r"(?i)^bp(-\d+)?$|\b(abp|art|nibp|finapres|blood.?pressure|arterial)\b")),
    ("resp", re.compile(r"(?i)^resp(-\d+)?$|resp|breath|thorax|abdomen|airflow|pneumo")),
]


def _type_from_name(name: str) -> str:
    for t, rx in TYPE_BY_NAME:
        if rx.search(name):
            return t
    return "other"


def _nwb_channels(path: Path) -> list[ChannelInfo]:
    """Channels of an NWB file without channels.csv: types from the names (physio-import's naming
    `ecg`, `bp`, `resp-2`, or common names), calibrated when the unit is a physical one."""
    out = []
    with h5py.File(path, "r") as f:
        if "acquisition" not in f:
            return out
        for name, g in f["acquisition"].items():
            if not isinstance(g, h5py.Group) or "data" not in g:
                continue
            unit = _s(g["data"].attrs.get("unit", ""))
            typ = _type_from_name(name)
            if typ == "other":
                typ = _type_from_name(_s(g.attrs.get("description", "")))
            cal = unit.strip().lower() not in ("", "a.u.", "au", "unknown", "n/a", "no unit")
            if typ == "bp":
                cal = unit.strip().lower() in ("mmhg", "mm hg")
            out.append(ChannelInfo(name, typ, unit, cal, "", "", name, "name"))
    return out


def find_physio(root: Path) -> PhysioInput:
    """recordings.csv (+ channels.csv) when present, else every NWB file found."""
    files = list(walk(root))
    nwbs = [p for p in files if p.suffix.lower() == ".nwb"]
    rec_csvs = [p for p in files if p.name == "recordings.csv"]
    ch_csvs = [p for p in files if p.name == "channels.csv"]
    used: list[Path] = []
    recs: dict[str, Recording] = {}
    by_nwb: dict[Path, Recording] = {}
    for rc in rec_csvs:
        used.append(rc)
        for r in read_delimited(rc):
            rid = (r.get("recording_id") or "").strip()
            rel = (r.get("path") or "").strip()
            if not rid or not rel:
                continue
            p = rc.parent / rel
            if not p.exists():
                cands = [q for q in nwbs if q.name == Path(rel).name]
                if not cands:
                    log.warning("recordings.csv lists %s but %s was not found", rid, rel)
                    continue
                p = cands[0]
            rec = Recording(rid, p, subject=r.get("subject", "") or "", session=r.get("session", "") or "",
                            species=r.get("species", "") or "", duration_s=fnum(r.get("duration_s")),
                            age=r.get("age", "") or "", sex=r.get("sex", "") or "")
            recs[rid] = rec
            by_nwb[p.resolve()] = rec
    for p in nwbs:
        if p.resolve() in by_nwb:
            continue
        try:
            with h5py.File(p, "r") as f:
                rid = _s(f["identifier"][()]) if "identifier" in f else p.stem
                sp = _s(f["general/subject/species"][()]) if "general/subject/species" in f else ""
                subj = _s(f["general/subject/subject_id"][()]) if "general/subject/subject_id" in f else ""
        except Exception as e:  # noqa: BLE001
            log.warning("%s: not a readable NWB file (%s)", p, e)
            continue
        rid = re.sub(r"[^A-Za-z0-9_-]+", "-", rid) or p.stem
        if rid in recs:
            rid = f"{rid}-{p.stem}"
        recs[rid] = Recording(rid, p, subject=subj, species=sp)
        by_nwb[p.resolve()] = recs[rid]
    for cc in ch_csvs:
        used.append(cc)
        for r in read_delimited(cc):
            rec = recs.get((r.get("recording_id") or "").strip())
            name = (r.get("channel") or "").strip()
            if rec is None or not name:
                continue
            rec.from_index = True
            rec.channels.append(ChannelInfo(
                name, (r.get("type") or "other").strip().lower(), (r.get("unit") or "").strip(),
                (r.get("calibrated") or "").strip().lower() in ("true", "1", "yes"),
                (r.get("status") or "").strip().lower(), (r.get("status_description") or "").strip(),
                (r.get("original_name") or name).strip(), (r.get("typing") or "").strip(),
                (r.get("derived_from") or "").strip()))
    for rec in recs.values():
        if not rec.channels:
            try:
                rec.channels = _nwb_channels(rec.nwb)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: channels could not be listed (%s)", rec.nwb, e)
    rels = []
    for p in used + [r.nwb for r in recs.values()]:
        try:
            rels.append(str(p.relative_to(root)))
        except ValueError:
            rels.append(str(p))
    return PhysioInput(sorted(recs.values(), key=lambda r: r.recording_id), sorted(set(rels)))


def pick_channel(rec: Recording, typ: str) -> tuple[ChannelInfo | None, str]:
    """The channel of a type to analyse, and a note when the choice needs explaining. Raw channels
    come first (conventions.md: a processed copy, `derived_from` set, is used only when no raw channel
    of the type exists), then status (`good` before unknown, `noisy`, `saturated`; never `flat`), then
    the name (`bp` before `bp-2`)."""
    cands = [c for c in rec.channels if c.type == typ and c.status != "flat"]
    if not cands:
        return None, ""
    rank = {"good": 0, "": 1, "noisy": 2, "saturated": 3}

    def key(c):
        m = re.search(r"-(\d+)$", c.name)
        return (rank.get(c.status, 5), int(m.group(1)) if m else 0, c.name)

    raw = [c for c in cands if not c.derived_from]
    derived = [c for c in cands if c.derived_from]
    if raw:
        best = sorted(raw, key=key)[0]
        note = ""
        better = [d for d in derived if rank.get(d.status, 5) < rank.get(best.status, 5)]
        if better:
            note = (f"raw '{best.name}' ({best.original_name}, status {best.status or 'unknown'}) used rather than "
                    f"the processed copy '{better[0].name}' ({better[0].original_name}, status {better[0].status}): "
                    "conventions prefer raw channels")
        return best, note
    best = sorted(derived, key=key)[0]
    return best, (f"only a processed copy is available: '{best.name}' ({best.original_name}, derived from "
                  f"{best.derived_from})")


# --------------------------------------------------------------------------- events and beats

def find_events(root: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """stim-events' events.tsv rows by recording_id (stimulation and block epochs). physio-import's
    events.tsv (raw markers, in the folder that holds recordings.csv) is never read."""
    out: dict[str, list[dict]] = {}
    used = []
    for p in walk(root):
        if p.name != "events.tsv":
            continue
        if (p.parent / "recordings.csv").exists():   # physio-import's raw markers
            continue
        if "stim-events" not in p.parts and not p.parent.name.startswith("events"):
            # an events.tsv outside a stim-events folder is accepted only when it has the stim-events shape
            head = p.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
            if "event_id" not in head or "trial_type" not in head:
                continue
        rows = read_delimited(p)
        if rows and "onset" in rows[0] and "trial_type" in rows[0]:
            used.append(p)
            for r in rows:
                rid = (r.get("recording_id") or "").strip()
                if rid and (r.get("trial_type") or "") in ("stimulation", "block"):
                    out.setdefault(rid, []).append(r)
    rels = []
    for p in used:
        try:
            rels.append(str(p.relative_to(root)))
        except ValueError:
            rels.append(str(p))
    return out, rels


def stim_epochs(rows: list[dict], after_s: float) -> list[tuple[float, float, str]]:
    """(start, end + after_s, event_id) of every stimulation/block epoch."""
    out = []
    for r in rows:
        on = fnum(r.get("onset"))
        if on is None:
            continue
        dur = fnum(r.get("duration")) or fnum(r.get("train_duration_s")) or 0.0
        out.append((on, on + dur + after_s, r.get("event_id") or ""))
    return sorted(out)


@dataclass
class EcgBeats:
    times: np.ndarray        # R-peak times, s
    good: np.ndarray         # bool: beat labelled N (usable for RR intervals)
    source: str              # "ecg-hrv" / "detected" / "bp-pulse"
    note: str = ""


def find_hrv_beats(root: Path) -> tuple[dict[str, EcgBeats], list[str]]:
    """ecg-hrv's beats.csv (recording_id, beat, time_s, rr_ms, label, detector) by recording_id."""
    out: dict[str, EcgBeats] = {}
    used = []
    for p in walk(root):
        if p.name != "beats.csv":
            continue
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or not {"recording_id", "time_s", "label"} <= set(rd.fieldnames):
                continue
            tmp: dict[str, tuple[list, list]] = {}
            for r in rd:
                t = fnum(r.get("time_s"))
                if t is None:
                    continue
                a, b = tmp.setdefault(r["recording_id"].strip(), ([], []))
                a.append(t)
                b.append((r.get("label") or "N").strip() == "N")
        used.append(p)
        for rid, (t, g) in tmp.items():
            t = np.asarray(t, float)
            order = np.argsort(t)
            out[rid] = EcgBeats(t[order], np.asarray(g, bool)[order], "ecg-hrv",
                                f"R-peaks from ecg-hrv ({p.name})")
    rels = []
    for p in used:
        try:
            rels.append(str(p.relative_to(root)))
        except ValueError:
            rels.append(str(p))
    return out, rels


# --------------------------------------------------------------------------- signals

@dataclass
class Segment:
    t0: float                 # time of the first sample, s
    fs: float                 # sampling rate of x, Hz
    x: np.ndarray             # values in the channel's unit (float64), NaN-free

    @property
    def t1(self) -> float:
        return self.t0 + (len(self.x) - 1) / self.fs

    def times(self) -> np.ndarray:
        return self.t0 + np.arange(len(self.x)) / self.fs


def _antialias(fs: float, q: int):
    from scipy.signal import butter  # noqa: PLC0415
    return butter(8, 0.8 * (fs / q / 2.0) / (fs / 2.0), btype="low", output="sos")


def _raw_segments(g: h5py.Group, n: int) -> list[tuple[int, int, float, float]]:
    """(start, stop, t0, fs) of the contiguous runs of samples of a TimeSeries."""
    if "starting_time" in g:
        st = g["starting_time"]
        return [(0, n, float(st[()]), float(st.attrs["rate"]))]
    ts = g["timestamps"]
    edges, periods = [0], []
    for i in range(0, n, CHUNK):
        d = np.diff(np.asarray(ts[i:i + CHUNK], dtype=np.float64))
        if d.size:
            periods.append(np.median(d))
    period = float(np.median(periods)) if periods else 1.0
    if not math.isfinite(period) or period <= 0:
        period = 1.0
    prev = None
    for i in range(0, n, CHUNK):
        t = np.asarray(ts[i:i + CHUNK], dtype=np.float64)
        if prev is not None:
            t = np.concatenate([[prev], t])
            off = i - 1
        else:
            off = i
        d = np.diff(t)
        edges.extend((off + np.nonzero(d > GAP_FACTOR * period)[0] + 1).tolist())
        prev = t[-1]
    edges.append(n)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2:
            continue
        t0, t1 = float(ts[a]), float(ts[b - 1])
        out.append((a, b, t0, (b - a - 1) / (t1 - t0) if t1 > t0 else 1.0 / period))
    return out


def channel_extent(nwb: Path, name: str) -> tuple[float, float]:
    """Time of the first and last sample of a channel."""
    with h5py.File(nwb, "r") as f:
        g = f["acquisition"][name]
        segs = _raw_segments(g, int(g["data"].shape[0]))
    if not segs:
        return 0.0, 0.0
    a, b, t0, fs = segs[0]
    a2, b2, t02, fs2 = segs[-1]
    return t0, t02 + (b2 - a2 - 1) / fs2


def native_rate(nwb: Path, name: str) -> float:
    with h5py.File(nwb, "r") as f:
        g = f["acquisition"][name]
        segs = _raw_segments(g, int(g["data"].shape[0]))
        return segs[0][3] if segs else 0.0


def read_channel(nwb: Path, name: str, max_hz: float) -> list[Segment]:
    """A channel as NaN-free segments at a rate <= max_hz (integer decimation with an 8th-order
    Butterworth anti-alias filter, applied a chunk at a time with overlap). NaN runs shorter than
    NAN_GAP_S are linearly interpolated, longer ones split the segment."""
    from scipy.signal import sosfiltfilt  # noqa: PLC0415

    out: list[Segment] = []
    with h5py.File(nwb, "r") as f:
        g = f["acquisition"][name]
        d = g["data"]
        conv = float(d.attrs.get("conversion", 1.0))
        offs = float(d.attrs.get("offset", 0.0))
        n = int(d.shape[0])
        col = 0 if d.ndim == 2 else None
        for a, b, t0, fs in _raw_segments(g, n):
            q = max(1, int(math.floor(fs / max_hz + 1e-9)))
            sos = _antialias(fs, q) if q > 1 else None
            pad = int(min(b - a, max(64 * q, 4 * fs)))
            parts = []
            for i in range(a, b, CHUNK):
                lo, hi = max(a, i - pad), min(b, i + CHUNK + pad)
                x = np.asarray(d[lo:hi] if col is None else d[lo:hi, col], dtype=np.float64)
                if conv != 1.0:
                    x *= conv
                if offs:
                    x += offs
                if sos is not None:
                    bad = ~np.isfinite(x)
                    if bad.any():
                        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad]) if (~bad).any() else 0.0
                    x = sosfiltfilt(sos, x)
                    x[bad] = np.nan
                    # keep the samples of [i, i+CHUNK) on the global decimation grid (multiples of q from a)
                    first = i + ((-(i - a)) % q)
                    idx = np.arange(first, min(b, i + CHUNK), q) - lo
                    parts.append(x[idx])
                else:
                    parts.append(x[i - lo:min(b, i + CHUNK) - lo])
            x = np.concatenate(parts) if parts else np.zeros(0)
            fs_out = fs / q
            out.extend(_split_nan(x, t0, fs_out))
    return out


def iter_windows(nwb: Path, name: str, max_hz: float, win_s: float = 600.0, pad_s: float = 10.0):
    """The channel a window at a time: yields (segments, (core_t0, core_t1)) where segments are the
    NaN-free pieces of the window plus pad_s on each side, decimated to <= max_hz like read_channel.
    A caller keeps what it finds inside the core, so every event is reported once, and memory stays
    bounded by the window whatever the recording's length."""
    from scipy.signal import sosfiltfilt  # noqa: PLC0415

    with h5py.File(nwb, "r") as f:
        g = f["acquisition"][name]
        d = g["data"]
        conv = float(d.attrs.get("conversion", 1.0))
        offs = float(d.attrs.get("offset", 0.0))
        n = int(d.shape[0])
        col = 0 if d.ndim == 2 else None
        segs = _raw_segments(g, n)
    for a, b, t0, fs in segs:
        q = max(1, int(math.floor(fs / max_hz + 1e-9)))
        sos = _antialias(fs, q) if q > 1 else None
        win = max(q, int(round(win_s * fs / q)) * q)
        pad = int(round(pad_s * fs / q)) * q
        for i in range(a, b, win):
            lo, hi = max(a, i - pad), min(b, i + win + pad)
            with h5py.File(nwb, "r") as f:
                dd = f["acquisition"][name]["data"]
                x = np.asarray(dd[lo:hi] if col is None else dd[lo:hi, col], dtype=np.float64)
            if conv != 1.0:
                x *= conv
            if offs:
                x += offs
            first = lo + ((a - lo) % q)          # samples on the global decimation grid a + k q
            if sos is not None:
                bad = ~np.isfinite(x)
                if bad.any():
                    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad]) if (~bad).any() else 0.0
                if x.size > 27:
                    x = sosfiltfilt(sos, x)
                x[bad] = np.nan
            xd = x[first - lo::q]
            core = (t0 + (i - a) / fs, t0 + (min(b, i + win) - a) / fs)
            yield _split_nan(xd, t0 + (first - a) / fs, fs / q), core


def _split_nan(x: np.ndarray, t0: float, fs: float) -> list[Segment]:
    bad = ~np.isfinite(x)
    if not bad.any():
        return [Segment(t0, fs, x)] if x.size >= 2 else []
    # runs of NaN
    edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    max_run = int(NAN_GAP_S * fs)
    cuts = [(s, e) for s, e in zip(starts, stops) if e - s > max_run]
    good_idx = np.flatnonzero(~bad)
    if good_idx.size:
        x = x.copy()
        x[bad] = np.interp(np.flatnonzero(bad), good_idx, x[good_idx])
    out, prev = [], 0
    for s, e in cuts:
        if s - prev >= 2:
            out.append(Segment(t0 + prev / fs, fs, x[prev:s]))
        prev = e
    if x.size - prev >= 2:
        out.append(Segment(t0 + prev / fs, fs, x[prev:]))
    # leading/trailing NaN runs were filled by interp with edge values: trim them
    res = []
    for sg in out:
        i0 = int(round((sg.t0 - t0) * fs))
        m = bad[i0:i0 + len(sg.x)]
        lead = int(np.argmax(~m)) if (~m).any() else len(m)
        trail = int(np.argmax(~m[::-1])) if (~m).any() else 0
        xs = sg.x[lead:len(sg.x) - trail]
        if xs.size >= 2:
            res.append(Segment(sg.t0 + lead / fs, fs, xs))
    return res
