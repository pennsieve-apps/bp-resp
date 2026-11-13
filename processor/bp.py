"""
Arterial pressure beats: pulse onsets (feet), systolic peaks, diastolic troughs, beat-mean pressure.

Pulse onsets are found with the slope sum function (SSF) of Zong, Heldt, Moody and Mark (2003), "An
open-source algorithm to detect onset of arterial blood pressure pulses" (Computers in Cardiology 30:
259-262), implemented here from the paper (the PhysioNet `wabp` C program was read for the method,
not copied):

  1. low-pass the pressure (Zong: 16 Hz, for adult human pulses; the cutoff here scales with the
     species' heart rate, see presets.py) with a zero-phase Butterworth filter;
  2. SSF(i) = sum over the last w samples of the positive first differences (Zong: w = 128 ms; scaled
     with heart rate for other species): it rises steeply only on the systolic upstroke;
  3. an adaptive threshold: an SSF pulse counts when its peak exceeds THRESHOLD_FRAC times the
     running median of the last accepted pulse peaks (initialised from the first 10 s and re-learnt
     after a long pause), with a refractory period after each pulse;
  4. the onset is where the SSF, going back from its peak, falls below ONSET_FRAC of that peak; the
     foot (diastolic trough) is the pressure minimum between there and a short window before it.

Per beat (one foot to the next): DBP = pressure at the foot, SBP = the maximum, MAP = the integral
mean of the waveform over the beat (not (2D+S)/3), PP = SBP - DBP. Beats are labelled `N`,
`ectopic` (inter-beat interval far from the local median: a premature or missed beat) or `artifact`
(outside the plausible range of the species, clipped or flat waveform, a jump from the neighbouring
beats); only `N` beats enter series.csv, the baroreflex and the summaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from processor.inputs import Segment


@dataclass
class BpSettings:
    lowpass_hz: float
    ssf_window_s: float
    refractory_s: float
    threshold_frac: float
    onset_frac: float
    ibi_min_s: float
    ibi_max_s: float
    sbp_range: tuple
    dbp_range: tuple
    pp_min_mmhg: float
    jump_frac: float          # SBP/DBP jump from the local median, as a fraction of the local pulse pressure
    ectopic_frac: float       # IBI deviation from the local median, fraction


@dataclass
class BpBeats:
    onset_t: np.ndarray       # foot time, s
    sys_t: np.ndarray         # systolic peak time, s (the beat's time_s)
    sbp: np.ndarray
    dbp: np.ndarray
    map: np.ndarray
    pp: np.ndarray
    ibi_s: np.ndarray         # foot to next foot
    label: np.ndarray         # N / ectopic / artifact
    why: np.ndarray           # reason for a non-N label
    seg: np.ndarray           # index of the segment the beat is in

    @property
    def n(self) -> int:
        return int(self.sys_t.size)

    @property
    def good(self) -> np.ndarray:
        return self.label == "N"


def _lowpass(x: np.ndarray, fs: float, hz: float) -> np.ndarray:
    hz = min(hz, 0.45 * fs)
    sos = butter(2, hz / (fs / 2.0), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def ssf(y: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Slope sum function: the sum of the positive first differences over the last window_s."""
    du = np.diff(y, prepend=y[0])
    du[du < 0] = 0.0
    w = max(2, int(round(window_s * fs)))
    c = np.cumsum(du)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def detect_onsets(x: np.ndarray, fs: float, s: BpSettings) -> tuple[np.ndarray, np.ndarray]:
    """(feet, SSF peak) sample indices of the pulses in one NaN-free segment."""
    if x.size < int(2 * fs):
        return np.zeros(0, int), np.zeros(0, int)
    y = _lowpass(x, fs, s.lowpass_hz)
    f = ssf(y, fs, s.ssf_window_s)
    refr = max(1, int(round(s.refractory_s * fs)))
    cand, _ = find_peaks(f, distance=refr)
    if cand.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    h = f[cand]
    learn = int(10 * fs)

    def base_at(i0: int) -> float:
        m = (cand >= i0) & (cand < i0 + learn)
        hh = h[m] if m.any() else h
        return float(np.percentile(hh, 90))

    recent: list[float] = []
    base = base_at(int(cand[0]))
    accepted = []
    last = -10 ** 12
    relearn = int(2.0 * s.ibi_max_s * fs)
    for p, hp in zip(cand, h):
        if accepted and p - last > relearn:
            base = base_at(int(p))
            recent = []
        thr = s.threshold_frac * (float(np.median(recent)) if recent else base)
        if hp < thr or p - last < refr:
            continue
        accepted.append(int(p))
        last = int(p)
        recent.append(float(hp))
        if len(recent) > 8:
            recent.pop(0)
    peaks = np.asarray(accepted, int)
    feet = np.empty_like(peaks)
    w = max(2, int(round(s.ssf_window_s * fs)))
    for j, p in enumerate(peaks):
        lim = f[p] * s.onset_frac
        k = p
        stop = max(0, p - 2 * w, (peaks[j - 1] if j else 0))
        while k > stop and f[k] > lim:
            k -= 1
        a = max(stop, k - w // 2)
        feet[j] = a + int(np.argmin(y[a:p + 1]))
    keep = np.concatenate([[True], np.diff(feet) > 0]) if feet.size else np.zeros(0, bool)
    return feet[keep], peaks[keep]


def _local_median(v: np.ndarray, half: int = 5) -> np.ndarray:
    out = np.empty_like(v)
    n = v.size
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        w = v[a:b]
        w = w[np.isfinite(w)]
        out[i] = np.median(w) if w.size else np.nan
    return out


def flat_mask(x: np.ndarray, fs: float, flat_s: float, tol: float) -> np.ndarray:
    """Samples inside a run of at least flat_s seconds over which the signal stays within tol: a
    clipped amplifier or transducer, a disconnected line, a calibration plateau or a finger-cuff
    re-calibration hold. A real pressure pulse is never that still for that long."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d  # noqa: PLC0415
    w = max(3, int(round(flat_s * fs)))
    if x.size < w:
        return np.zeros(x.size, bool)
    spread = maximum_filter1d(x, w) - minimum_filter1d(x, w)
    still = spread <= tol
    # a window centred on i is still: mark the whole window
    return maximum_filter1d(still.astype(np.uint8), w).astype(bool)


def robust_range(segments: list[Segment]) -> float:
    xs = np.concatenate([sg.x[:: max(1, sg.x.size // 200000)] for sg in segments]) if segments else np.zeros(0)
    return float(np.percentile(xs, 99.9) - np.percentile(xs, 0.1)) if xs.size else 0.0


def beats(segments: list[Segment], s: BpSettings, calibrated: bool, flat_s: float = 0.1,
          flat_frac: float = 0.001) -> BpBeats:
    """Every beat of every segment, measured and labelled. flat_s and flat_frac (of the recording's
    0.1-99.9 percentile range) define a flat or clipped stretch (see flat_mask)."""
    tol = flat_frac * robust_range(segments)
    cols = {k: [] for k in ("onset_t", "sys_t", "sbp", "dbp", "map", "pp", "ibi_s", "label", "why", "seg")}
    for si, seg in enumerate(segments):
        x, fs = seg.x, seg.fs
        feet, _ = detect_onsets(x, fs, s)
        if feet.size < 2:
            continue
        half = max(1, int(round(0.02 * fs)))       # +-20 ms around a fiducial for the extreme value
        y = _lowpass(x, fs, s.lowpass_hz)
        flat = flat_mask(x, fs, flat_s, tol)
        for a, b in zip(feet[:-1], feet[1:]):
            seg_y = y[a:b]
            ps = a + int(np.argmax(seg_y))
            sbp = float(np.max(x[max(a, ps - half):min(b, ps + half + 1)]))
            dbp = float(np.min(x[max(0, a - half):a + half + 1]))
            mp = float(np.trapezoid(x[a:b + 1], dx=1.0 / fs) / ((b - a) / fs))
            cols["onset_t"].append(seg.t0 + a / fs)
            cols["sys_t"].append(seg.t0 + ps / fs)
            cols["sbp"].append(sbp)
            cols["dbp"].append(dbp)
            cols["map"].append(mp)
            cols["pp"].append(sbp - dbp)
            cols["ibi_s"].append((b - a) / fs)
            why = "flat or clipped waveform" if flat[a:b].any() else ""
            cols["why"].append(why)
            cols["label"].append("artifact" if why else "N")
            cols["seg"].append(si)
    a = {k: np.asarray(v) for k, v in cols.items()}
    if not a["sys_t"].size:
        return BpBeats(*(np.zeros(0) for _ in range(7)), np.zeros(0, dtype="<U8"), np.zeros(0, dtype="<U64"),
                       np.zeros(0, int))
    label = a["label"].astype("<U8")
    why = a["why"].astype("<U64")

    def mark(mask, lab, reason):
        m = mask & (label == "N")
        label[m] = lab
        why[m] = reason

    ibi = a["ibi_s"]
    mark((ibi < s.ibi_min_s) | (ibi > s.ibi_max_s), "artifact", "inter-beat interval outside the species range")
    if calibrated:
        mark((a["sbp"] < s.sbp_range[0]) | (a["sbp"] > s.sbp_range[1]), "artifact", "systolic pressure implausible")
        mark((a["dbp"] < s.dbp_range[0]) | (a["dbp"] > s.dbp_range[1]), "artifact", "diastolic pressure implausible")
        mark(a["pp"] < s.pp_min_mmhg, "artifact", "pulse pressure too small")
    else:
        mark(a["pp"] <= 0, "artifact", "no pulse")
    # neighbours: medians over +-5 beats of the beats still labelled N, per segment
    for si in np.unique(a["seg"]):
        m = a["seg"] == si
        idx = np.flatnonzero(m)
        good = label[idx] == "N"
        if good.sum() < 3:
            continue

        def lm(v):
            vv = v[idx].astype(float).copy()
            vv[~good] = np.nan
            return _local_median(vv)

        pp_med = lm(a["pp"])
        jump = s.jump_frac * pp_med
        bad = (np.abs(a["sbp"][idx] - lm(a["sbp"])) > jump) | (np.abs(a["dbp"][idx] - lm(a["dbp"])) > jump)
        bad &= np.isfinite(jump)
        mm = np.zeros(m.size, bool)
        mm[idx[bad]] = True
        mark(mm, "artifact", "jump from the neighbouring beats")
        ibi_med = lm(ibi)
        ect = np.abs(ibi[idx] - ibi_med) > s.ectopic_frac * ibi_med
        mm = np.zeros(m.size, bool)
        mm[idx[ect & np.isfinite(ibi_med)]] = True
        mark(mm, "ectopic", "inter-beat interval far from the local median")
    return BpBeats(a["onset_t"].astype(float), a["sys_t"].astype(float), a["sbp"].astype(float),
                   a["dbp"].astype(float), a["map"].astype(float), a["pp"].astype(float), ibi.astype(float),
                   label, why, a["seg"].astype(int))


def _concat(parts: list[BpBeats]) -> BpBeats:
    if not parts:
        return BpBeats(*(np.zeros(0) for _ in range(7)), np.zeros(0, dtype="<U8"), np.zeros(0, dtype="<U64"),
                       np.zeros(0, int))
    f = BpBeats.__dataclass_fields__
    return BpBeats(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in f})


def beats_windowed(windows, s: BpSettings, calibrated: bool, zoom_t: float | None = None,
                   zoom_span: float = 0.0) -> tuple[BpBeats, list, list]:
    """beats() over iter_windows(): each window is measured and labelled with its padding (so the
    neighbour rules see neighbours across window edges) and only the beats whose pulse onset lies in
    the window's core are kept. Returns the beats, the signal spans [(t0, t1)] covered, and the
    samples of [zoom_t, zoom_t + zoom_span] (Segments) for the figure."""
    parts, spans, zoom = [], [], []
    k = 0
    for segs, (c0, c1) in windows:
        if not segs:
            continue
        bt = beats(segs, s, calibrated)
        if bt.n:
            keep = (bt.onset_t >= c0) & (bt.onset_t < c1)
            bt = BpBeats(**{f: getattr(bt, f)[keep] for f in BpBeats.__dataclass_fields__})
            bt.seg = bt.seg + k
            parts.append(bt)
        k += len(segs)
        for sg in segs:
            a, b = max(sg.t0, c0), min(sg.t1, c1)
            if b > a:
                spans.append((a, b))
            if zoom_t is not None and sg.t0 <= zoom_t + zoom_span and sg.t1 >= zoom_t:
                tt = sg.times()
                m = (tt >= zoom_t) & (tt <= zoom_t + zoom_span)
                if m.any():
                    from processor.inputs import Segment  # noqa: PLC0415
                    zoom.append(Segment(float(tt[m][0]), sg.fs, sg.x[m].copy()))
    # neighbouring spans of one continuous segment: merge
    merged = []
    for a, b in sorted(spans):
        if merged and a - merged[-1][1] < 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    bt = _concat(parts)
    # beats whose window edge cut the beat-to-beat chain: same segment id across windows of one span
    if bt.n:
        span_id = np.searchsorted([a for a, _ in merged], bt.onset_t, side="right") - 1
        bt.seg = span_id.astype(int)
    return bt, merged, zoom


def level_steps(bt: BpBeats, step_mmhg: float = 8.0, n: int = 10) -> list[tuple[float, float]]:
    """Abrupt level changes of the systolic pressure (a finger-cuff or vascular-unloading device
    re-calibrating, a transducer re-zeroed or moved): times where the median SBP of the next n good
    beats differs from that of the previous n by more than step_mmhg, the change happening within 3
    beats. Returns [(time_s, change_mmhg)], one per step."""
    g = np.flatnonzero(bt.good)
    if g.size < 2 * n + 1:
        return []
    m = bt.sbp[g]
    t = bt.sys_t[g]
    out = []
    i = n
    while i < g.size - n:
        before = np.median(m[i - n:i])
        after = np.median(m[i:i + n])
        if abs(after - before) > step_mmhg:
            # abrupt: the change happens across at most 3 beats around i
            near_b = np.median(m[max(0, i - 3):i])
            near_a = np.median(m[i:i + 3])
            if abs(near_a - near_b) > 0.6 * step_mmhg:
                out.append((float(t[i]), float(after - before)))
                i += n
                continue
        i += 1
    return out


def drift(bt: BpBeats, minutes: float = 2.0) -> float | None:
    """Median SBP of the last `minutes` minus that of the first (mmHg), or None when too short."""
    g = bt.good
    if g.sum() < 20:
        return None
    t, v = bt.sys_t[g], bt.sbp[g]
    if t[-1] - t[0] < 4 * 60 * minutes:
        return None
    a = np.median(v[t <= t[0] + 60 * minutes])
    b = np.median(v[t >= t[-1] - 60 * minutes])
    return float(b - a)
