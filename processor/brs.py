"""
Spontaneous baroreflex sensitivity (BRS), in-house (no maintained, permissively licensed package
exists), from beat-to-beat systolic pressure (SBP) and the RR (or pulse) interval.

Sequence method (Bertinieri et al. 1985; Parati et al. 1988; as compared in EuroBaVar, Laude et al.
2004): runs of at least SEQ_MIN_BEATS consecutive beats in which SBP rises by at least the SBP
threshold at every beat and the RR interval, LAG beats later, lengthens by at least the RR threshold
(an "up" sequence), or both fall ("down"); the RR-on-SBP regression slope of a run is its BRS when
the correlation r >= SEQ_MIN_R. BRS = the mean slope over the sequences (up, down, all). LAG is 0, 1
or 2 beats, or auto: the lag with the most sequences.

Spectral methods (EuroBaVar): SBP and RR are interpolated onto a regular grid; Welch-averaged
(Hann, 50 % overlap, linear detrend per window) auto- and cross-spectra over the windows that are
free of gaps, artefacts and stimulation:
  alpha-lf, alpha-hf    sqrt(P_RR / P_SBP) over the LF or HF band, valid when the mean squared
                        coherence over that band is >= COHERENCE_MIN;
  transfer-lf           the mean transfer-function gain |P_SBP,RR| / P_SBP over the LF frequencies
                        whose coherence is >= COHERENCE_MIN.
All values are in ms/mmHg; they need calibrated pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from processor.bp import BpBeats
from processor.inputs import EcgBeats


@dataclass
class Intervals:
    """The RR interval paired with each BP beat at lag L, and whether the pair is usable."""
    rr_ms: np.ndarray        # (n_bp,)
    ok: np.ndarray           # (n_bp,) bool
    k: np.ndarray            # index of the ECG beat (or BP beat) the RR interval starts at
    source: str


def pair_rr(bp: BpBeats, ecg: EcgBeats | None, lag: int, rr_range_s: tuple) -> Intervals:
    """Lag 0 pairs a beat's SBP with the RR interval in which its systolic peak falls (R_k <= t_sys <
    R_k+1); lag L with the interval L beats later. Without R-peaks the pulse interval (systolic peak
    to systolic peak) is used instead."""
    n = bp.n
    rr = np.full(n, np.nan)
    ok = np.zeros(n, bool)
    if ecg is not None and ecg.times.size >= 3:
        R, g = ecg.times, ecg.good
        k = np.searchsorted(R, bp.sys_t, side="right") - 1
        a, b = k + lag, k + lag + 1
        inside = (k >= 0) & (b < R.size)
        aa, bb = np.where(inside, a, 0), np.where(inside, b, 0)
        rr_s = R[bb] - R[aa]
        ok = inside & g[aa] & g[bb] & (rr_s >= rr_range_s[0]) & (rr_s <= rr_range_s[1]) & bp.good
        # the systolic peak must fall inside its own RR interval (a missed R-peak would pair it wrongly)
        kk = np.where(inside, k, 0)
        kk1 = np.clip(kk + 1, 0, R.size - 1)
        ok &= (R[kk1] - R[kk]) <= rr_range_s[1]
        rr = np.where(ok, rr_s * 1000.0, np.nan)
        return Intervals(rr, ok, k, ecg.source)
    # pulse interval from the BP beats themselves
    k = np.arange(n)
    a, b = k + lag, k + lag + 1
    inside = b < n
    aa, bb = np.where(inside, a, 0), np.where(inside, b, 0)
    pi = bp.sys_t[bb] - bp.sys_t[aa]
    ok = inside & bp.good & bp.good[aa] & bp.good[bb] & (bp.seg[aa] == bp.seg[bb]) & \
        (pi >= rr_range_s[0]) & (pi <= rr_range_s[1])
    # consecutive beats only (no missed beat between them)
    ok &= np.abs(pi - bp.ibi_s[aa]) <= 0.5 * bp.ibi_s[aa]
    rr = np.where(ok, pi * 1000.0, np.nan)
    return Intervals(rr, ok, k, "bp-pulse")


@dataclass
class Sequence:
    kind: str                # up / down
    i0: int                  # first BP beat
    n_beats: int
    slope: float             # ms/mmHg
    r: float


def sequences(sbp: np.ndarray, iv: Intervals, use: np.ndarray, thr_sbp: float, thr_rr_ms: float,
              min_beats: int, min_r: float) -> tuple[list[Sequence], int]:
    """Accepted sequences and the number of candidate ramps (before the r test)."""
    n = sbp.size
    usable = use & iv.ok & np.isfinite(sbp)
    # step i -> i+1 is usable when both beats are and they are consecutive (and so are their RR intervals)
    step_ok = usable[:-1] & usable[1:] & (np.diff(iv.k) == 1)
    ds, dr = np.diff(sbp), np.diff(iv.rr_ms)
    up = step_ok & (ds >= thr_sbp) & (dr >= thr_rr_ms)
    dn = step_ok & (ds <= -thr_sbp) & (dr <= -thr_rr_ms)
    out: list[Sequence] = []
    ramps = 0
    for kind, m in (("up", up), ("down", dn)):
        i = 0
        while i < n - 1:
            if not m[i]:
                i += 1
                continue
            j = i
            while j < n - 1 and m[j]:
                j += 1
            nb = j - i + 1            # beats i..j
            if nb >= min_beats:
                ramps += 1
                x, y = sbp[i:j + 1], iv.rr_ms[i:j + 1]
                if np.ptp(x) > 0 and np.ptp(y) > 0:
                    r = float(np.corrcoef(x, y)[0, 1])
                    if r >= min_r:
                        out.append(Sequence(kind, i, nb, float(np.polyfit(x, y, 1)[0]), r))
            i = j
    return out, ramps


@dataclass
class SpectralResult:
    f: np.ndarray
    p_sbp: np.ndarray
    p_rr: np.ndarray
    coh: np.ndarray
    gain: np.ndarray
    n_windows: int
    window_s: float


def spectral(t: np.ndarray, sbp: np.ndarray, rr_ms: np.ndarray, ok: np.ndarray, hz: float, window_s: float,
             max_gap_s: float, exclude: list[tuple[float, float]]) -> SpectralResult | None:
    """Welch auto/cross spectra of SBP and RR over windows free of gaps and excluded epochs."""
    tt, s, r = t[ok], sbp[ok], rr_ms[ok]
    if tt.size < 10:
        return None
    grid = np.arange(np.ceil(tt[0] * hz) / hz, tt[-1], 1.0 / hz)
    if grid.size < window_s * hz:
        return None
    gs, gr = np.interp(grid, tt, s), np.interp(grid, tt, r)
    j = np.clip(np.searchsorted(tt, grid), 1, tt.size - 1)
    valid = (tt[j] - tt[j - 1]) <= max_gap_s
    for a, b in exclude:
        valid &= ~((grid >= a) & (grid <= b))
    nw = int(round(window_s * hz))
    step = max(1, nw // 2)
    win = np.hanning(nw)
    u = float(np.sum(win ** 2))
    pxx = pyy = pxy = None
    count = 0
    for i0 in range(0, grid.size - nw + 1, step):
        if not valid[i0:i0 + nw].all():
            continue
        xs = gs[i0:i0 + nw]
        ys = gr[i0:i0 + nw]
        tw = np.arange(nw)
        xs = xs - np.polyval(np.polyfit(tw, xs, 1), tw)
        ys = ys - np.polyval(np.polyfit(tw, ys, 1), tw)
        X, Y = np.fft.rfft(xs * win), np.fft.rfft(ys * win)
        a, b, c = np.abs(X) ** 2, np.abs(Y) ** 2, np.conj(X) * Y
        pxx = a if pxx is None else pxx + a
        pyy = b if pyy is None else pyy + b
        pxy = c if pxy is None else pxy + c
        count += 1
    if count == 0:
        return None
    scale = 2.0 / (hz * u * count)
    pxx, pyy, pxy = pxx * scale, pyy * scale, pxy * scale
    f = np.fft.rfftfreq(nw, 1.0 / hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(pxy) ** 2 / (pxx * pyy)
        gain = np.abs(pxy) / pxx
    return SpectralResult(f, pxx, pyy, coh, gain, count, window_s)


def band_alpha(sp: SpectralResult, band: tuple) -> tuple[float, float, int]:
    """(alpha ms/mmHg, mean coherence, n bins) over a band."""
    m = (sp.f >= band[0]) & (sp.f < band[1])
    if not m.any():
        return np.nan, np.nan, 0
    return float(np.sqrt(np.sum(sp.p_rr[m]) / np.sum(sp.p_sbp[m]))), float(np.mean(sp.coh[m])), int(m.sum())


def band_transfer(sp: SpectralResult, band: tuple, min_coh: float) -> tuple[float, float, int]:
    """(mean gain over the band's coherent bins, their mean coherence, n coherent bins)."""
    m = (sp.f >= band[0]) & (sp.f < band[1]) & (sp.coh >= min_coh)
    if not m.any():
        mb = (sp.f >= band[0]) & (sp.f < band[1])
        return np.nan, float(np.mean(sp.coh[mb])) if mb.any() else np.nan, 0
    return float(np.mean(sp.gain[m])), float(np.mean(sp.coh[m])), int(m.sum())
