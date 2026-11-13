"""
Respiratory sinus arrhythmia (RSA): the rise and fall of heart period with breathing.

The two estimates of NeuroKit2's `nk.hrv_rsa` (Makowski et al. 2021), computed with NeuroKit2's rules
at the **exact R-peak times** instead of on NeuroKit2's sample grid. hrv_rsa needs R-peaks and breaths
as sample indices of one signal; at the respiration's working rate (50-100 Hz) rounding the R-peaks
moves each RR interval by up to 10-20 ms, which inflates peak-to-trough RSA (in a rat with no RSA
at all, by about 5 ms at 100 Hz). tests/test_processor.py checks that these functions give
NeuroKit2's values when NeuroKit2 is run on a 1 kHz grid.

  p2t            peak-to-trough: per breath (inspiration onset to the next), the longest RR interval
                 of expiration minus the shortest of inspiration (0 when negative), averaged, ms.
                 As NeuroKit2 does, the R-peaks of each phase run up to a vagal delay after its end
                 (NeuroKit2 cites Grossman et al. 1990 for 750 ms); for non-human species the delay is
                 the smaller of 750 ms and a quarter of the median breath, because 750 ms is longer
                 than a whole rat or mouse breath.
  porges-bohrer  heart period (ms) at 2 Hz (monotone cubic interpolation, as NeuroKit2's signal_rate),
                 detrended with a 21-point cubic Savitzky-Golay filter, band-passed to 0.12-0.40 Hz
                 (NeuroKit2's filters), log of its variance per 30 s epoch, averaged; ln(ms^2) (Lewis et
                 al. 2012). Human breathing frequencies only: not computed for other species.

RSA is computed over the stretches of recording that are free of stimulation and gaps.
"""
from __future__ import annotations

import numpy as np

from processor.resp import RespSegment


def fill_beats(times: np.ndarray, good: np.ndarray) -> np.ndarray:
    """R-peak times with every run of unusable beats (ectopic, artefact) replaced by the same number
    of beats evenly spaced between the good beats on either side, as HRV software interpolates
    ectopic beats; leading and trailing unusable beats are dropped."""
    if times.size == 0 or not good.any():
        return np.zeros(0)
    gi = np.flatnonzero(good)
    out = [times[gi[0]]]
    for a, b in zip(gi[:-1], gi[1:]):
        k = b - a - 1
        if k > 0:
            out.extend(np.linspace(times[a], times[b], k + 2)[1:-1].tolist())
        out.append(times[b])
    return np.asarray(out)


def p2t(onsets_t: np.ndarray, peaks_t: np.ndarray, r_t: np.ndarray, offset_s: float) -> np.ndarray:
    """Per-breath peak-to-trough RSA (ms) at exact times: NeuroKit2's rule (_hrv_rsa_p2t)."""
    vals = np.full(max(0, onsets_t.size - 1), np.nan)
    for i in range(onsets_t.size - 1):
        a, b = onsets_t[i], onsets_t[i + 1]
        pk = peaks_t[(peaks_t > a) & (peaks_t < b)]
        if pk.size == 0:
            continue
        p = pk[0]
        inh = r_t[(r_t >= a) & (r_t < p + offset_s)]
        exh = r_t[(r_t >= p) & (r_t < b + offset_s)]
        ri, re_ = np.diff(inh) * 1000.0, np.diff(exh) * 1000.0
        if ri.size and re_.size:
            vals[i] = max(0.0, float(np.max(re_) - np.min(ri)))
    return vals


def porges_bohrer(r_t: np.ndarray, t0: float, t1: float) -> float:
    """NeuroKit2's Porges-Bohrer estimate (_hrv_rsa_pb) on exact R-peak times, ln(ms^2)."""
    import neurokit2 as nk  # noqa: PLC0415
    from scipy.interpolate import PchipInterpolator  # noqa: PLC0415

    r = r_t[(r_t >= t0) & (r_t <= t1)]
    if r.size < 10:
        return float("nan")
    period = np.diff(r) * 1000.0
    period = np.concatenate([[np.mean(period)], period])     # as signal_period: the first gets the mean
    grid = np.arange(t0, t1, 0.5)
    hp = PchipInterpolator(r, period, extrapolate=False)(grid)
    hp[grid < r[0]] = period[0]
    hp[grid > r[-1]] = period[-1]
    if grid.size < 30:
        return float("nan")
    trend = nk.signal_filter(hp, sampling_rate=2, lowcut=0.095, highcut=None, method="savgol", order=3,
                             window_size=21)
    zm = nk.signal_filter(hp - trend, sampling_rate=2, lowcut=0.12, highcut=0.40)
    epoch = (np.arange(zm.size) / 2.0 // 30).astype(int)
    v = []
    for e in np.unique(epoch):
        x = zm[epoch == e]
        if x.size > 1:
            v.append(np.log(np.var(x, ddof=1)))
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def stretch_rsa(seg: RespSegment, a: int, b: int, r_times: np.ndarray, human: bool, offset_s: float) -> dict | None:
    """RSA over samples a..b of one respiration segment. Returns {'p2t': (mean, n_breaths),
    'porges-bohrer': value or None, 'why_pb': reason}."""
    fs = seg.fs
    t0, t1 = seg.t0 + a / fs, seg.t0 + (b - 1) / fs
    tr = seg.troughs[(seg.troughs >= a) & (seg.troughs < b)]
    pk = seg.peaks[(seg.peaks >= a) & (seg.peaks < b)]
    r = r_times[(r_times >= t0) & (r_times <= t1)]
    if r.size < 10 or tr.size < 4:
        return None
    vals = p2t(seg.t0 + tr / fs, seg.t0 + pk / fs, r, offset_s if not human else 0.75)
    ok = np.isfinite(vals)
    out = {"p2t": (float(np.mean(vals[ok])) if ok.any() else np.nan, int(ok.sum()))}
    if human:
        out["porges-bohrer"] = porges_bohrer(r, t0, t1)
        out["why_pb"] = ""
    else:
        out["porges-bohrer"] = None
        out["why_pb"] = "Porges-Bohrer's 0.12-0.40 Hz filter is defined for human breathing; not computed"
    return out
