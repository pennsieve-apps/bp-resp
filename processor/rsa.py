"""
Respiratory sinus arrhythmia (RSA): the rise and fall of heart rate with breathing.

Human recordings: NeuroKit2's `nk.hrv_rsa` (Makowski et al. 2021) on the R-peaks and the breaths:
  p2t            peak-to-trough: per breath, the longest RR interval of expiration minus the shortest
                 of inspiration (0 when negative), averaged over breaths, ms (Grossman's method as
                 described by Lewis et al. 2012; NeuroKit2 adds 750 ms to the phase ends for the
                 vagal delay);
  porges-bohrer  the Porges-Bohrer estimate: heart period at 2 Hz, detrended with a 21-point cubic
                 polynomial, band-passed to 0.12-0.40 Hz (human breathing), log of its variance per
                 30 s epoch, averaged, ln(ms^2) (Lewis et al. 2012).

Other species: Porges-Bohrer's filter is defined for human breathing frequencies, so it is not
reported; peak-to-trough is computed with the same rule (this module, `p2t`) but with the vagal
delay scaled to the breath (the smaller of 750 ms and a quarter of the median breath), because
750 ms is longer than a whole breath of a rat or mouse.

RSA is computed over the stretches of recording that are free of stimulation and gaps.
"""
from __future__ import annotations

import warnings

import numpy as np

from processor.resp import RespSegment


def p2t(rsp_onsets: np.ndarray, rsp_peaks: np.ndarray, rpeaks: np.ndarray, fs: float, offset_s: float) -> np.ndarray:
    """Per-breath peak-to-trough RSA (ms), NeuroKit2's rule with a configurable phase offset."""
    off = offset_s * fs
    vals = np.full(max(0, rsp_onsets.size - 1), np.nan)
    for i in range(rsp_onsets.size - 1):
        a, b = rsp_onsets[i], rsp_onsets[i + 1]
        pk = rsp_peaks[(rsp_peaks > a) & (rsp_peaks < b)]
        if pk.size == 0:
            continue
        p = pk[0]
        inh = rpeaks[(rpeaks >= a) & (rpeaks < p + off)]
        exh = rpeaks[(rpeaks >= p) & (rpeaks < b + off)]
        ri, re_ = np.diff(inh) / fs * 1000.0, np.diff(exh) / fs * 1000.0
        if ri.size and re_.size:
            vals[i] = max(0.0, float(np.max(re_) - np.min(ri)))
    return vals


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


def stretch_rsa(seg: RespSegment, a: int, b: int, r_times: np.ndarray, human: bool, offset_s: float) -> dict | None:
    """RSA over samples a..b of one respiration segment. Returns {'p2t': (mean, n_breaths),
    'porges-bohrer': value or None, 'why_pb': reason}."""
    import neurokit2 as nk  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    fs = seg.fs
    rp = np.round((r_times - seg.t0) * fs).astype(np.int64)
    rp = rp[(rp >= a) & (rp < b)] - a
    tr = seg.troughs[(seg.troughs >= a) & (seg.troughs < b)] - a
    pk = seg.peaks[(seg.peaks >= a) & (seg.peaks < b)] - a
    n = b - a
    if rp.size < 10 or tr.size < 4:
        return None
    out: dict = {}
    if human:
        ecg = pd.DataFrame({"ECG_R_Peaks": np.zeros(n, np.int8)})
        ecg.loc[rp, "ECG_R_Peaks"] = 1
        rsp = pd.DataFrame({"RSP_Clean": seg.clean[a:b]})
        phase = nk.rsp_phase(pk, tr, desired_length=n)
        rsp = pd.concat([rsp, phase], axis=1)
        rsp["RSP_Peaks"] = 0
        rsp["RSP_Troughs"] = 0
        rsp.loc[pk, "RSP_Peaks"] = 1
        rsp.loc[tr, "RSP_Troughs"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = nk.hrv_rsa(ecg, rsp, sampling_rate=fs, continuous=False)
        if not res:
            return None
        vals = p2t(tr, pk, rp, fs, 0.75)
        out["p2t"] = (float(res.get("RSA_P2T_Mean", np.nan)), int(np.isfinite(vals).sum()))
        out["porges-bohrer"] = float(res.get("RSA_PorgesBohrer", np.nan))
        out["why_pb"] = ""
    else:
        vals = p2t(tr, pk, rp, fs, offset_s)
        out["p2t"] = (float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan, int(np.isfinite(vals).sum()))
        out["porges-bohrer"] = None
        out["why_pb"] = "Porges-Bohrer's 0.12-0.40 Hz filter is defined for human breathing; not computed"
    return out
