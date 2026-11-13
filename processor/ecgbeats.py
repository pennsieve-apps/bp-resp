"""
R-peaks for the baroreflex and RSA when ecg-hrv's beats are not linked (the `hrv` port is optional).

A short version of ecg-hrv's detector: the ECG is band-passed with the species preset's band, R-peaks
are found with NeuroKit2's default `neurokit` QRS detector (Makowski et al. 2021) with the preset's
smoothing and averaging windows and refractory period, on whichever polarity gives the larger
peaks (an inverted lead), after notching mains interference when it stands out (as ecg-hrv). Beats are labelled simply: an RR interval outside the species range, or more
than ECTOPIC_FRAC from the median of its neighbours, makes the beats on both sides unusable for RR.
Link ecg-hrv's `hrv` port for its full labelling (Lipponen & Tarvainen 2019 via NeuroKit2) and its
second-detector cross-check.
"""
from __future__ import annotations

import warnings

import numpy as np

from processor.inputs import EcgBeats, Segment


def detect_mains(x: np.ndarray, fs: float) -> float:
    """50 or 60 Hz when that line stands out of the ECG spectrum (more than 20 times the median power
    density of 35-45 Hz), else 0 (the same rule as ecg-hrv)."""
    import scipy.signal as sps  # noqa: PLC0415

    if fs < 130 or x.size < 4 * fs:
        return 0.0
    f, p = sps.welch(x - np.mean(x), fs=fs, nperseg=int(min(x.size, 4 * fs)))
    ref = np.median(p[(f >= 35) & (f <= 45)])
    best, ratio = 0.0, 20.0
    for f0 in (50.0, 60.0):
        m = (f >= f0 - 1) & (f <= f0 + 1)
        if m.any() and ref > 0 and p[m].max() / ref > ratio:
            best, ratio = f0, p[m].max() / ref
    return best


def _notch(y: np.ndarray, fs: float, f0: float) -> np.ndarray:
    import scipy.signal as sps  # noqa: PLC0415

    f = f0
    while f0 and f < 0.45 * fs:      # mains and its harmonics: zero-phase IIR notches (Q 30)
        b, a = sps.iirnotch(f, 30.0, fs)
        y = sps.filtfilt(b, a, y)
        f += f0
    return y


def detect(segments: list[Segment], preset, ectopic_frac: float = 0.2) -> EcgBeats:
    import neurokit2 as nk  # noqa: PLC0415

    times, good = [], []
    for seg in segments:
        x, fs = seg.x, seg.fs
        if x.size < int(3 * fs):
            continue
        lo, hi = preset.ecg_band_hz
        hi = min(hi, 0.45 * fs)
        clean = nk.signal_filter(x, sampling_rate=fs, lowcut=lo, highcut=hi, method="butterworth", order=2)
        clean = _notch(clean, fs, detect_mains(x[: int(600 * fs)], fs))
        best = None
        for sign in (1.0, -1.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pk = np.asarray(nk.ecg_findpeaks(sign * clean, sampling_rate=fs, method="neurokit",
                                                 smoothwindow=preset.ecg_smooth_s, avgwindow=preset.ecg_avg_s,
                                                 mindelay=preset.ecg_refractory_s)["ECG_R_Peaks"], dtype=np.int64)
            amp = float(np.median(sign * clean[pk])) if pk.size else -np.inf
            if best is None or amp > best[0]:
                best = (amp, pk)
        pk = best[1]
        if pk.size < 3:
            continue
        t = seg.t0 + pk / fs
        rr = np.diff(t)
        ok_rr = (rr >= preset.rr_min_s) & (rr <= preset.rr_max_s)
        med = np.array([np.median(rr[max(0, i - 5):i + 6]) for i in range(rr.size)])
        ok_rr &= np.abs(rr - med) <= ectopic_frac * med
        g = np.ones(t.size, bool)
        # beat k is usable when both RR intervals touching it are normal
        g[:-1] &= ok_rr
        g[1:] &= ok_rr
        # the first and last beat of a segment only border one interval: keep them when it is normal
        g[0] = ok_rr[0]
        g[-1] = ok_rr[-1]
        times.append(t)
        good.append(g)
    if not times:
        return EcgBeats(np.zeros(0), np.zeros(0, bool), "detected", "no R-peaks found")
    return EcgBeats(np.concatenate(times), np.concatenate(good), "detected",
                    "R-peaks detected here (NeuroKit2 'neurokit' detector, species preset)")
