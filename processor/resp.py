"""
Breaths from a respiration channel (belt, airflow, thermistor, plethysmography, impedance) with
NeuroKit2 (Makowski et al. 2021): the steps of `nk.rsp_process` with its default method
`khodadad2018` (Khodadad et al. 2018, a zero-crossing algorithm with an amplitude threshold), called
one by one so the band-pass can follow the species:

  clean    `nk.rsp_clean(method="khodadad2018")` = 2nd-order Butterworth 0.05-3 Hz band-pass, used as
           is when the species preset's band is that band (human, pig, dog); faster-breathing species
           get the same filter with their own band (`nk.signal_filter`, Butterworth, order 2);
  extrema  `nk.rsp_peaks(method="khodadad2018")`: zero crossings of the cleaned signal, the extreme
           between each pair, extrema closer than RESP_AMPLITUDE_MIN x the median peak-trough
           distance dropped;
  breaths  one breath per trough-to-trough cycle (trough = inspiration onset, peak = end of
           inspiration): onset, duration, rate (60 / duration), amplitude (peak minus the preceding
           trough, as `nk.rsp_amplitude` does), inspiration and expiration times.

RSP_RVT and the symmetry features of rsp_process are not computed (not used, and slow on long
recordings). Breaths whose rate is outside the species range are labelled `artifact`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from processor.inputs import Segment

KHODADAD_BAND = (0.05, 3.0)


@dataclass
class RespSegment:
    t0: float
    fs: float
    clean: np.ndarray
    peaks: np.ndarray        # sample indices (end of inspiration)
    troughs: np.ndarray      # sample indices (inspiration onsets)


@dataclass
class Breaths:
    onset_s: np.ndarray
    duration_s: np.ndarray
    rate_bpm: np.ndarray
    amplitude: np.ndarray
    insp_s: np.ndarray
    exp_s: np.ndarray
    label: np.ndarray
    segments: list           # RespSegment per analysed segment

    @property
    def n(self) -> int:
        return int(self.onset_s.size)

    @property
    def good(self) -> np.ndarray:
        return self.label == "N"


def clean_signal(x: np.ndarray, fs: float, band: tuple) -> np.ndarray:
    import neurokit2 as nk  # noqa: PLC0415

    lo, hi = band
    hi = min(hi, 0.45 * fs)
    if (lo, hi) == KHODADAD_BAND:
        return np.asarray(nk.rsp_clean(x, sampling_rate=fs, method="khodadad2018"), dtype=float)
    return np.asarray(nk.signal_filter(x, sampling_rate=fs, lowcut=lo, highcut=hi, method="butterworth", order=2),
                      dtype=float)


def breaths(segments: list[Segment], band: tuple, rate_range: tuple, amplitude_min: float = 0.3,
            inverted: bool = False) -> Breaths:
    import neurokit2 as nk  # noqa: PLC0415

    cols = {k: [] for k in ("onset_s", "duration_s", "rate_bpm", "amplitude", "insp_s", "exp_s", "label")}
    rsegs = []
    min_len = 3 * 60.0 / max(rate_range[0], 1e-3)
    for seg in segments:
        if seg.x.size / seg.fs < min_len:
            continue
        x = -seg.x if inverted else seg.x
        clean = clean_signal(x, seg.fs, band)
        try:
            _, info = nk.rsp_peaks(clean, sampling_rate=seg.fs, method="khodadad2018", amplitude_min=amplitude_min)
        except (IndexError, ValueError, UnboundLocalError):
            continue
        pk = np.asarray(info["RSP_Peaks"], dtype=np.int64)
        tr = np.asarray(info["RSP_Troughs"], dtype=np.int64)
        if tr.size < 2 or pk.size < 1:
            continue
        rsegs.append(RespSegment(seg.t0, seg.fs, clean, pk, tr))
        for a, b in zip(tr[:-1], tr[1:]):
            inside = pk[(pk > a) & (pk < b)]
            if inside.size == 0:
                continue
            p = int(inside[np.argmax(clean[inside])])
            dur = (b - a) / seg.fs
            rate = 60.0 / dur
            cols["onset_s"].append(seg.t0 + a / seg.fs)
            cols["duration_s"].append(dur)
            cols["rate_bpm"].append(rate)
            cols["amplitude"].append(float(clean[p] - clean[a]))
            cols["insp_s"].append((p - a) / seg.fs)
            cols["exp_s"].append((b - p) / seg.fs)
            cols["label"].append("N" if rate_range[0] <= rate <= rate_range[1] else "artifact")
    a = {k: np.asarray(v, dtype=float if k != "label" else "<U8") for k, v in cols.items()}
    return Breaths(**a, segments=rsegs)
