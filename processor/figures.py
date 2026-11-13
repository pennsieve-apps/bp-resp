"""Per-recording figures (PNG, 150 dpi) with captions for summary.json."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESEARCH = "Research output, not a clinical finding."
DPI = 150


def _shade(ax, epochs):
    for k, (a, b, _) in enumerate(epochs):
        ax.axvspan(a, b, color="#f2c14e", alpha=0.25, lw=0, label="stimulation (+ after)" if k == 0 else None)


def _zoom_start(t: np.ndarray, good: np.ndarray, span: float) -> float:
    """Start of a zoom window in the middle of the recording where the beats/breaths are good."""
    if t.size == 0:
        return 0.0
    mid = t[t.size // 2]
    ok = t[good]
    if ok.size:
        mid = ok[np.argmin(np.abs(ok - mid))]
    return max(t[0], mid - span / 2)


def bp_figure(path: Path, rid: str, res: dict) -> str:
    bt, segs, cal, epochs = res["bp"], res["bp_segs"], res["calibrated"], res["epochs"]
    unit = "mmHg" if cal else "a.u. (uncalibrated)"
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [1, 1.2]})
    span = 10.0 * 75.0 / max(res["pre"].typical_hr_bpm, 1)
    a = _zoom_start(bt.sys_t, bt.good, span)
    for sg in segs:
        t = sg.times()
        m = (t >= a) & (t <= a + span)
        if m.any():
            ax[0].plot(t[m], sg.x[m], lw=0.8, color="#333333")
    m = (bt.sys_t >= a) & (bt.sys_t <= a + span)
    g = bt.good
    ax[0].plot(bt.sys_t[m & g], bt.sbp[m & g], "v", color="#c0392b", ms=6, label="systolic peak")
    ax[0].plot(bt.onset_t[m & g], bt.dbp[m & g], "^", color="#2471a3", ms=6, label="pulse onset (foot)")
    for i in np.flatnonzero(m & g):
        ax[0].hlines(bt.map[i], bt.onset_t[i], bt.onset_t[i] + bt.ibi_s[i], color="#27ae60", lw=1.5,
                     label="beat mean" if i == np.flatnonzero(m & g)[0] else None)
    bad = m & ~g
    if bad.any():
        ax[0].plot(bt.sys_t[bad], bt.sbp[bad], "x", color="#7f8c8d", ms=7, label="not used (artifact/ectopic)")
    ax[0].set_ylabel(f"pressure ({unit})")
    ax[0].set_xlabel("time (s)")
    ax[0].legend(loc="upper right", fontsize=7, ncol=4)
    ax[0].set_title(f"{rid}: arterial pressure, {span:.0f} s zoom", fontsize=10)
    _shade(ax[1], epochs)
    if cal:
        for v, lab, c in ((bt.sbp, "systolic", "#c0392b"), (bt.map, "mean", "#27ae60"), (bt.dbp, "diastolic", "#2471a3")):
            ax[1].plot(bt.sys_t[g], v[g], ".", ms=1.5, color=c, label=lab)
        if (~g).any():
            ax[1].plot(bt.sys_t[~g], bt.sbp[~g], "x", ms=3, color="#7f8c8d", label="not used")
        ax[1].set_ylabel("pressure (mmHg)")
    else:
        ax[1].plot(bt.sys_t[g], 60.0 / bt.ibi_s[g], ".", ms=1.5, color="#8e44ad", label="pulse rate")
        ax[1].set_ylabel("pulse rate (beats/min)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(loc="upper right", fontsize=7, ncol=4, markerscale=4)
    ax[1].set_title("beat by beat, whole recording", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    n_bad = int((~g).sum())
    if cal:
        return (f"{rid}: top, {span:.0f} s of the arterial pressure with each beat's pulse onset (blue triangle, the "
                "diastolic pressure), systolic peak (red) and beat-mean pressure (green line, the integral mean over the "
                "beat). Check that every pulse has one onset and one peak and that the dicrotic notch is not taken for a "
                f"beat. Bottom, systolic, mean and diastolic pressure beat by beat over the whole recording; {n_bad} "
                "beats marked x were left out (artefact, flat or clipped trace, ectopic interval)"
                + ("; shaded: stimulation epochs" if epochs else "") + ". " + RESEARCH)
    return (f"{rid}: the pressure channel is uncalibrated, so only pulse timing is used. Top, {span:.0f} s of the "
            "trace with each pulse onset (blue) and systolic peak (red): check that each pulse is found once. Bottom, "
            f"pulse rate from the inter-beat interval over the whole recording; {n_bad} pulses were left out. "
            + RESEARCH)


def resp_figure(path: Path, rid: str, res: dict) -> str:
    br, epochs = res["breaths"], res["epochs"]
    has_rr = res["ecg"] is not None and res.get("rsa") is not None
    n = 4 if has_rr else 3
    fig, ax = plt.subplots(n, 1, figsize=(11, 2.2 * n + 1), sharex=False)
    med = float(np.median(br.duration_s[br.good])) if br.good.any() else 4.0
    span = max(30.0, 10 * med)
    a = _zoom_start(br.onset_s, br.good, span)
    for sg in br.segments:
        t = sg.t0 + np.arange(sg.clean.size) / sg.fs
        m = (t >= a) & (t <= a + span)
        if not m.any():
            continue
        ax[0].plot(t[m], sg.clean[m], lw=0.9, color="#333333")
        tr = sg.troughs[(t[sg.troughs] >= a) & (t[sg.troughs] <= a + span)]
        pk = sg.peaks[(t[sg.peaks] >= a) & (t[sg.peaks] <= a + span)]
        ax[0].plot(t[tr], sg.clean[tr], "^", color="#2471a3", ms=6, label="inspiration onset")
        ax[0].plot(t[pk], sg.clean[pk], "v", color="#c0392b", ms=6, label="end of inspiration")
        if has_rr:
            r_t = res["rsa"]["r_t"]
            rm = (r_t[1:] >= a) & (r_t[1:] <= a + span)
            ax[3].plot(r_t[1:][rm], np.diff(r_t)[rm] * 1000.0, "o-", ms=2.5, lw=0.8, color="#8e44ad")
            for x in t[tr]:
                ax[3].axvline(x, color="#2471a3", lw=0.5, alpha=0.6)
            ax[3].set_ylabel("RR (ms)")
            ax[3].set_xlabel("time (s)")
            ax[3].set_title("RR intervals in the same window (blue lines: inspiration onsets): RSA is the rise and "
                            "fall of RR with each breath", fontsize=9)
    h, lab = ax[0].get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax[0].legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=7)
    ax[0].set_ylabel("respiration\n(cleaned, a.u.)")
    ax[0].set_title(f"{rid}: respiration, {span:.0f} s zoom", fontsize=10)
    g = br.good
    for k, (v, yl) in enumerate(((br.rate_bpm, "rate (breaths/min)"), (br.amplitude, "amplitude (a.u.)")), start=1):
        _shade(ax[k], epochs)
        ax[k].plot(br.onset_s[g], v[g], ".", ms=2, color="#16a085")
        if (~g).any():
            ax[k].plot(br.onset_s[~g], v[~g], "x", ms=3, color="#7f8c8d")
        ax[k].set_ylabel(yl)
    ax[2].set_xlabel("time (s)")
    ax[1].set_title("breath by breath, whole recording", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return (f"{rid}: top, {span:.0f} s of the cleaned respiration with each inspiration onset (blue) and end of "
            "inspiration (red): check that every breath is found once and that small ripples are not counted. Middle, "
            "breathing rate and amplitude breath by breath over the whole recording (x: outside the species' plausible "
            "rate)" + ("; shaded: stimulation epochs" if epochs else "") + "."
            + (" Bottom, the RR intervals in the zoom window: they should lengthen and shorten with each breath "
               "(respiratory sinus arrhythmia)." if has_rr else "") + " " + RESEARCH)


def brs_figure(path: Path, rid: str, res: dict, coh_min: float) -> str:
    b, bt, pre = res["brs"], res["bp"], res["pre"]
    seqs, iv, sp = b["seqs"], b["iv"], b["spec"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ok = iv.ok & bt.good
    ax[0].plot(bt.sbp[ok], iv.rr_ms[ok], ".", ms=1.5, color="#bbbbbb", label="all beats")
    for q in seqs:
        s = slice(q.i0, q.i0 + q.n_beats)
        ax[0].plot(bt.sbp[s], iv.rr_ms[s], "-", lw=0.8, color="#c0392b" if q.kind == "up" else "#2471a3", alpha=0.7)
    ax[0].plot([], [], "-", color="#c0392b", label="up sequence")
    ax[0].plot([], [], "-", color="#2471a3", label="down sequence")
    ax[0].set_xlabel("systolic pressure (mmHg)")
    ax[0].set_ylabel(f"{'RR' if iv.source != 'bp-pulse' else 'pulse'} interval, lag {b['lag']} (ms)")
    ax[0].legend(fontsize=7)
    slopes = [q.slope for q in seqs]
    ax[0].set_title(f"{rid}\n{len(seqs)} sequences", fontsize=9)
    if slopes:
        ax[1].hist(slopes, bins=min(30, max(5, len(slopes) // 3)), color="#7f8c8d")
        ax[1].axvline(np.mean(slopes), color="k", lw=1.2, label=f"mean {np.mean(slopes):.1f} ms/mmHg")
        ax[1].legend(fontsize=7)
    ax[1].set_xlabel("sequence slope (ms/mmHg)")
    ax[1].set_ylabel("sequences")
    ax[1].set_title("sequence method", fontsize=9)
    if sp is not None:
        m = sp.f <= max(pre.hf_hz[1] * 1.2, 0.5)
        ax2 = ax[2]
        ax2.semilogy(sp.f[m], sp.p_sbp[m], color="#c0392b", lw=1, label="SBP (mmHg²/Hz)")
        ax2.semilogy(sp.f[m], sp.p_rr[m], color="#8e44ad", lw=1, label="RR (ms²/Hz)")
        ax2.set_xlabel("frequency (Hz)")
        ax2.set_ylabel("power")
        for band, c in ((pre.lf_hz, "#f5b041"), (pre.hf_hz, "#85c1e9")):
            ax2.axvspan(band[0], band[1], color=c, alpha=0.2, lw=0)
        axc = ax2.twinx()
        axc.plot(sp.f[m], sp.coh[m], color="#27ae60", lw=1, label="coherence")
        axc.axhline(coh_min, color="#27ae60", ls=":", lw=1)
        axc.set_ylim(0, 1.02)
        axc.set_ylabel("coherence")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = axc.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
        ax2.set_title(f"spectra, {sp.n_windows} windows of {sp.window_s:g} s (LF orange, HF blue)", fontsize=9)
    else:
        ax[2].text(0.5, 0.5, "not enough clean data\nfor spectra", ha="center", va="center", transform=ax[2].transAxes)
        ax[2].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return (f"{rid}: spontaneous baroreflex. Left, every beat's systolic pressure against the "
            f"{'RR' if iv.source != 'bp-pulse' else 'pulse'} interval {b['lag']} beat(s) later (grey), with the "
            "sequences in which both rose (red) or fell (blue) together for at least three beats: the slope of each is "
            "a baroreflex gain. Middle, the distribution of those slopes and their mean (the sequence BRS). Right, "
            "spectra of systolic pressure and RR with their coherence; the spectral BRS is only valid in a band where "
            f"coherence (green) is above {coh_min:g} (dotted line). Lower values mean a weaker cardiac baroreflex; "
            "values depend on age, posture, breathing and anaesthesia. " + RESEARCH)


def recording_figures(figdir: Path, rid: str, res: dict, coh_min: float = 0.5) -> list[dict]:
    out = []
    if res["bp"] is not None and res["bp"].n:
        name = f"{rid}_bp.png"
        out.append({"path": f"figures/{name}", "caption": bp_figure(figdir / name, rid, res)})
    if res["breaths"] is not None and res["breaths"].n:
        name = f"{rid}_resp.png"
        out.append({"path": f"figures/{name}", "caption": resp_figure(figdir / name, rid, res)})
    if res.get("brs") is not None:
        name = f"{rid}_brs.png"
        out.append({"path": f"figures/{name}", "caption": brs_figure(figdir / name, rid, res, coh_min)})
    return out
