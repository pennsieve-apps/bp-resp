"""
Species presets: the numbers that make pressure-beat detection, breath detection and the baroreflex
work on a given species, each with its source. Every value can be overridden by its own parameter
(see main.read_params). The README shows this table.

Sources (verified DOIs, full references in CITATION.cff):
  ZONG   Zong et al. 2003, Computers in Cardiology 30:259, doi:10.1109/CIC.2003.1291140: slope sum
         function for adult human arterial pressure: 16 Hz low-pass, 128 ms SSF window, 250 ms
         refractory ("eye-closing") period.
  PZ     Behar et al. 2018, PhysioZoo, Front Physiol 9:1390, doi:10.3389/fphys.2018.01390: Table 2
         (typical heart rate, RRmin, RRmax) and Table 3 (LF/HF bands). Numbers only; no PhysioZoo
         code is used (it is GPL-3.0). These are the same numbers ecg-hrv uses.
  TF     Task Force 1996, Circulation 93:1043, doi:10.1161/01.CIR.93.5.1043 (human LF/HF bands).
  CARN   Carnevali et al. 2023 (rat LF/HF bands, as ecg-hrv), POL/SAL Poletto et al. 2011 and
         Salomao et al. 2015 (pig bands, as ecg-hrv): taken over from ecg-hrv's preset table so the
         two apps use the same bands.
  BAG    Bagnall-Hare et al. 2024, Phys Eng Sci Med 47:503, doi:10.1007/s13246-023-01380-y, a review
         of sequence-method settings: "Bertinieri et al [32] further refined the sequence selection
         criteria to include a minimum changes of at least >= 1 mmHg in blood pressure, >= 4 ms in
         RR-interval, and a delay of one beat". The thresholds of Bertinieri et al. 1988 and Parati
         et al. 1988 themselves could not be read (no open full text); BAG is the verified source.
  MOL    Molcan et al. 2026, Hypertens Res 49:1349, doi:10.1038/s41440-026-02579-8 (rat):
         ">= 1 mmHg for systolic BP and >= 5 ms for IBI".
  PAN    Pan et al. 2023, Front Physiol 14:1185744, doi:10.3389/fphys.2023.1185744 (mouse): SBP and
         R-R "with three beats delay", "threshold for changes in SBP was set at 0.5 mmHg and ... R-R
         interval ... 5 ms", following Laude et al. 2009 (doi:10.1109/MEMB.2009.934630: "A delay of
         0 or 3 beats between BP and PI should be applied").
  KHO    NeuroKit2's khodadad2018 cleaning band (0.05-3 Hz, "preserves breathing rates ... slower
         than 180 breath per minute"), Khodadad et al. 2018, doi:10.1088/1361-6579/aad7e6.
  PENN   Pennsieve choice where no published preset exists, stated as such: other species' pressure
         detector timing scaled from ZONG by the typical heart rate (75 bpm for ZONG's adults),
         plausibility ranges for pressures and breathing rates, breath band for fast breathers,
         series rate, gap and window lengths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SPECIES = ("human", "pig", "dog", "rabbit", "rat", "mouse")

SPECIES_ALIASES = {
    "homo sapiens": "human", "human": "human", "man": "human", "woman": "human",
    "sus scrofa": "pig", "sus scrofa domesticus": "pig", "sus domesticus": "pig", "pig": "pig",
    "swine": "pig", "minipig": "pig", "yucatan minipig": "pig", "porcine": "pig",
    "canis familiaris": "dog", "canis lupus familiaris": "dog", "dog": "dog", "canine": "dog",
    "oryctolagus cuniculus": "rabbit", "rabbit": "rabbit",
    "rattus norvegicus": "rat", "rattus": "rat", "rat": "rat",
    "mus musculus": "mouse", "mouse": "mouse", "mice": "mouse",
}


@dataclass(frozen=True)
class Preset:
    name: str
    typical_hr_bpm: float
    rr_min_s: float                # plausible inter-beat interval range (ECG and pressure)
    rr_max_s: float
    # ECG fallback detector (when ecg-hrv's beats are not linked): as ecg-hrv
    ecg_band_hz: tuple
    ecg_refractory_s: float
    ecg_smooth_s: float
    ecg_avg_s: float
    # arterial pressure beats
    bp_hz: float                   # working rate (pressure is decimated to at most this)
    bp_lowpass_hz: float
    ssf_window_s: float
    bp_refractory_s: float
    sbp_range: tuple               # plausible systolic pressure, mmHg (calibrated only)
    dbp_range: tuple
    pp_min_mmhg: float
    # respiration
    resp_hz: float
    resp_band_hz: tuple
    resp_rate_range: tuple         # breaths/min
    # baroreflex
    seq_sbp_mmhg: float
    seq_rr_ms: float
    seq_lag: str                   # "auto" or a lag in beats
    seq_lags: tuple                # lags tried when seq_lag is auto
    lf_hz: tuple
    hf_hz: tuple
    spectral_hz: float
    spectral_window_s: float
    # series and stimulation windows
    series_hz: float
    max_gap_s: float
    after_s: float
    sources: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


def _bp_timing(hr: float) -> tuple[float, float, float]:
    """ZONG's 16 Hz / 128 ms / 250 ms for ~75 bpm adults, scaled to the species' heart rate."""
    k = 75.0 / hr
    return round(16.0 / k, 1), round(0.128 * k, 4), round(0.25 * k, 4)


def _nk(hr: float, qrs: float) -> tuple[float, float]:
    return round(0.1 * qrs / 0.07, 4), round(0.75 * 75.0 / hr, 4)


_DET = "detector and RR range: PZ Table 2 / ecg-hrv preset"
_BP_SCALED = "pressure detector: ZONG scaled by 75 bpm / typical HR (PENN)"
_RANGES = "plausible SBP/DBP/PP and breathing-rate ranges: PENN"
_COMMON = "series_hz per conventions.md; max_gap_s, after_s, spectral window: PENN"

SEQ_SOURCE = {
    "human": "BAG (Bertinieri's 1 mmHg / 4 ms); lags 0-2 beats per the brief",
    "pig": "PENN (human thresholds: pig heart rates and pressures are close to human)",
    "dog": "PENN (human thresholds)",
    "rabbit": "PENN (0.5 mmHg / 2 ms, between human and rodent practice)",
    "rat": "MOL (1 mmHg / 5 ms); lags 0-2: PENN",
    "mouse": "PAN (0.5 mmHg / 5 ms); lags 0-3 (Laude et al. 2009: 0 or 3 beats)",
}

PRESETS: dict[str, Preset] = {
    "human": Preset(
        "human", 75, 0.28, 2.4, (3.0, 100.0), 0.25, 0.1, 0.75,
        250, 16.0, 0.128, 0.25, (50, 260), (20, 180), 10,
        50, (0.05, 3.0), (4, 60),
        1.0, 4.0, "auto", (0, 1, 2), (0.04, 0.15), (0.15, 0.4), 4, 120,
        4, 3.0, 30,
        {"bp": "ZONG", "detector": _DET, "resp": "KHO; " + _RANGES, "brs": SEQ_SOURCE["human"],
         "bands": "TF", "other": _COMMON}),
    "pig": Preset(
        "pig", 100, 0.25, 2.0, (3.0, 100.0), 0.2, *_nk(100, 0.06),
        250, *_bp_timing(100), (40, 250), (15, 180), 8,
        50, (0.05, 3.0), (5, 60),
        1.0, 4.0, "auto", (0, 1, 2), (0.02, 0.09), (0.09, 2.0), 4, 120,
        4, 3.0, 10,
        {"bp": _BP_SCALED, "detector": "PENN (close to PZ dog), as ecg-hrv", "resp": "KHO; " + _RANGES,
         "brs": SEQ_SOURCE["pig"], "bands": "POL/SAL (as ecg-hrv)", "other": _COMMON}),
    "dog": Preset(
        "dog", 109.5, 0.25, 1.2, (3.0, 125.0), 0.17, *_nk(109.5, 0.04),
        250, *_bp_timing(109.5), (50, 260), (20, 180), 10,
        50, (0.05, 3.0), (5, 60),
        1.0, 4.0, "auto", (0, 1, 2), (0.067, 0.235), (0.235, 0.877), 4, 120,
        4, 2.0, 10,
        {"bp": _BP_SCALED, "detector": _DET, "resp": "KHO; " + _RANGES, "brs": SEQ_SOURCE["dog"],
         "bands": "PZ Table 3", "other": _COMMON}),
    "rabbit": Preset(
        "rabbit", 264, 0.14, 0.58, (3.0, 150.0), 0.088, *_nk(264, 0.02),
        500, *_bp_timing(264), (40, 200), (15, 150), 5,
        50, (0.05, 3.0), (10, 150),
        0.5, 2.0, "auto", (0, 1, 2), (0.088, 0.341), (0.341, 1.155), 10, 60,
        10, 1.0, 10,
        {"bp": _BP_SCALED, "detector": _DET, "resp": "KHO; " + _RANGES, "brs": SEQ_SOURCE["rabbit"],
         "bands": "PZ Table 3", "other": _COMMON}),
    "rat": Preset(
        "rat", 350, 0.10, 1.0, (3.0, 200.0), 0.06, *_nk(350, 0.015),
        1000, *_bp_timing(350), (40, 250), (15, 200), 5,
        100, (0.1, 5.0), (20, 250),
        1.0, 5.0, "auto", (0, 1, 2), (0.2, 0.75), (0.75, 2.5), 10, 60,
        10, 1.0, 10,
        {"bp": _BP_SCALED, "detector": "PENN (between PZ rabbit and mouse), as ecg-hrv",
         "resp": "PENN band (KHO's 3 Hz upper edge is too low for rat breathing); " + _RANGES,
         "brs": SEQ_SOURCE["rat"], "bands": "CARN (as ecg-hrv)", "other": _COMMON}),
    "mouse": Preset(
        "mouse", 608, 0.05, 0.24, (3.0, 300.0), 0.030, *_nk(608, 0.00718),
        1000, *_bp_timing(608), (40, 220), (15, 180), 5,
        100, (0.2, 8.0), (40, 450),
        0.5, 5.0, "auto", (0, 1, 2, 3), (0.152, 1.24), (1.24, 5.0), 20, 30,
        10, 0.5, 10,
        {"bp": _BP_SCALED, "detector": _DET,
         "resp": "PENN band (mouse breathing reaches 5-7 Hz); " + _RANGES, "brs": SEQ_SOURCE["mouse"],
         "bands": "PZ Table 3 (from Thireau et al. 2008)", "other": _COMMON}),
}


def resolve(name: str | None) -> str | None:
    """Preset name for a species string (preset name, Latin binomial or synonym), else None."""
    if not name:
        return None
    s = " ".join(str(name).strip().lower().replace("_", " ").split())
    if s in PRESETS:
        return s
    if s in SPECIES_ALIASES:
        return SPECIES_ALIASES[s]
    for k, v in SPECIES_ALIASES.items():
        if s.startswith(k):
            return v
    return None
