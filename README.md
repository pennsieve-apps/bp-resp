# bp-resp

**Blood Pressure, Respiration and Baroreflex** measures arterial blood pressure beat by beat, finds
every breath, and estimates the spontaneous baroreflex sensitivity and respiratory sinus arrhythmia
of physiological recordings from humans and animals. Breathing and respiratory sinus arrhythmia use
**NeuroKit2**, developed by Dominique Makowski and colleagues
([github.com/neuropsychology/NeuroKit](https://github.com/neuropsychology/NeuroKit),
[Behavior Research Methods 2021](https://doi.org/10.3758/s13428-020-01516-y)): its `khodadad2018`
breath detection (after Davood Khodadad and colleagues,
[Physiological Measurement 2018](https://doi.org/10.1088/1361-6579/aad7e6)) and its `hrv_rsa`
peak-to-trough and Porges-Bohrer estimates (as compared by Gregory F. Lewis, Senta A. Furman,
Martha F. McCool and Stephen W. Porges, [Biological Psychology 2012](https://doi.org/10.1016/j.biopsycho.2011.11.009)).
Pressure pulses are found with the slope sum function of W. Zong, T. Heldt, G. B. Moody and
R. G. Mark ([Computers in Cardiology 2003](https://doi.org/10.1109/CIC.2003.1291140)), the method
behind PhysioNet's `wabp`. The baroreflex follows the sequence method introduced by G. Bertinieri, M. Di Rienzo, G. Mancia and
colleagues (J Hypertens Suppl 1985, [PubMed 2856787](https://pubmed.ncbi.nlm.nih.gov/2856787/); validated
in cats in [Am J Physiol 1988](https://doi.org/10.1152/ajpheart.1988.254.2.H377)) and applied in humans
by G. Parati and colleagues ([Hypertension 1988](https://doi.org/10.1161/01.HYP.12.2.214)), and the
spectral alpha index and transfer-function gain, two of the techniques compared in the EuroBaVar study
by Dominique Laude and colleagues ([Am J Physiol Regul Integr Comp Physiol 2004](https://doi.org/10.1152/ajpregu.00709.2002)).
**Pennsieve packages these methods and did not develop them**: NeuroKit2 is used as published; the
pulse detector and the baroreflex have no maintained, openly licensed implementation, so they are
written here from the papers (Apache-2.0) and credited to their authors. The app reads the NWB files
written by physio-import (NWB: Oliver Rübel and colleagues, [eLife 2022](https://doi.org/10.7554/eLife.78362)).
It was tested on PhysioNet's Autonomic Aging database (Andy Schumann and Karl-Jürgen Bär,
[doi:10.13026/2hsy-t491](https://doi.org/10.13026/2hsy-t491)) and Fantasia database (Nikhil Iyengar
and colleagues, [doi:10.13026/C2RG61](https://doi.org/10.13026/C2RG61)), and on Pennsieve Discover
datasets 349 (Oliver Armitage and colleagues, [doi:10.26275/dm5l-tspb](https://doi.org/10.26275/dm5l-tspb))
and 655 (Shane A. Bender and colleagues, [doi:10.26275/5cwo-mkpz](https://doi.org/10.26275/5cwo-mkpz)).

**For a physiologist:** give it your recordings (after physio-import). For every recording with an
arterial or finger blood pressure trace it finds each pulse and gives systolic, diastolic, mean
(the true average over the beat, not (2D+S)/3) and pulse pressure, beat by beat. For every
respiration trace (belt, airflow, thermistor, plethysmograph) it finds each breath and gives its rate,
depth and inspiration/expiration times. With calibrated pressure and an ECG (or just the pressure) it
estimates how strongly heart period follows spontaneous pressure changes (baroreflex sensitivity,
ms/mmHg) in two independent ways, and with respiration and an ECG how much heart period rises and
falls with breathing (RSA). Animal recordings use species presets (rat, mouse, pig, dog, rabbit):
pulses at 400 beats/min and breaths at 100/min are found with the right timing. When the pressure is
not calibrated (no mmHg), you get pulse timing only and the tables say why; a recording without
pressure or respiration is listed as skipped, not failed. Figures show what was detected so you can
check it. Every number is a research output, not a clinical finding.

**For a data scientist:** input is NWB (one file per recording, `acquisition/<channel>` TimeSeries,
read with h5py by slice and decimated to the species' working rate), typed by physio-import's
`channels.csv` (`bp`, `resp`, `ecg`; `calibrated`). Outputs are long-format CSVs keyed by
`recording_id`, a regularly gridded `series.csv` (`recording_id, time_s, variable, value, valid`) that
stim-dose-response consumes, `features.csv`, PNG figures with captions in `summary.json`, and
`provenance.json` with every parameter and the preset used.

## Ports

| Port | Direction | Contents |
|---|---|---|
| `physio` | input, required | physio-import's `physio` port: `recordings.csv`, `channels.csv`, `sub-*/<recording_id>.nwb`; or any NWB files (channels then typed from their names: `bp`/ABP/ART/NIBP, `resp`, `ecg`). physio-import's own `events.tsv` (raw markers) is never read. |
| `events` | input, optional | stim-events' `events` port: `stim-events/events.tsv`. Stimulation and block epochs (plus `AFTER_S`) are left out of the resting baroreflex and RSA and shaded in the figures; beat and breath tables and `series.csv` keep them (stim-dose-response needs them). |
| `hrv` | input, optional | ecg-hrv's `hrv` port: `ecg-hrv/beats.csv`. Its R-peaks and labels give the RR intervals. Without it, R-peaks are detected here (NeuroKit2 `neurokit` detector with the preset; mains notch when needed) and `qc.csv` says so. |
| `bp` | output | `bp-resp/` (below) |

All files are found by name and content, recursively, also when the ports are merged side by side.

## Methods

**Pressure beats.** The pressure is low-pass filtered (zero-phase 2nd-order Butterworth; 16 Hz for
humans as in Zong et al. 2003) and turned into the slope sum function (SSF: the sum of the positive
first differences over the last 128 ms, Zong's window). An SSF pulse counts when it reaches
`SSF_THRESHOLD` (0.5) of the running median of the last eight accepted pulses (initialised from the
first 10 s, re-learnt after a pause), at least `BP_REFRACTORY_S` after the previous one. The onset is
where the SSF, going back from its peak, falls below `SSF_ONSET` (2 %) of the peak; the foot is the
pressure minimum just before it. Each beat runs from one foot to the next: DBP = pressure at the foot,
SBP = the maximum, MAP = integral mean over the beat, PP = SBP − DBP; the beat's `time_s` is its
systolic peak. For other species the filter, window and refractory period scale with the typical
heart rate (preset table). Differences from Zong's paper: a zero-phase filter instead of the causal
recursive one (no 20 ms delay to correct), a running-median threshold instead of the running-maximum
base, and a 2 % instead of 1 % onset fraction (less sensitive to noise before the upstroke). Beat
labels: `artifact` when the trace is flat or clipped for ≥ 100 ms within the beat (a disconnected
line, a calibration square wave, transducer saturation), when the inter-beat interval is outside the
species range, when (calibrated) SBP, DBP or PP are implausible, or when SBP or DBP jump from the
median of the 10 neighbouring beats by more than `BP_JUMP` × their pulse pressure; `ectopic` when the
interval differs from its neighbours' median by more than `ECTOPIC_FRAC`. Only `N` beats are used.

**Breaths.** NeuroKit2's `rsp_process` steps with its default method `khodadad2018`, called one by
one: `rsp_clean` (2nd-order Butterworth 0.05–3 Hz; for rats and mice the same filter with the
preset's wider band, because 3 Hz cuts off fast rodent breathing), `rsp_peaks` (zero crossings, the
extreme between them, extrema closer than `RESP_AMPLITUDE_MIN` × the median distance dropped). One
breath = one trough-to-trough cycle (trough = inspiration onset): duration, rate = 60/duration,
amplitude = peak − preceding trough (as `rsp_amplitude`), inspiration and expiration times. RVT and
symmetry features are not computed. Breaths outside the species' rate range are labelled `artifact`.

**Baroreflex sensitivity** (calibrated pressure only). Each pressure beat's SBP is paired with the RR
interval in which its systolic peak falls (lag 0) or the one `SEQ_LAG` beats later; without R-peaks the
pulse interval (systolic peak to systolic peak) is used. *Sequence method*: runs of ≥ `SEQ_MIN_BEATS`
(3) consecutive beats in which SBP rises by ≥ `SEQ_SBP_MMHG` and RR lengthens by ≥ `SEQ_RR_MS` at
every beat (up) or both fall (down), with RR-on-SBP correlation r ≥ `SEQ_MIN_R` (0.8); BRS = mean
slope (up, down, all); valid with ≥ `SEQ_MIN_N` (3) sequences. With `SEQ_LAG=auto` the lag (0, 1, 2;
mouse 0–3) with the most sequences is used and `qc.csv` lists the counts. *Spectral methods*: SBP and
RR (lag 0) are interpolated onto a 4 Hz grid (rodents 10–20 Hz); Welch averages (Hann,
`SPECTRAL_WINDOW_S`, 50 % overlap, linear detrend) over the windows free of gaps, artefacts and
stimulation; at least 3 windows are needed (coherence from one window is meaningless).
`alpha-lf`/`alpha-hf` = √(P_RR / P_SBP) over the band, valid when the mean squared coherence over the
band is ≥ `COHERENCE_MIN` (0.5); `transfer-lf` = mean |P_SBP,RR| / P_SBP over the LF frequencies whose
coherence is ≥ 0.5. The 0.5 coherence threshold is the usual convention of spectral BRS studies; it
is not taken from a source verified for this app (the EuroBaVar abstract does not state it). Bands are
the species' HRV bands, the same as ecg-hrv's.

**Respiratory sinus arrhythmia** (respiration and R-peaks). Ectopic or artefact beats are replaced by
evenly spaced beats between their good neighbours (as HRV software interpolates them). The two
estimates of NeuroKit2's `hrv_rsa`, computed with NeuroKit2's rules at the **exact R-peak times**:
`hrv_rsa` takes R-peaks as sample indices of the respiration signal, and rounding them to the
respiration's 50–100 Hz working rate moves each RR interval by up to 10–20 ms (on Fantasia it raised
P2T by 2–9 %; in a simulated rat with no RSA it gave 5.5 ms instead of about 1 ms). A test checks that
the exact-time code gives NeuroKit2's values when NeuroKit2 runs on a 1 kHz grid. `p2t` (peak-to-trough:
per breath, the longest expiratory RR minus the shortest inspiratory RR, negative values set to 0,
averaged; ms; Grossman's peak-to-valley method as compared by Lewis et al. 2012) and `porges-bohrer`
(heart period at 2 Hz, 21-point cubic detrend, 0.12–0.40 Hz band, log variance per 30 s epoch,
averaged; ln(ms²); Lewis et al. 2012). Each phase's R-peaks run up to a vagal delay after the phase
ends: NeuroKit2's 750 ms, which its code attributes to Grossman et al. 1990
([Psychophysiology](https://doi.org/10.1111/j.1469-8986.1990.tb03198.x)); for other species the
delay is shortened to a quarter of the median breath when that is shorter, because 750 ms is longer
than a whole rat breath, and Porges-Bohrer is not reported (its filter is defined for human
breathing). RSA is computed over stretches of ≥ 60 s (or 10 breaths) free of stimulation and gaps,
weighted by breaths.

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `SPECIES` | `auto` | Preset: `auto` (recordings.csv / NWB Subject, else human), `human`, `pig`, `dog`, `rabbit`, `rat`, `mouse` |
| `SERIES_HZ` | `auto` | Grid of series.csv (preset: 4 Hz humans and large animals, 10 Hz rodents) |
| `MAX_GAP_S` | `auto` | series.csv `valid=false` where usable beats are further apart (breaths: at least 3 median breaths) |
| `BP_LOWPASS_HZ`, `SSF_WINDOW_S`, `BP_REFRACTORY_S` | `auto` | Pulse detector timing (preset) |
| `SSF_THRESHOLD` | `0.5` | Pulse threshold as a fraction of the recent pulses' SSF peaks |
| `SSF_ONSET` | `0.02` | Onset where the SSF falls below this fraction of its peak |
| `BP_JUMP` | `0.5` | Artifact when SBP/DBP jump by more than this × local pulse pressure |
| `ECTOPIC_FRAC` | `0.3` | Ectopic when the inter-beat interval differs from the local median by more than this fraction |
| `RESP_BAND` | `auto` | Breath band-pass (Hz, `low-high`) |
| `RESP_AMPLITUDE_MIN` | `0.3` | Smallest breath, fraction of the median peak-trough distance (NeuroKit2 `amplitude_min`) |
| `RESP_POLARITY` | `normal` | `inverted` when inspiration goes down on the trace (qc.csv flags likely inversions) |
| `RR_SOURCE` | `auto` | `auto` (ecg-hrv beats, else ECG here, else pressure pulse interval), `hrv`, `ecg`, `bp` |
| `SEQ_SBP_MMHG`, `SEQ_RR_MS` | `auto` | Sequence thresholds (preset) |
| `SEQ_LAG` | `auto` | `auto` or 0–3 beats |
| `SEQ_MIN_BEATS` | `3` | Shortest sequence |
| `SEQ_MIN_R` | `0.8` | Smallest correlation per sequence |
| `SEQ_MIN_N` | `3` | Fewest sequences for a valid sequence BRS |
| `LF_BAND`, `HF_BAND` | `auto` | Spectral bands (Hz, `low-high`) |
| `COHERENCE_MIN` | `0.5` | Coherence needed for a valid spectral BRS |
| `SPECTRAL_WINDOW_S` | `auto` | Welch window (preset) |
| `EXCLUDE_STIM` | `true` | Leave stimulation epochs (events port) out of resting BRS and RSA |
| `AFTER_S` | `auto` | Also leave out this long after each epoch (preset) |
| `FIGURES` | `true` | Figures per recording |
| `THREADS` | `0` | BLAS threads; 0 = from the cgroup CPU quota |

### Species presets

Every value can be overridden by its parameter; the preset used is written to `provenance.json`.

| | human | pig | dog | rabbit | rat | mouse |
|---|---|---|---|---|---|---|
| typical HR (bpm) | 75 | 100 | 109.5 | 264 | 350 | 608 |
| RR / inter-beat range (s) | 0.28–2.4 | 0.25–2.0 | 0.25–1.2 | 0.14–0.58 | 0.10–1.0 | 0.05–0.24 |
| pressure low-pass (Hz) / SSF window (s) / refractory (s) | 16 / 0.128 / 0.25 | 21.3 / 0.096 / 0.1875 | 23.4 / 0.0877 / 0.1712 | 56.3 / 0.0364 / 0.071 | 74.7 / 0.0274 / 0.0536 | 129.7 / 0.0158 / 0.0308 |
| plausible SBP / DBP (mmHg), min PP | 50–260 / 20–180, 10 | 40–250 / 15–180, 8 | 50–260 / 20–180, 10 | 40–200 / 15–150, 5 | 40–250 / 15–200, 5 | 40–220 / 15–180, 5 |
| breath band (Hz) / rate range (/min) | 0.05–3 / 4–60 | 0.05–3 / 5–60 | 0.05–3 / 5–60 | 0.05–3 / 10–150 | 0.1–5 / 20–250 | 0.2–8 / 40–450 |
| sequence SBP / RR step, lags | 1 mmHg / 4 ms, 0–2 | 1 / 4, 0–2 | 1 / 4, 0–2 | 0.5 / 2, 0–2 | 1 / 5, 0–2 | 0.5 / 5, 0–3 |
| LF / HF band (Hz) | 0.04–0.15 / 0.15–0.4 | 0.02–0.09 / 0.09–2.0 | 0.067–0.235 / 0.235–0.877 | 0.088–0.341 / 0.341–1.155 | 0.2–0.75 / 0.75–2.5 | 0.152–1.24 / 1.24–5.0 |
| spectral grid (Hz) / window (s) | 4 / 120 | 4 / 120 | 4 / 120 | 10 / 60 | 10 / 60 | 20 / 30 |
| series Hz / max gap (s) / after stim (s) | 4 / 3 / 30 | 4 / 3 / 10 | 4 / 2 / 10 | 10 / 1 / 10 | 10 / 1 / 10 | 10 / 0.5 / 10 |

Sources: pressure timing for humans from Zong et al. 2003 (16 Hz, 128 ms; 250 ms refractory, the
value of PhysioNet's `wabp`; the paper uses 300 ms), scaled by 75 bpm / typical HR for the other
species (a Pennsieve choice). Heart-rate and RR ranges and LF/HF bands are ecg-hrv's preset table:
PhysioZoo (Behar et al. 2018, Tables 2 and 3; numbers only, its code is GPL), Task Force 1996 (human
bands), Behar et al. 2018b (mouse bands, via PhysioZoo; HF upper edge 5 Hz after Thireau et al. 2008), Carnevali et al. 2023 (rat bands), Poletto
et al. 2011 and Salomão et al. 2015 (pig bands); rat and pig detector ranges are Pennsieve choices.
Human sequence thresholds: 1 mmHg and 4 ms. Their only verified source for this app is the review by
Bagnall-Hare et al. 2024, which attributes them to Bertinieri et al. 1988 ("a minimum changes of at
least ≥ 1 mmHg in blood pressure, ≥ 4 ms in RR-interval, and a delay of one beat is assumed"); the
1985 and 1988 papers themselves could not be read (no open full text). Rat: 1 mmHg and 5 ms
as used by Molcan et al. 2026. Mouse: 0.5 mmHg and 5 ms as Pan et al. 2023 used (with a fixed 3-beat delay), and lags 0-3 tried
following Laude et al. 2009 ("A delay of 0 or 3 beats between BP and PI should be applied").
Pig, dog: human thresholds; rabbit 0.5 mmHg / 2 ms (Pennsieve choices: no published preset found).
Breath band: NeuroKit2's `khodadad2018` 0.05–3 Hz; rodent bands, plausible pressure and breathing
ranges, spectral windows, series rate, gaps and after-stimulation windows are Pennsieve choices.

## Outputs (`bp` port, under `bp-resp/`)

| File | Columns |
|---|---|
| `bp_beats.csv` | recording_id, beat, time_s (systolic peak), sbp_mmhg, dbp_mmhg, map_mmhg, pp_mmhg (empty when uncalibrated), calibrated, onset_s (foot), ibi_ms, label (`N` / `ectopic` / `artifact`) |
| `breaths.csv` | recording_id, breath, onset_s, duration_s, rate_bpm, amplitude (channel unit), inspiration_s, expiration_s, label |
| `series.csv` | recording_id, time_s, variable (`sbp_mmhg`, `dbp_mmhg`, `map_mmhg`, `pp_mmhg` for calibrated pressure; `resp_rate_bpm`, `resp_amp`), value, valid |
| `brs.csv` | recording_id, method (`sequence-up`, `sequence-down`, `sequence-all`, `alpha-lf`, `alpha-hf`, `transfer-lf`), value_ms_per_mmhg, n (sequences / Welch windows / coherent frequencies), coherence, valid, lag_beats, rr_source (`ecg-hrv` / `detected` / `bp-pulse`), note (why a value is empty or invalid) |
| `rsa.csv` | recording_id, method (`p2t` / `porges-bohrer`), value, unit (`ms` / `ln(ms^2)`), n_breaths, note |
| `qc.csv` | recording_id, check, value, note: beat and breath counts, why beats were labelled, channel status, inspiration/expiration ratio, RR source, sequence lag, stimulation left out, calibration doubts |
| `features.csv` | recording_id, channel, feature, value: `bp_{sbp,dbp,map,pp}_{mean,sd}_mmhg`, `bp_pulse_rate_bpm`, `bp_n_beats`, `bp_pct_artifact`, `resp_rate_median_bpm`, `resp_rate_sd_bpm`, `resp_amplitude_median`, `resp_n_breaths`, `brs_sequence_{up,down,all}`, `brs_alpha_{lf,hf}`, `brs_transfer_lf` (valid ones only), `rsa_p2t_ms`, `rsa_porges_bohrer` |
| `figures/<recording_id>_bp.png` | 10-beat zoom with onsets, peaks and beat means; beat-by-beat pressures (or pulse rate when uncalibrated) |
| `figures/<recording_id>_resp.png` | zoom with inspiration onsets and ends; rate and amplitude per breath; RR intervals in the zoom window |
| `figures/<recording_id>_brs.png` | SBP vs RR with the sequences; slope histogram; SBP and RR spectra with coherence |
| `summary.json`, `provenance.json` | as conventions.md; `summary.json` has a `notice` when any pressure is uncalibrated |

Missing signals: a recording without pressure and respiration is `skipped` in `provenance.json`
(nothing in the tables); without pressure, `brs.csv` has the six methods empty with the reason;
without respiration or R-peaks, `rsa.csv` likewise. Uncalibrated pressure (`calibrated=false`, or a
unit other than mmHg): beats with timing and `ibi_ms`, no pressures, no pressure series, no BRS.

## Runtime

`cpu: 1024`, `memory: 2048` (MiB), `timeoutSeconds: 900`; Lambda-eligible. Measured in Docker with
`--cpus 1 --memory 2g --read-only --user 65534:65534` (the Lambda contract), amd64 emulation on an
M-series Mac, peak memory from the container's cgroup `memory.peak`:

| Demo | Recordings | Wall time | Peak memory |
|---|---|---|---|
| Fantasia (6 × ~2 h, 250 Hz; ECG, respiration, BP in 4) | 6 analysed | 9.2 s | 510 MB |
| Autonomic Aging (3 × 15–32 min, 1000 Hz; ECG, NIBP) | 3 analysed | 2.9 s | 242 MB |
| Discover 349 / 655 / 406 (no pressure or respiration) | 1 / 3 / 20 skipped | < 0.1 s | 111–113 MB |

Image: 159 MB (`docker image inspect`). One recording at a time. Pressure and ECG are read, decimated
and analysed 10 minutes (plus 10 s on each side) at a time, so their memory does not grow with the
recording's length; only the beat and breath lists and the respiration at its working rate (50–100
Hz) are kept for the whole recording. Every table is written while it is produced, never collected for
the run. Long recordings, measured the same way on a synthetic rat (`validation/make_long_recording.py`:
ECG, arterial pressure and respiration at 1 kHz):

| Synthetic rat | Beats / breaths | Wall time | Peak memory |
|---|---|---|---|
| 6 h, 3 channels at 1 kHz (65 million samples) | 127,008 / 32,397 | 45 s | 586 MB |

(Before this design a 6 h rat was killed at 2048 MiB.) Lambda's 15-minute limit allows a few days of
rodent recording at this rate; the 10 GB `/tmp` limit (inputs + outputs) comes first for very long
1 kHz recordings.

## Validation

All numbers below are in `validation/results.csv`, made by `validation/validate.py` from the app's
outputs (Docker image, the outputs in `workflow-physiology/data/work/<demo>/bp-resp/`).

**Pulse detection on uncalibrated pressure (Fantasia f2 records), against PhysioNet's reference beat
annotations:** every reference normal QRS should be followed by exactly one detected systolic peak
60–500 ms later.

| Record | Sensitivity | Positive predictive value | Median R-to-systolic-peak |
|---|---|---|---|
| f2o01 (73 y) | 99.47 % (7071/7109) | 99.16 % | 0.380 s |
| f2o02 (75 y) | 99.32 % (6308/6351) | 99.67 % | 0.348 s |
| f2y01 (23 y) | 99.52 % (8003/8042) | 98.30 % | 0.284 s |
| f2y02 (23 y) | 99.22 % (6459/6510) | 98.78 % | 0.334 s |

**Baroreflex on calibrated pressure (Autonomic Aging), R-peaks from ecg-hrv**, next to the
age-expected value from Kardos et al. 2001 (lnBRS = 3.24 − 0.03 × age, sequence method, Finapres, 1134
healthy adults 18–60 y; the regression in the abstract) at the middle of the age group, and the mean
sequence slopes Parati et al. 1988 reported for 24 h intra-arterial recordings in humans (7.6 ± 2.0
up, 6.4 ± 1.5 down ms/mmHg):

| Record (age group, sex) | sequence all (n) | alpha-LF (coh.) | alpha-HF (coh.) | transfer-LF (n bins) | Kardos 2001 expected |
|---|---|---|---|---|---|
| 0001 (20–24 y, F) | 34.5 (12, lag 2) | 24.3 (0.43, invalid) | 35.4 (0.13, invalid) | 27.2 (6) | 13.2 |
| 0002 (20–24 y, F) | 35.3 (56, lag 0) | 24.7 (0.56) | 57.0 (0.50) | 34.5 (7) | 13.2 |
| 0003 (45–49 y, F) | 5.5 (1: invalid) | 5.1 (0.45, invalid) | 9.8 (0.55) | 4.1 (7) | 6.2 |

The 45–49-year-old's BRS is a fifth of the two 20–24-year-olds', as the age dependence predicts, and
sequence and spectral estimates agree within each subject. The two young women are well above the
regression line (r² = 0.23 in Kardos et al., so individuals scatter widely around it). The NIBP of this database is a 100 Hz vascular-unloading signal
spline-resampled to 1000 Hz (Schumann & Bär 2022) and very smooth beat to beat: subject 0003's SBP
changes by a median of 0.4 mmHg per beat, so the 1 mmHg / 4 ms thresholds find only one sequence.
With `SEQ_SBP_MMHG=0.5 SEQ_RR_MS=2` (Laude et al. 2009 found "No thresholds in BP or PI were
necessary" in mice) the sequence BRS is 49.5 (47 sequences), 40.0 (147) and 7.4 (26) ms/mmHg.
Subject 0001's pressure has level steps of 20–30 mmHg (device re-calibrations), which is why its
coherence is low. Subject 0002's two calibration square waves (25 ↔ 175 mmHg) are labelled artifact
(43 beats, `flat or clipped waveform`).

**Synthetic recordings with planted values** (`tests/`): baroreflex gains of 12 ms/mmHg (human) and
3 ms/mmHg (rat) are recovered by every method within 2 % (sequence), 1 % (alpha) and 10–15 %
(transfer-LF); SBP and DBP means within 0.1 mmHg; breathing rates of 12, 15 and 90 /min exactly; a
planted ±30 ms respiratory swing gives a peak-to-trough RSA of 61 ms.

**RSA (Fantasia), with ecg-hrv's R-peaks** (and with R-peaks detected inside bp-resp, no hrv port):
p2t 114.6 (103.8) ms in f1y01 (23 y), 60.6 (62.4) f2y01 (23 y), 79.9 (77.4) f2y02 (23 y), 27.4 (26.2)
f1o01 (77 y), 50.6 (49.9) f2o01 (73 y), 36.8 (36.6) f2o02 (75 y); Porges-Bohrer 7.69, 6.19, 7.49 vs
4.79, 6.17, 5.20 ln(ms²): higher in the young than the elderly, and the two R-peak sources agree
within 10 %. Breathing rates 12.6–20.6 /min (median per record).

**Missing signals:** Discover 349 (pig) and 655 (rat) have ECG, heart rate and stimulation channels
but no pressure or respiration, and Discover 406 has ECG only: every recording is listed as skipped
with "no blood pressure or respiration channel", the tables have their headers only, and the step
exits 0.

## Limitations

- **Research outputs.** Nothing here is a clinical measurement or diagnosis.
- Spontaneous BRS describes the cardiac (vagal) limb of the baroreflex around the operating point;
  it is not the full sigmoid of a pharmacological (Oxford) test and says nothing about the
  sympathetic vascular limb. Values depend strongly on age, posture, breathing pattern, recording
  length and method; sequence and spectral estimates are not interchangeable (EuroBaVar).
- Finger or vascular-unloading pressure (Finapres, CNAP, Task Force Monitor) is reconstructed and
  periodically re-calibrated; level steps and smoothing change the sequence counts.
- Anaesthesia blunts the baroreflex and changes breathing; animal presets were checked only on
  synthetic data (no demo dataset has animal pressure or respiration). The rodent and rabbit presets
  are partly Pennsieve choices (above): check the figures.
- Uncalibrated pressure gives timing only. A pressure signal whose `calibrated` flag says true but
  whose mean is outside 20–200 mmHg is flagged in `qc.csv` but still analysed.
- Breath amplitude is in the channel's unit (usually arbitrary); it is not a tidal volume unless the
  channel is calibrated in volume. Airflow and thermistor traces mark breaths as well, but their
  "inspiration time" is that of the flow signal. An inverted trace (inspiration down) swaps
  inspiration and expiration: set `RESP_POLARITY=inverted` when `qc.csv` suggests it.
- Without R-peaks, the baroreflex uses the pressure's pulse interval, which includes beat-to-beat
  changes in pulse transit time.
- Porges-Bohrer RSA is reported for humans only. Respiration-derived signals from the ECG (EDR) are
  not used: RSA needs a respiration channel.

## Licences

This app's code: Apache-2.0 (`LICENSE`). NeuroKit2 MIT; everything else in the image BSD, MIT,
Apache-2.0, MPL-2.0 (certifi) or PSF-style (`THIRD_PARTY.md`). No GPL code: PhysioZoo's preset
numbers are used, not its code. No licence notice is required for users. The demo data keep their
licences (Autonomic Aging ODbL 1.0, Fantasia ODC-By 1.0, Discover CC BY 4.0) and are not bundled.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in CITATION.cff.

Methods and software this app is built on:

- Makowski D, Pham T, Lau ZJ, Brammer JC, Lespinasse F, Pham H, Schölzel C, Chen SHA (2021).
  NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*
  53:1689–1696. [doi:10.3758/s13428-020-01516-y](https://doi.org/10.3758/s13428-020-01516-y), PMID 33528817.
- Zong W, Heldt T, Moody GB, Mark RG (2003). An open-source algorithm to detect onset of arterial
  blood pressure pulses. *Computers in Cardiology* 30:259–262.
  [doi:10.1109/CIC.2003.1291140](https://doi.org/10.1109/CIC.2003.1291140).
- Parati G, Di Rienzo M, Bertinieri G, Pomidossi G, Casadei R, Groppelli A, Pedotti A, Zanchetti A,
  Mancia G (1988). Evaluation of the baroreceptor-heart rate reflex by 24-hour intra-arterial blood
  pressure monitoring in humans. *Hypertension* 12:214–222.
  [doi:10.1161/01.HYP.12.2.214](https://doi.org/10.1161/01.HYP.12.2.214), PMID 3410530.
- Bertinieri G, di Rienzo M, Cavallazzi A, Ferrari AU, Pedotti A, Mancia G (1985). A new approach to
  analysis of the arterial baroreflex. *Journal of Hypertension. Supplement* 3:S79–S81. No DOI;
  [PubMed 2856787](https://pubmed.ncbi.nlm.nih.gov/2856787/).
- Bertinieri G, Di Rienzo M, Cavallazzi A, Ferrari AU, Pedotti A, Mancia G (1988). Evaluation of
  baroreceptor reflex by blood pressure monitoring in unanesthetized cats. *American Journal of
  Physiology-Heart and Circulatory Physiology* 254:H377–H383.
  [doi:10.1152/ajpheart.1988.254.2.H377](https://doi.org/10.1152/ajpheart.1988.254.2.H377), PMID 3344828.
- Laude D, Elghozi JL, Girard A, Bellard E, Bouhaddi M, Castiglioni P, Cerutti C, Cividjian A,
  Di Rienzo M, Fortrat JO, Janssen B, Karemaker JM, Lefthériotis G, Parati G, Persson PB, Porta A,
  Quintin L, Regnard J, Rüdiger H, Stauss HM (2004). Comparison of various techniques used to estimate
  spontaneous baroreflex sensitivity (the EuroBaVar study). *American Journal of Physiology-Regulatory,
  Integrative and Comparative Physiology* 286:R226–R231.
  [doi:10.1152/ajpregu.00709.2002](https://doi.org/10.1152/ajpregu.00709.2002), PMID 14500269.
- Khodadad D, Nordebo S, Müller B, Waldmann A, Yerworth R, Becher T, Frerichs I, Sophocleous L,
  van Kaam A, Miedema M, Seifnaraghi N, Bayford R (2018). Optimized breath detection algorithm in
  electrical impedance tomography. *Physiological Measurement* 39:094001.
  [doi:10.1088/1361-6579/aad7e6](https://doi.org/10.1088/1361-6579/aad7e6), PMID 30074906.
- Lewis GF, Furman SA, McCool MF, Porges SW (2012). Statistical strategies to quantify respiratory
  sinus arrhythmia: Are commonly used metrics equivalent? *Biological Psychology* 89:349–364.
  [doi:10.1016/j.biopsycho.2011.11.009](https://doi.org/10.1016/j.biopsycho.2011.11.009), PMID 22138367.

Preset values, reference values and background (full author lists in CITATION.cff):
Bagnall-Hare et al. 2024 [doi:10.1007/s13246-023-01380-y](https://doi.org/10.1007/s13246-023-01380-y);
Laude et al. 2009 [doi:10.1109/MEMB.2009.934630](https://doi.org/10.1109/MEMB.2009.934630);
Pan et al. 2023 [doi:10.3389/fphys.2023.1185744](https://doi.org/10.3389/fphys.2023.1185744);
Molcan et al. 2026 [doi:10.1038/s41440-026-02579-8](https://doi.org/10.1038/s41440-026-02579-8);
Kardos et al. 2001 [doi:10.1161/01.HYP.37.3.911](https://doi.org/10.1161/01.HYP.37.3.911);
Grossman & Taylor 2007 [doi:10.1016/j.biopsycho.2005.11.014](https://doi.org/10.1016/j.biopsycho.2005.11.014);
Grossman, van Beek & Wientjes 1990 (the 750 ms delay in NeuroKit2's P2T) [doi:10.1111/j.1469-8986.1990.tb03198.x](https://doi.org/10.1111/j.1469-8986.1990.tb03198.x);
Behar et al. 2018 (PhysioZoo) [doi:10.3389/fphys.2018.01390](https://doi.org/10.3389/fphys.2018.01390);
Behar et al. 2018b (mammalian HRV bands) [doi:10.3389/fphys.2018.01001](https://doi.org/10.3389/fphys.2018.01001);
Task Force of the European Society of Cardiology and the North American Society of Pacing and
Electrophysiology 1996 [doi:10.1161/01.CIR.93.5.1043](https://doi.org/10.1161/01.CIR.93.5.1043);
Thireau et al. 2008 [doi:10.1113/expphysiol.2007.040733](https://doi.org/10.1113/expphysiol.2007.040733);
Carnevali et al. 2023 [doi:10.3389/fphys.2023.1170320](https://doi.org/10.3389/fphys.2023.1170320);
Poletto et al. 2011 [doi:10.1016/j.physbeh.2011.01.019](https://doi.org/10.1016/j.physbeh.2011.01.019);
Salomão et al. 2015 [doi:10.1371/journal.pone.0134387](https://doi.org/10.1371/journal.pone.0134387);
Rübel et al. 2022 (NWB) [doi:10.7554/eLife.78362](https://doi.org/10.7554/eLife.78362);
Virtanen et al. 2020 (SciPy) [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2);
Xie et al. 2023 (WFDB for Python, validation) [doi:10.13026/9njx-6322](https://doi.org/10.13026/9njx-6322).

Demo and validation data: Schumann A, Bär KJ (2021). Autonomic Aging (PhysioNet)
[doi:10.13026/2hsy-t491](https://doi.org/10.13026/2hsy-t491) and Schumann & Bär 2022, *Scientific Data*
9:95 [doi:10.1038/s41597-022-01202-y](https://doi.org/10.1038/s41597-022-01202-y); Iyengar N et al.
Fantasia Database (PhysioNet) [doi:10.13026/C2RG61](https://doi.org/10.13026/C2RG61) and Iyengar et al.
1996, *Am J Physiol* 271:R1078 [doi:10.1152/ajpregu.1996.271.4.R1078](https://doi.org/10.1152/ajpregu.1996.271.4.R1078);
Goldberger et al. 2000 (PhysioNet) [doi:10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215);
Armitage et al. 2023, Pennsieve Discover 349 [doi:10.26275/dm5l-tspb](https://doi.org/10.26275/dm5l-tspb);
Bender et al. 2026, Pennsieve Discover 655 [doi:10.26275/5cwo-mkpz](https://doi.org/10.26275/5cwo-mkpz).

## Acknowledgements

NeuroKit2 is developed by Dominique Makowski and the NeuroKit2 contributors. The pressure-pulse
method is the one behind PhysioNet's `wabp` program (Zong, Heldt, Moody and Mark). PhysioNet asks
users of its databases to cite the database, its paper and Goldberger et al. 2000. Thanks to the authors of the Autonomic Aging and
Fantasia databases and of Pennsieve Discover datasets 349 and 655 for sharing their recordings.
