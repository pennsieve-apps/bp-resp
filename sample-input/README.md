# Sample input

bp-resp reads physio-import's `physio` output (port `physio`) and, optionally, stim-events' `events`
output (port `events`, `stim-events/events.tsv`) and ecg-hrv's `hrv` output (port `hrv`,
`ecg-hrv/beats.csv`).

## Synthetic (no download)

    python -m tests.make_fixture data/fixture   # data/fixture/physio, events, hrv and truth.json

## Real recordings used to test this app

Run physio-import on these first; its `physio` folder is this app's `physio` input. Link ecg-hrv's
output for R-peaks (otherwise bp-resp detects them itself).

- **PhysioNet Autonomic Aging** (doi:10.13026/2hsy-t491, ODbL 1.0; Schumann & Bär): records
  `0001`, `0002`, `0003` (`.hea` + `.dat`) and `subject-info.csv` from
  <https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/>: calibrated continuous
  non-invasive blood pressure (`NIBP`, mmHg) and two ECG leads at 1000 Hz, supine rest. Calibrated
  pressure, so the baroreflex is computed.
- **PhysioNet Fantasia** (doi:10.13026/C2RG61, ODC-By 1.0; Iyengar et al.): records `f1o01`, `f1y01`,
  `f2o01`, `f2o02`, `f2y01`, `f2y02` (`.hea`, `.dat`, `.ecg` beat annotations) from
  <https://physionet.org/content/fantasia/1.0.0/>: respiration and ECG, and in the f2 records an
  uncalibrated blood pressure (pulse timing only; no baroreflex).
- **Pennsieve Discover 349** (doi:10.26275/dm5l-tspb) and **655** (doi:10.26275/5cwo-mkpz): no blood
  pressure or respiration channels; bp-resp lists every recording as skipped (the "missing signals"
  path).
