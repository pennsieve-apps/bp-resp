# Third-party components

The Docker image bundles these Python packages (exact versions in `requirements.txt`), installed
from PyPI wheels on `python:3.12.16-slim-bookworm` (Debian bookworm with CPython 3.12, PSF licence).
No data, models or other people's analysis code are bundled. The pressure-beat detector (after Zong
et al. 2003), the baroreflex (sequence and spectral methods), the beat and breath labelling and the
in-house peak-to-trough RSA for non-human species are this app's own code (Apache-2.0), written from
the published methods; PhysioNet's `wabp` C program was read for the method, not copied.

| Component | Version | Authors / copyright holder | Licence | URL |
|---|---|---|---|---|
| NeuroKit2 | 0.3.0 | Dominique Makowski, Tam Pham, Zen Juen Lau, Jan C. Brammer, François Lespinasse and the NeuroKit2 contributors | MIT | https://github.com/neuropsychology/NeuroKit |
| NumPy | 2.5.4 | NumPy Developers | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | SciPy Developers (Enthought, Inc. and contributors) | BSD-3-Clause (bundled parts under BSD/MIT-style licences) | https://scipy.org |
| pandas | 3.1.1 | The pandas Development Team | BSD-3-Clause | https://pandas.pydata.org |
| h5py | 3.16.0 | Andrew Collette and the h5py contributors | BSD-3-Clause | https://github.com/h5py/h5py |
| Matplotlib | 3.11.2 | John D. Hunter, Michael Droettboom and the Matplotlib Development Team | Matplotlib License (PSF-based) | https://matplotlib.org |
| scikit-learn | 1.9.1 | The scikit-learn developers | BSD-3-Clause | https://scikit-learn.org |
| joblib | 1.6.0 | Gael Varoquaux and the joblib developers | BSD-3-Clause | https://github.com/joblib/joblib |
| threadpoolctl | 3.8.0 | Thomas Moreau and contributors | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| cloudpickle | 3.1.2 | The cloudpickle developer team | BSD-3-Clause | https://github.com/cloudpipe/cloudpickle |
| narwhals | 2.28.0 | Marco Gorelli and the narwhals contributors | MIT | https://github.com/narwhals-dev/narwhals |
| PyWavelets | 1.10.0 | The PyWavelets Developers | MIT AND BSD-3-Clause | https://github.com/PyWavelets/pywt |
| requests | 2.34.2 | Kenneth Reitz and contributors | Apache-2.0 | https://github.com/psf/requests |
| certifi | 2026.10.23 | Kenneth Reitz and contributors | MPL-2.0 | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.2 | Ahmed R. Tahri | MIT | https://github.com/jawah/charset_normalizer |
| idna | 3.20 | Kim Davies | BSD-3-Clause | https://github.com/kjd/idna |
| urllib3 | 2.8.0 | Andrey Petrov and contributors | MIT | https://github.com/urllib3/urllib3 |
| setuptools | 81.0.0 | Python Packaging Authority | MIT | https://github.com/pypa/setuptools |
| python-dateutil | 4.0.0 | Gustavo Niemeyer, Paul Ganssle and dateutil contributors | Apache-2.0 AND BSD-3-Clause | https://github.com/dateutil/dateutil |
| six | 1.17.0 | Benjamin Peterson | MIT | https://github.com/benjaminp/six |
| tzdata | 2026.6 | Python Software Foundation | Apache-2.0 | https://github.com/python/tzdata |
| contourpy | 1.4.0 | Ian Thomas and the ContourPy developers | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | Matplotlib project | BSD-3-Clause | https://github.com/matplotlib/cycler |
| fonttools | 4.68.0 | Just van Rossum, Behdad Esfahbod and fontTools contributors | MIT | https://github.com/fonttools/fonttools |
| kiwisolver | 1.5.1 | Nucleic Development Team | BSD-3-Clause | https://github.com/nucleic/kiwi |
| packaging | 26.3 | Donald Stufft and contributors | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| Pillow | 13.0.0 | Jeffrey A. Clark and contributors | MIT-CMU | https://github.com/python-pillow/Pillow |
| pyparsing | 3.3.3 | Paul McGuire | MIT | https://github.com/pyparsing/pyparsing |

scikit-learn, joblib, threadpoolctl, cloudpickle, narwhals, PyWavelets, requests and its
dependencies, and setuptools are in the image only because NeuroKit2 declares them and imports
some of them when it loads; this app makes no network requests.

Test and validation only (not in the image): pytest (MIT); pynwb (BSD-3-Clause, Neurodata Without
Borders) writes the synthetic fixture's NWB files; wfdb (MIT, MIT Laboratory for Computational
Physiology) reads Fantasia's reference beat annotations in `validation/validate.py`.

Numbers taken from publications (not code): the species presets' detector, band and threshold
values (PhysioZoo, Behar et al. 2018, whose code is GPL-3.0 and is not used; Task Force 1996;
Thireau et al. 2008; Carnevali et al. 2023; Poletto et al. 2011; Salomão et al. 2015; Zong et al.
2003; Bagnall-Hare et al. 2024; Molcan et al. 2026; Pan et al. 2023; Laude et al. 2009), cited in
README.md and CITATION.cff.
