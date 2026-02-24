# XRD Analysis

Processing and peak fitting of X-ray diffraction (XRD) data collected on sub-100nm thin films using a Panalytical XRD system.

---

## XRD_Analysis.ipynb

An exploratory notebook for processing XRD data. Steps through data extraction, preprocessing, peak detection, and pseudo-Voigt peak fitting in a cell-by-cell format, with inline plots for visual inspection at each stage.

## XRDAnalyzer.py

A production-ready OOP refactor of the notebook. Encapsulates the full analysis pipeline — CSV parsing, Gaussian smoothing, polynomial baseline subtraction, peak detection, and pseudo-Voigt fitting — into a single `XRDAnalyzer` class. Designed for reproducible, scriptable analysis with methods for displaying and saving figures. The full pipeline can be executed in a single `run()` call.

```python
from XRDAnalyzer import XRDAnalyzer

analyzer = XRDAnalyzer("GV008_GSO_Wide.csv")
analyzer.run()

# Save peak fits plot
analyzer.save_peak_fits("peak_fits.png", dpi=150)
```
