import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import numpy as np


class XRDAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = {}
        self.scan_df = None
        self.smoothed = None
        self.baseline = None
        self.corrected = None
        self.peaks = None
        self.fit_results = []

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(self):
        """Parse the Panalytical .csv file into metadata and scan DataFrames."""
        scan_rows = []
        in_scan_section = False

        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()

                if line.startswith("[Scan points]"):
                    in_scan_section = True
                    continue

                if not in_scan_section:
                    if not line.startswith("["):
                        key, value = line.split(",", 1)
                        self.metadata[key.strip()] = value.strip()
                    continue

                if line.startswith("Angle"):
                    continue
                if line:
                    angle, intensity = line.split(",")
                    scan_rows.append([float(angle), float(intensity)])

        self.scan_df = pd.DataFrame(scan_rows, columns=["Angle", "Intensity"])
        return self

    def get_metadata(self):
        """Return metadata as a formatted DataFrame."""
        return pd.DataFrame(list(self.metadata.items()), columns=["Field", "Value"])

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    def smooth(self, sigma=1):
        """Apply Gaussian smoothing to raw intensity."""
        self.smoothed = gaussian_filter1d(self.scan_df.Intensity, sigma=sigma)
        return self

    def fit_baseline(self, mask_ranges=None, deg=3):
        """Fit a polynomial baseline to non-peak regions and subtract it.

        Args:
            mask_ranges: list of (min, max) angle tuples to exclude from baseline fit.
                         Defaults to the two peak regions in GV008_GSO_Wide.csv.
            deg: polynomial degree for baseline fit.
        """
        if self.smoothed is None:
            self.smooth()

        two_theta = self.scan_df.Angle

        if mask_ranges is None:
            mask_ranges = [(20, 25), (42, 48)]

        mask = np.ones(len(two_theta), dtype=bool)
        for lo, hi in mask_ranges:
            mask &= ~((two_theta >= lo) & (two_theta <= hi))

        coeffs = np.polyfit(two_theta[mask], self.smoothed[mask], deg=deg)
        self.baseline = np.polyval(coeffs, two_theta)
        self.corrected = self.smoothed - self.baseline
        return self

    # -------------------------------------------------------------------------
    # Peak Detection and Fitting
    # -------------------------------------------------------------------------

    def detect_peaks(self, prominence=400, distance=20):
        """Find peaks in the baseline-corrected signal."""
        if self.corrected is None:
            self.fit_baseline()

        self.peaks, _ = find_peaks(self.corrected, prominence=prominence, distance=distance)
        return self

    @staticmethod
    def pseudo_voigt(x, x0, A, sigma, eta):
        """Pseudo-Voigt profile: linear combination of Lorentzian and Gaussian."""
        gaussian = A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
        lorentz = A * (sigma ** 2 / ((x - x0) ** 2 + sigma ** 2))
        return eta * lorentz + (1 - eta) * gaussian

    def fit_peaks(self, window=20):
        """Fit a pseudo-Voigt profile to each detected peak.

        Args:
            window: number of data points on each side of the peak to use for fitting.
        """
        if self.peaks is None:
            self.detect_peaks()

        two_theta = self.scan_df.Angle.values
        self.fit_results = []

        for p in self.peaks:
            left = max(0, p - window)
            right = min(len(two_theta), p + window)
            x = two_theta[left:right]
            y = self.corrected[left:right]

            p0 = [two_theta[p], self.corrected[p], 0.05, 0.5]
            popt, _ = curve_fit(self.pseudo_voigt, x, y, p0=p0)
            self.fit_results.append(popt)

        return self

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_scan(self):
        """Plot the raw XRD scan."""
        plt.figure(figsize=(10, 5))
        plt.semilogy(self.scan_df.Angle, self.scan_df.Intensity, linewidth=1)
        plt.xlabel(r"2$\theta$ Angle (degrees)")
        plt.ylabel("Intensity (counts)")
        plt.title("XRD Scan")
        plt.grid(True)
        plt.show()

    def plot_preprocessing(self):
        """Plot raw, smoothed, baseline, and corrected signals with detected peaks."""
        two_theta = self.scan_df.Angle

        plt.figure(figsize=(10, 6))
        plt.ylim([1, 1e6])
        plt.semilogy(two_theta, self.scan_df.Intensity, label="Raw", alpha=0.4)
        plt.semilogy(two_theta, self.smoothed, label="Smoothed")
        plt.semilogy(two_theta, self.baseline, label="Baseline")
        plt.semilogy(two_theta, self.corrected, label="Corrected")
        plt.scatter(two_theta.values[self.peaks], self.corrected[self.peaks],
                    color="red", label="Peaks")
        plt.xlabel(r"2$\theta$ (degrees)")
        plt.ylabel("Intensity (counts)")
        plt.title("XRD Peak Detection and Fitting")
        plt.legend()
        plt.show()

    def _build_peak_fits_figure(self, window=25):
        """Build and return the peak fits figure without displaying it."""
        two_theta = self.scan_df.Angle.values
        n_peaks = len(self.peaks)
        ncols = 2
        nrows = int(np.ceil(n_peaks / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()

        for i, (p, popt) in enumerate(zip(self.peaks, self.fit_results)):
            ax = axes[i]
            left = max(0, p - window)
            right = min(len(two_theta), p + window)

            x = two_theta[left:right]
            y = self.corrected[left:right]
            x_fit = np.linspace(x.min(), x.max(), 400)
            y_fit = self.pseudo_voigt(x_fit, *popt)

            ax.plot(x, y, "k.", label="Corrected data")
            ax.plot(x_fit, y_fit, "r-", label="Fitted curve", linewidth=2)
            ax.axvline(popt[0], color="blue", linestyle="--",
                       label=f"Peak @ {popt[0]:.3f}°")
            ax.set_title(f"Peak {i+1}: x0={popt[0]:.3f}, sigma={popt[2]:.4f}, eta={popt[3]:.2f}")
            ax.set_xlabel(r"2$\theta$ (degrees)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.legend()

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        return fig

    def plot_peak_fits(self, window=25):
        """Plot individual pseudo-Voigt fits for each detected peak."""
        fig = self._build_peak_fits_figure(window)
        plt.show()

    def save_peak_fits(self, filename="peak_fits.png", window=25, dpi=150):
        """Save the peak fits plot to a .png file in the current directory.

        Args:
            filename: output file name (default: 'peak_fits.png').
            window: number of data points on each side of each peak to include.
            dpi: image resolution (default: 150).
        """
        fig = self._build_peak_fits_figure(window)
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved peak fits plot to: {filename}")

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def run(self):
        """Execute the full analysis pipeline and display all plots."""
        self.load_data()
        self.smooth()
        self.fit_baseline()
        self.detect_peaks()
        self.fit_peaks()

        self.plot_scan()
        self.plot_preprocessing()
        self.plot_peak_fits()

        print(self.get_metadata().to_string(index=False))
        return self


if __name__ == "__main__":
    analyzer = XRDAnalyzer("GV008_GSO_Wide.csv")
    analyzer.run()
