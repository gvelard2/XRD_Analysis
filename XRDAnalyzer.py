import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson, norm


class XRDAnalyzer:
    """Analyzer for Panalytical XRD .csv files.

    Performs Gaussian smoothing, polynomial and SNIP baseline subtraction,
    peak detection, pseudo-Voigt peak fitting, noise floor characterization,
    per-peak SNR analysis, and a data quality scorecard.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = {}
        self.scan_df = None
        self.smoothed = None
        self.baseline = None
        self.corrected = None
        self.baseline_snip = None
        self.corrected_snip = None
        self.peaks = None
        self.fit_results = []
        self.noise_df = None
        self.snr_df = None
        self._is_baseline = None   # True where no peak mask covers the point
        self._mask_ranges = None   # stored from fit_baseline() for reuse

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
                    if not line.startswith("[") and "," in line:
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

        Also builds the boolean baseline mask used by noise and SNR methods.

        Args:
            mask_ranges: list of (min, max) angle tuples to exclude from fit.
                         Defaults to [(20, 25), (42, 48)].
            deg: polynomial degree for baseline fit.
        """
        if self.smoothed is None:
            self.smooth()

        two_theta = self.scan_df.Angle
        if mask_ranges is None:
            mask_ranges = [(20, 25), (42, 48)]

        self._mask_ranges = mask_ranges

        mask = np.ones(len(two_theta), dtype=bool)
        for lo, hi in mask_ranges:
            mask &= ~((two_theta >= lo) & (two_theta <= hi))
        self._is_baseline = mask

        coeffs = np.polyfit(two_theta[mask], self.smoothed[mask], deg=deg)
        self.baseline = np.polyval(coeffs, two_theta)
        self.corrected = self.smoothed - self.baseline
        return self

    def fit_snip_baseline(self, n_iterations=40):
        """Estimate background using the SNIP algorithm and subtract it.

        SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
        is a non-parametric method widely used in XRF/XRD background
        correction (Ryan et al. 1988). Operates on the smoothed signal.

        Args:
            n_iterations: number of clipping iterations (default: 40).
        """
        if self.smoothed is None:
            self.smooth()

        y = self.smoothed.astype(float)
        # Variance-stabilising transform
        v = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
        v_work = v.copy()
        for p in range(1, n_iterations + 1):
            shifted_avg = (np.roll(v_work, p) + np.roll(v_work, -p)) / 2
            v_work = np.minimum(v_work, shifted_avg)
        # Invert transform
        self.baseline_snip = (np.exp(np.exp(v_work) - 1) - 1) ** 2 - 1
        self.corrected_snip = y - self.baseline_snip
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
            window: number of data points on each side of the peak to use.
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
    # Noise & Quality Analysis
    # -------------------------------------------------------------------------

    def analyze_noise_regions(self, noise_regions=None):
        """Characterize the noise floor in baseline-only angular sub-regions.

        Computes per-region mean, std dev, and Poisson ratio (σ/√μ).
        A ratio near 1.0 indicates shot-noise-limited (ideal) measurement;
        values above ~1.5 indicate excess noise sources.

        Args:
            noise_regions: list of (label, lo, hi) tuples defining sub-regions.
                           Defaults to four non-peak segments for a 15–55° scan.

        Returns:
            self (sets self.noise_df).
        """
        if self._is_baseline is None:
            self.fit_baseline()

        if noise_regions is None:
            noise_regions = [
                ("15–20°", 15, 20),
                ("25–32°", 25, 32),
                ("32–42°", 32, 42),
                ("48–55°", 48, 55),
            ]

        two_theta = self.scan_df.Angle.values
        intensity = self.scan_df.Intensity.values
        records = []
        for label, lo, hi in noise_regions:
            seg_mask = (two_theta >= lo) & (two_theta <= hi) & self._is_baseline
            seg = intensity[seg_mask]
            if len(seg) == 0:
                continue
            mu = seg.mean()
            sigma = seg.std()
            poisson_sigma = np.sqrt(mu)
            ratio = sigma / poisson_sigma if poisson_sigma > 0 else np.nan
            records.append({
                "Region": label,
                "N points": int(seg_mask.sum()),
                "Mean (counts)": round(mu, 1),
                "Std Dev": round(sigma, 2),
                "√Mean (Poisson)": round(poisson_sigma, 2),
                "σ / √μ ratio": round(ratio, 3),
                "Poisson-limited?": "Yes" if ratio < 1.5 else "No – excess noise",
            })
        self.noise_df = pd.DataFrame(records)
        print(self.noise_df.to_string(index=False))
        return self

    def compute_snr(self, window=60, excl_center=15):
        """Compute signal-to-noise ratio for each detected peak.

        SNR = peak amplitude above baseline / local noise std dev, estimated
        from a window around the peak excluding the peak center itself.

        Args:
            window: half-width (points) of the local noise estimation window.
            excl_center: half-width (points) excluded from noise estimate at
                         the peak center.

        Returns:
            self (sets self.snr_df).
        """
        if self.peaks is None:
            self.detect_peaks()

        two_theta = self.scan_df.Angle.values
        intensity = self.scan_df.Intensity.values
        records = []
        for p in self.peaks:
            lo = max(0, p - window)
            hi = min(len(two_theta), p + window)
            local_idx = np.arange(lo, hi)
            center_excl = (local_idx >= p - excl_center) & (local_idx <= p + excl_center)
            noise_counts = intensity[local_idx[~center_excl]]
            local_noise_std = noise_counts.std()
            local_noise_mean = noise_counts.mean()
            peak_amplitude = float(self.corrected[p])
            snr = peak_amplitude / local_noise_std if local_noise_std > 0 else np.nan
            poisson_snr = peak_amplitude / np.sqrt(local_noise_mean) if local_noise_mean > 0 else np.nan
            records.append({
                "2θ (°)": round(two_theta[p], 3),
                "Peak amplitude": int(peak_amplitude),
                "Local noise σ": round(local_noise_std, 1),
                "Measured SNR": round(snr, 1),
                "Poisson-limit SNR": round(poisson_snr, 1),
                "Quality": ("Reliable" if snr > 10 else
                            ("Marginal" if snr > 3 else "Below detection limit")),
            })
        self.snr_df = pd.DataFrame(records)
        print(self.snr_df.to_string(index=False))
        return self

    def print_scorecard(self):
        """Print a data quality summary scorecard to the console."""
        if self._is_baseline is None:
            self.fit_baseline()
        if self.corrected_snip is None:
            self.fit_snip_baseline()
        if self.snr_df is None:
            self.compute_snr()

        intensity = self.scan_df.Intensity.values
        two_theta = self.scan_df.Angle.values
        global_bl = intensity[self._is_baseline]
        global_mu = global_bl.mean()
        global_sigma = global_bl.std()
        global_ratio = global_sigma / np.sqrt(global_mu)

        res_poly = self.corrected[self._is_baseline].std()
        res_snip = self.corrected_snip[self._is_baseline].std()
        best = "Polynomial" if res_poly <= res_snip else "SNIP"

        anode = self.metadata.get("Anode material", "N/A")
        time_step = self.metadata.get("Time per step", "N/A")

        n_reliable = (self.snr_df["Measured SNR"] > 10).sum()
        n_marginal = ((self.snr_df["Measured SNR"] > 3) & (self.snr_df["Measured SNR"] <= 10)).sum()
        n_below = (self.snr_df["Measured SNR"] <= 3).sum()

        scorecard = {
            "Instrument": f"{self.metadata.get('Diffractometer system', 'N/A')} | {anode} Kα | {time_step}s/step",
            "Angular range": f"{two_theta[0]:.1f}° – {two_theta[-1]:.1f}° ({len(two_theta)} points)",
            "Global noise floor (mean)": f"{global_mu:.1f} counts",
            "Global noise σ": f"{global_sigma:.2f} counts",
            "Poisson ratio (σ/√μ)": f"{global_ratio:.3f}  {'← near ideal' if global_ratio < 1.5 else '← excess noise detected'}",
            "Best baseline method": f"{best} (residual σ: poly={res_poly:.2f}, SNIP={res_snip:.2f})",
            "Peaks detected": str(len(self.snr_df)),
            "Reliable peaks (SNR>10)": str(n_reliable),
            "Marginal peaks (3<SNR≤10)": str(n_marginal),
            "Below detection (SNR≤3)": str(n_below),
        }

        width = 60
        print("\n" + "=" * width)
        print("       XRD DATA QUALITY SCORECARD")
        print("=" * width)
        for k, v in scorecard.items():
            print(f"  {k:<34} {v}")
        print("=" * width)
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

    def plot_raw_overview(self):
        """Plot the raw pattern on both log and linear scales with peak regions shaded."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        two_theta = self.scan_df.Angle
        intensity = self.scan_df.Intensity
        mask_ranges = self._mask_ranges or [(20, 25), (42, 48)]

        axes[0].semilogy(two_theta, intensity, color='steelblue', linewidth=0.8, label='Raw counts')
        axes[0].set_ylabel('Intensity (counts, log)')
        axes[0].set_title('Raw XRD Pattern')
        axes[0].legend()

        axes[1].plot(two_theta, intensity, color='steelblue', linewidth=0.8, label='Raw counts')
        axes[1].set_ylabel('Intensity (counts, linear)')
        axes[1].set_xlabel(r'2$\theta$ (degrees)')
        axes[1].legend()

        for ax in axes:
            for i, (lo, hi) in enumerate(mask_ranges):
                ax.axvspan(lo, hi, alpha=0.15, color='orange',
                           label='Peak region' if i == 0 else '')
        plt.tight_layout()
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

    def plot_baselines(self):
        """Plot and compare polynomial vs. SNIP baseline estimates."""
        if self.baseline_snip is None:
            self.fit_snip_baseline()
        two_theta = self.scan_df.Angle
        mask_ranges = self._mask_ranges or [(20, 25), (42, 48)]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.semilogy(two_theta, self.smoothed, color='steelblue', lw=0.8, alpha=0.7, label='Smoothed data')
        ax.semilogy(two_theta, self.baseline, 'r--', lw=1.5, label='Polynomial (deg 3)')
        ax.semilogy(two_theta, self.baseline_snip, 'g-', lw=1.5, label='SNIP')
        for lo, hi in mask_ranges:
            ax.axvspan(lo, hi, alpha=0.1, color='orange')
        ax.set_xlabel(r'2$\theta$ (degrees)')
        ax.set_ylabel('Intensity (counts, log)')
        ax.set_title('Baseline Estimation: Polynomial vs. SNIP')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_baseline_residuals(self):
        """Plot baseline residuals in non-peak regions for both baseline methods."""
        if self.baseline_snip is None:
            self.fit_snip_baseline()
        if self._is_baseline is None:
            self.fit_baseline()
        two_theta = self.scan_df.Angle
        mask_ranges = self._mask_ranges or [(20, 25), (42, 48)]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for ax, residual, label, color in zip(
            axes,
            [self.corrected, self.corrected_snip],
            ['Polynomial residual (signal – poly baseline)',
             'SNIP residual (signal – SNIP baseline)'],
            ['tomato', 'seagreen']
        ):
            res_bl = residual[self._is_baseline]
            ax.plot(two_theta[self._is_baseline], res_bl, '.', color=color, ms=2, alpha=0.6)
            ax.axhline(0, color='black', lw=0.8, ls='--')
            ax.axhline(3 * res_bl.std(), color='gray', lw=0.8, ls=':', label='±3σ')
            ax.axhline(-3 * res_bl.std(), color='gray', lw=0.8, ls=':')
            for lo, hi in mask_ranges:
                ax.axvspan(lo, hi, alpha=0.1, color='orange')
            ax.set_ylabel('Residual (counts)')
            ax.set_title(f"{label}  |  σ_residual = {res_bl.std():.2f} counts")
            ax.legend()

        axes[-1].set_xlabel(r'2$\theta$ (degrees)')
        plt.suptitle('Baseline Residuals in Non-Peak Regions', fontsize=12)
        plt.tight_layout()
        plt.show()

    def _build_noise_distributions_figure(self, noise_regions=None):
        """Build and return the 2×2 noise distribution figure."""
        if self._is_baseline is None:
            self.fit_baseline()
        if noise_regions is None:
            noise_regions = [
                ("15–20°", 15, 20),
                ("25–32°", 25, 32),
                ("32–42°", 32, 42),
                ("48–55°", 48, 55),
            ]
        two_theta = self.scan_df.Angle.values
        intensity = self.scan_df.Intensity.values

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes = axes.flatten()
        for ax, (label, lo, hi) in zip(axes, noise_regions):
            seg_mask = (two_theta >= lo) & (two_theta <= hi) & self._is_baseline
            seg = intensity[seg_mask]
            mu, sigma = seg.mean(), seg.std()
            poisson_sigma = np.sqrt(mu)

            bins = np.arange(seg.min(), seg.max() + 2) - 0.5
            ax.hist(seg, bins=bins, color='steelblue', alpha=0.7, density=True, label='Observed')

            k_range = np.arange(max(0, int(mu - 4 * poisson_sigma)), int(mu + 4 * poisson_sigma))
            ax.plot(k_range, poisson.pmf(k_range, mu), 'r-o', markersize=3, linewidth=1.5,
                    label=f'Poisson(μ={mu:.1f})')

            x_range = np.linspace(seg.min(), seg.max(), 200)
            ax.plot(x_range, norm.pdf(x_range, mu, sigma), 'g--', linewidth=1.5,
                    label=f'Gaussian(σ={sigma:.1f})')

            ratio = sigma / poisson_sigma
            ax.set_title(f"{label}  |  σ/√μ = {ratio:.2f}", fontsize=10)
            ax.set_xlabel('Intensity (counts)')
            ax.set_ylabel('Probability density')
            ax.legend(fontsize=8)

        plt.suptitle('Noise Distribution by Angular Region (σ/√μ ≈ 1 = Poisson-limited)',
                     fontsize=12, y=1.01)
        plt.tight_layout()
        return fig

    def plot_noise_distributions(self, noise_regions=None):
        """Display noise histogram with Poisson and Gaussian overlays per region."""
        fig = self._build_noise_distributions_figure(noise_regions)
        plt.show()

    def save_noise_distributions(self, filename="noise_distributions.png",
                                  noise_regions=None, dpi=150):
        """Save the noise distribution figure to a PNG.

        Args:
            filename: output file name (default: 'noise_distributions.png').
            noise_regions: optional list of (label, lo, hi) sub-region tuples.
            dpi: image resolution (default: 150).
        """
        import os
        fig = self._build_noise_distributions_figure(noise_regions)
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved noise distributions to: {out_path}")

    def plot_angular_noise_profile(self, window=50):
        """Plot rolling noise floor mean, std band, and Poisson ratio vs. angle."""
        if self._is_baseline is None:
            self.fit_baseline()
        two_theta = self.scan_df.Angle.values
        intensity = self.scan_df.Intensity.values
        bt = two_theta[self._is_baseline]
        bc = intensity[self._is_baseline].astype(float)

        sort_idx = np.argsort(bt)
        bt, bc = bt[sort_idx], bc[sort_idx]

        rolling_mean = np.convolve(bc, np.ones(window) / window, mode='valid')
        rolling_std = np.array([bc[i:i + window].std() for i in range(len(bc) - window + 1)])
        rolling_poisson = np.sqrt(rolling_mean)
        rolling_theta = bt[window // 2: window // 2 + len(rolling_mean)]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(bt, bc, '.', ms=2, color='steelblue', alpha=0.5, label='Baseline counts')
        axes[0].plot(rolling_theta, rolling_mean, 'r-', lw=2, label='Rolling mean')
        axes[0].fill_between(rolling_theta,
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.25, color='red', label='±1σ band')
        axes[0].set_ylabel('Intensity (counts)')
        axes[0].set_title('Baseline Count Rate vs. Angle')
        axes[0].legend()

        ratio = rolling_std / rolling_poisson
        axes[1].plot(rolling_theta, ratio, color='purple', lw=1.5)
        axes[1].axhline(1.0, color='black', ls='--', lw=0.8, label='Ideal Poisson (ratio = 1)')
        axes[1].axhline(1.5, color='gray', ls=':', lw=0.8, label='1.5× threshold')
        axes[1].fill_between(rolling_theta, 1.0, ratio,
                             where=(ratio > 1.5), alpha=0.3, color='orange',
                             label='Excess noise region')
        axes[1].set_ylabel('σ_actual / √μ  (Poisson ratio)')
        axes[1].set_xlabel(r'2$\theta$ (degrees)')
        axes[1].set_title('Poisson Noise Ratio vs. Angle  (1.0 = shot-noise limited)')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def _build_snr_figure(self):
        """Build and return the per-peak SNR bar chart figure."""
        if self.snr_df is None:
            self.compute_snr()
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(self.snr_df))
        colors = ['seagreen' if snr > 10 else ('goldenrod' if snr > 3 else 'tomato')
                  for snr in self.snr_df['Measured SNR']]
        ax.bar(positions, self.snr_df['Measured SNR'], color=colors, alpha=0.85, label='Measured SNR')
        ax.bar(positions, self.snr_df['Poisson-limit SNR'], color='steelblue', alpha=0.35,
               label='Poisson-limit SNR')
        ax.axhline(10, color='seagreen', ls='--', lw=1.2, label='SNR = 10 (reliable)')
        ax.axhline(3, color='tomato', ls='--', lw=1.2, label='SNR = 3 (detection limit)')
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{t}°" for t in self.snr_df['2θ (°)']], rotation=30)
        ax.set_ylabel('Signal-to-Noise Ratio')
        ax.set_title('SNR per Detected Peak\n(green=reliable, gold=marginal, red=below limit)')
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_snr(self):
        """Display the per-peak SNR bar chart."""
        fig = self._build_snr_figure()
        plt.show()

    def save_snr(self, filename="snr_per_peak.png", dpi=150):
        """Save the per-peak SNR bar chart to a PNG.

        Args:
            filename: output file name (default: 'snr_per_peak.png').
            dpi: image resolution (default: 150).
        """
        import os
        fig = self._build_snr_figure()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved SNR figure to: {out_path}")

    def _build_peak_fits_figure(self, window=25):
        """Build and return the peak fits figure without displaying it."""
        two_theta = self.scan_df.Angle.values
        ncols = 2
        nrows = int(np.ceil(len(self.peaks) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()

        for i, (p, popt) in enumerate(zip(self.peaks, self.fit_results)):
            ax = axes[i]
            left = max(0, p - window)
            right = min(len(two_theta), p + window)
            x = two_theta[left:right]
            y = self.corrected[left:right]
            x_fit = np.linspace(x.min(), x.max(), 400)
            ax.plot(x, y, "k.", label="Corrected data")
            ax.plot(x_fit, self.pseudo_voigt(x_fit, *popt), "r-", linewidth=2, label="Fit")
            ax.axvline(popt[0], color="blue", linestyle="--", label=f"Peak @ {popt[0]:.3f}°")
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
        """Save the peak fits plot to a PNG.

        Args:
            filename: output file name (default: 'peak_fits.png').
            window: number of data points on each side of each peak to include.
            dpi: image resolution (default: 150).
        """
        import os
        fig = self._build_peak_fits_figure(window)
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved peak fits plot to: {out_path}")

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
        self.fit_snip_baseline()

        self.plot_scan()
        self.plot_raw_overview()
        self.plot_preprocessing()
        self.plot_baselines()
        self.plot_baseline_residuals()
        self.analyze_noise_regions()
        self.plot_noise_distributions()
        self.plot_angular_noise_profile()
        self.compute_snr()
        self.plot_snr()
        self.plot_peak_fits()
        self.print_scorecard()

        print(self.get_metadata().to_string(index=False))
        return self


if __name__ == "__main__":
    analyzer = XRDAnalyzer("GV008_GSO_Wide.csv")
    analyzer.run()
