import os
import sys
import math
import threading
import shutil
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import scipy.signal as sig

# Core: audio + features + match

def _ensure_wav_or_convert(path_in: str, sr: int = 22050, mono: bool = True) -> str:
    """
    Returns a path to a WAV file. If input is already .wav, returns as-is.
    If it's a compressed/container format (.m4a/.mp3/.mp4/.aac), tries to convert with ffmpeg.
    """
    root, ext = os.path.splitext(path_in)
    ext = ext.lower()
    if ext == ".wav":
        return path_in

    out_wav = f"{root}__converted.wav"
    if os.path.exists(out_wav):
        return out_wav

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Can't convert; caller may still try to read with soundfile (might fail for m4a/mp3)
        return path_in

    cmd = [
        ffmpeg, "-y", "-i", path_in,
        "-ac", "1" if mono else "2",
        "-ar", str(sr),
        out_wav
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_wav
    except Exception:
        return path_in


def load_audio(path: str, sr: int = 22050, mono: bool = True):
    """
    Load audio as mono float32 normalized to [-1, 1].
    Uses ffmpeg conversion for common compressed formats if available.
    """
    safe_path = _ensure_wav_or_convert(path, sr=sr, mono=mono)
    try:
        y, file_sr = sf.read(safe_path, dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        # Resample if needed
        if file_sr != sr:
            # Use high-quality resample
            g = math.gcd(file_sr, sr)
            up = sr // g
            down = file_sr // g
            y = sig.resample_poly(y, up, down)
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / (np.max(np.abs(y)) + 1e-9)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load '{path}': {e}")


def highpass(y: np.ndarray, sr: int, cutoff_hz: float = 80.0):
    b, a = sig.butter(4, cutoff_hz / (sr / 2), btype="highpass")
    return sig.filtfilt(b, a, y)


def stft_mag(y: np.ndarray, sr: int, n_fft=1024, hop=256, window="hann"):
    """
    Magnitude spectrogram using SciPy STFT (power).
    """
    f, t, Z = sig.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window=window, padded=True, boundary="zeros")
    S = (np.abs(Z)) ** 2  # power
    return S, f, t


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0**(m / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 64, fmin: float = 50.0, fmax: float | None = None):
    """
    Create mel filterbank (triangles) for power spectrogram.
    """
    if fmax is None:
        fmax = sr / 2
    # FFT bin frequencies:
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    # Mel centers:
    mmin, mmax = hz_to_mel(fmin), hz_to_mel(fmax)
    m_pts = np.linspace(mmin, mmax, n_mels + 2)
    f_pts = mel_to_hz(m_pts)

    # Create filterbank
    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = f_pts[m - 1], f_pts[m], f_pts[m + 1]
        left = (fft_freqs - f_left) / (f_center - f_left + 1e-9)
        right = (f_right - fft_freqs) / (f_right - f_center + 1e-9)
        fb[m - 1] = np.maximum(0.0, np.minimum(left, right))
    # Normalize filters to unit area to keep energy scale consistent
    fb /= (fb.sum(axis=1, keepdims=True) + 1e-9)
    return fb


def logmel_spectrogram(y: np.ndarray, sr: int, n_fft=1024, hop=256, n_mels=64, fmin=50.0, fmax=None):
    """
    Log-mel spectrogram with per-band z-score normalization (better for matching).
    """
    S, freqs, times = stft_mag(y, sr, n_fft=n_fft, hop=hop)
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    Smel = fb @ S[: n_fft // 2 + 1, :]  # (n_mels, time_frames)
    # log scale (dB-like)
    Smel = np.log10(Smel + 1e-9)
    # per-mel-bin z-score normalization
    Smel = (Smel - Smel.mean(axis=1, keepdims=True)) / (Smel.std(axis=1, keepdims=True) + 1e-8)
    return Smel.astype(np.float32), times


def normalized_xcorr2d(X: np.ndarray, T: np.ndarray):
    """
    Compute a normalized 2D cross-correlation (NCC-like) of template T within X.
    X: (F, N), T: (F, M). Returns (F_valid, N_valid) NCC values.
    We only need time dimension, so we take correlate2d and then normalize.
    """
    from scipy.signal import correlate2d

    # raw correlation
    corr = correlate2d(X, T, mode="valid")

    # Pre-compute energy terms for normalization
    T_energy = np.sqrt((T * T).sum())
    kernel = np.ones_like(T, dtype=np.float32)
    X_energy_sq = correlate2d(X * X, kernel, mode="valid")
    denom = np.sqrt(X_energy_sq) * (T_energy + 1e-12)
    ncc = corr / (denom + 1e-12)

    # Reduce across frequency: take mean (robust), you may try max for sparser matches
    ncc_time = ncc.mean(axis=0)
    return ncc_time


def detect_repetitions(
    template_y: np.ndarray,
    target_y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 64,
    hp_cut: float = 80.0,
    thresh_rel: float = 0.60,
    min_sep_factor: float = 0.80,
):
    """
    Returns (count, times_sec, confidences, details_dict)
    - thresh_rel: peaks above (thresh_rel * max_score) are counted
    - min_sep_factor: min distance between peaks as a fraction of template duration
    """
    # Pre-emphasis / high-pass to reduce rumble
    template_y = highpass(template_y, sr, cutoff_hz=hp_cut)
    target_y = highpass(target_y, sr, cutoff_hz=hp_cut)

    # Features
    Tmel, _ = logmel_spectrogram(template_y, sr, n_fft=n_fft, hop=hop, n_mels=n_mels)
    Xmel, t_times = logmel_spectrogram(target_y, sr, n_fft=n_fft, hop=hop, n_mels=n_mels)

    # NCC-style 2D correlation reduced along frequency
    score = normalized_xcorr2d(Xmel, Tmel)  # 1D over time placements
    if score.size == 0:
        return 0, [], [], {
            "max_score": 0.0,
            "threshold": 0.0,
            "hop": hop,
            "sr": sr,
            "template_frames": Tmel.shape[1],
        }

    max_score = float(score.max())
    threshold = float(thresh_rel * max_score)

    # Peak picking with minimum distance ~ 0.8 * template duration
    template_frames = Tmel.shape[1]
    template_sec = template_frames * hop / sr
    min_distance_frames = max(1, int((min_sep_factor * template_sec) * sr / hop))

    peaks, props = sig.find_peaks(score, height=threshold, distance=min_distance_frames)
    times_sec = (peaks * hop) / sr
    confidences = props.get("peak_heights", np.array([], dtype=np.float32))

    return int(len(peaks)), times_sec.tolist(), confidences.tolist(), {
        "max_score": max_score,
        "threshold": threshold,
        "hop": hop,
        "sr": sr,
        "template_frames": template_frames,
        "template_sec": template_sec,
        "min_distance_frames": int(min_distance_frames),
    }


# Minimal CLI for quick test 

def cli():
    import argparse
    p = argparse.ArgumentParser(description="Count repeated phrase occurrences in audio via template matching.")
    p.add_argument("--template", required=True, help="Path to short template clip (e.g., 'Maula Mere cut.wav')")
    p.add_argument("--target", required=True, help="Path to long audio (e.g., 'Maula Mere Maula full song.wav')")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--threshold", type=float, default=0.60, help="Relative threshold (0-1)")
    p.add_argument("--plot", action="store_true", help="Show a quick plot (requires matplotlib)")
    args = p.parse_args()

    y_t, sr = load_audio(args.template, sr=args.sr)
    y_x, sr = load_audio(args.target, sr=args.sr)

    count, times, confs, meta = detect_repetitions(y_t, y_x, sr=sr, thresh_rel=args.threshold)
    print(f"Detections: {count}")
    for i, (tt, cc) in enumerate(zip(times, confs), 1):
        print(f"  {i:02d}: {tt:.2f}s (confidence={cc:.3f})")

    if args.plot:
        import matplotlib.pyplot as plt
        # quick waveform plot with markers
        t = np.arange(len(y_x)) / sr
        plt.figure(figsize=(12, 3))
        plt.plot(t, y_x)
        for tt in times:
            plt.axvline(tt, linestyle="--")
        plt.title(f"Detections: {count}")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()


# Tkinter GUI

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class AudioPatternApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Phrase Counter")
        self.root.geometry("900x650")

        self.sample_rate = 22050
        self.threshold_val = tk.DoubleVar(value=0.60)

        self.template_path = tk.StringVar()
        self.target_path = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        pad = {"padx": 10, "pady": 8}

        title = ttk.Label(self.root, text="Audio Phrase Counter (Template Matching)", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w", **pad)

        frm_files = ttk.LabelFrame(self.root, text="Audio Files")
        frm_files.pack(fill="x", **pad)

        # Template
        row = ttk.Frame(frm_files)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Template (short clip):", width=24).pack(side="left")
        ttk.Entry(row, textvariable=self.template_path).pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Button(row, text="Browse", command=self.browse_template).pack(side="left")

        # Target
        row2 = ttk.Frame(frm_files)
        row2.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2, text="Target (full audio):", width=24).pack(side="left")
        ttk.Entry(row2, textvariable=self.target_path).pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Button(row2, text="Browse", command=self.browse_target).pack(side="left")

        # Settings
        frm_set = ttk.LabelFrame(self.root, text="Detection Settings")
        frm_set.pack(fill="x", **pad)

        row3 = ttk.Frame(frm_set)
        row3.pack(fill="x", padx=8, pady=6)
        ttk.Label(row3, text="Relative Threshold:").pack(side="left")
        scale = ttk.Scale(row3, from_=0.30, to=0.95, orient="horizontal",
                          variable=self.threshold_val, command=lambda e: self.lbl_thresh.configure(text=f"{self.threshold_val.get():.2f}"))
        scale.pack(side="left", padx=8)
        self.lbl_thresh = ttk.Label(row3, text=f"{self.threshold_val.get():.2f}")
        self.lbl_thresh.pack(side="left", padx=4)

        row4 = ttk.Frame(frm_set)
        row4.pack(fill="x", padx=8, pady=6)
        ttk.Label(row4, text="Sample Rate:").pack(side="left")
        self.sr_combo = ttk.Combobox(row4, values=[16000, 22050, 32000, 44100], state="readonly")
        self.sr_combo.set(self.sample_rate)
        self.sr_combo.pack(side="left", padx=8)

        # Actions
        frm_actions = ttk.Frame(self.root)
        frm_actions.pack(fill="x", **pad)
        self.btn_run = ttk.Button(frm_actions, text="Analyze", command=self.start_analysis)
        self.btn_run.pack(side="left")
        self.progress = ttk.Progressbar(frm_actions, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        # Results
        frm_res = ttk.LabelFrame(self.root, text="Results")
        frm_res.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(frm_res, height=10)
        self.txt.pack(fill="x", padx=8, pady=8)

        # Plot
        frm_plot = ttk.LabelFrame(self.root, text="Visualization")
        frm_plot.pack(fill="both", expand=True, **pad)

        self.fig, self.ax = plt.subplots(figsize=(10, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def browse_template(self):
        path = filedialog.askopenfilename(
            title="Select Template Audio",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.aac *.flac *.mp4"), ("All files", "*.*")]
        )
        if path:
            self.template_path.set(path)

    def browse_target(self):
        path = filedialog.askopenfilename(
            title="Select Target Audio",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.aac *.flac *.mp4"), ("All files", "*.*")]
        )
        if path:
            self.target_path.set(path)

    def start_analysis(self):
        tpath = self.template_path.get().strip()
        xpath = self.target_path.get().strip()
        if not tpath or not xpath:
            messagebox.showerror("Error", "Please choose both Template and Target audio files.")
            return
        if not os.path.exists(tpath) or not os.path.exists(xpath):
            messagebox.showerror("Error", "One or both files do not exist.")
            return

        try:
            self.sample_rate = int(self.sr_combo.get())
        except Exception:
            self.sample_rate = 22050

        self.btn_run.config(state="disabled")
        self.progress.start(10)
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "Analyzing...\n")

        thread = threading.Thread(target=self._run_analysis, args=(tpath, xpath), daemon=True)
        thread.start()

    def _run_analysis(self, tpath, xpath):
        try:
            y_t, sr = load_audio(tpath, sr=self.sample_rate)
            y_x, sr = load_audio(xpath, sr=self.sample_rate)

            count, times, confs, meta = detect_repetitions(
                y_t, y_x, sr=sr,
                thresh_rel=float(self.threshold_val.get())
            )

            # Update UI
            self.root.after(0, lambda: self._show_results(count, times, confs, meta, y_x, sr))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))

    def _show_results(self, count, times, confs, meta, y_x, sr):
        self.progress.stop()
        self.btn_run.config(state="normal")

        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "ANALYSIS RESULTS\n")
        self.txt.insert(tk.END, "=" * 40 + "\n")
        self.txt.insert(tk.END, f"Detections: {count}\n")
        self.txt.insert(tk.END, f"Threshold (relative): {self.threshold_val.get():.2f}\n")
        self.txt.insert(tk.END, f"Max score: {meta['max_score']:.3f}\n")
        self.txt.insert(tk.END, f"Peak threshold: {meta['threshold']:.3f}\n")
        self.txt.insert(tk.END, f"Template duration (s): {meta['template_sec']:.2f}\n\n")
        if count == 0:
            self.txt.insert(tk.END, "No occurrences found. Try lowering the threshold.\n")
        else:
            self.txt.insert(tk.END, "Detection details:\n")
            for i, (tt, cc) in enumerate(zip(times, confs), 1):
                self.txt.insert(tk.END, f"  {i:02d}: Time = {tt:.2f}s, Confidence = {cc:.3f}\n")

        # Plot waveform + detections
        self.ax.clear()
        t = np.arange(len(y_x)) / sr
        self.ax.plot(t, y_x, alpha=0.8, label="Target Audio")
        for tt in times:
            self.ax.axvline(tt, linestyle="--", alpha=0.9)
        self.ax.set_title(f"Detections: {count}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right")
        self.canvas.draw()

    def _show_error(self, msg):
        self.progress.stop()
        self.btn_run.config(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, f"Error: {msg}\n")
        messagebox.showerror("Error", msg)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli()
        return
    root = tk.Tk()
    app = AudioPatternApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
