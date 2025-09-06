# Audio-Pattern-Detection
This project detects how many times a **short audio clip** occurs inside a **longer audio file**.
It can be used for music analysis, speech phrase detection, or repeated pattern recognition.

The program is written in **Python** and comes with both a **command-line interface (CLI)** and a **Tkinter-based graphical user interface (GUI)**.

## ✨ Features

* Works with **.wav, .mp3, .m4a** and other formats (auto-converts to `.wav`).
* **Preprocessing**: resampling, normalization, and high-pass filtering.
* **Feature extraction**: log-mel spectrograms with normalization.
* **Pattern detection**: 2D normalized cross-correlation + peak detection.
* **Results**:
  * Number of detections
  * Timestamps for each match
  * Confidence scores
* **Visualization**: waveform plotted with markers at detection points.
* **User control**: adjustable threshold slider in GUI.

## 🚀 Usage

### Command Line

Run detection directly:

```bash
python audio_pattern_detector_app.py --template "clip.wav" --target "song.wav" --threshold 0.6
```

Output:

```
Number of detections: 4
Threshold used: 0.6
Template duration: 2.3s
Detections:
 - 00:15.2  (confidence: 0.82)
 - 01:12.7  (confidence: 0.79)
 - 02:45.1  (confidence: 0.88)
 - 03:33.0  (confidence: 0.81)
```

### GUI

Start the graphical interface:

```bash
python audio_pattern_detector_app.py
```

Steps:

1. Select the **template clip**, which is a short clip.
2. Select the **target audio file**, which is the full song.
3. Adjust the **threshold slider** (try to keep it in between 0.5-0.7 for the perfect result).
4. Click **Analyze**.

You’ll see results in a text box and a waveform with markers showing where the clip was detected.

## 🧠 How It Works

1. **Audio Processing**

   * Converts audio → consistent WAV format.
   * Resamples, normalizes, applies high-pass filter.
   * Extracts log-mel spectrograms for robust features.

2. **Pattern Matching**

   * Performs 2D normalized cross-correlation.
   * Detects peaks above a relative threshold.
   * Ensures minimum separation between detections.

3. **User Interface**

   * CLI for quick runs.
   * Tkinter GUI with file selection, threshold slider, and waveform visualization.

## 🎯 Applications

* Music phrase repetition detection
* Speech or word spotting
* Detecting plagiarism in audio samples
* Audio pattern recognition for research


