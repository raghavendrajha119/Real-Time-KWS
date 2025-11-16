# Keyword Spotting — Sequential & Parallel Real-time Demo

This repository contains a small real-time Keyword Spotting (KWS) demo and tooling:

* `microphone.py` — **Sequential** capture → process → predict loop (1s chunks).
* `microphone_parallel.py` — **Parallel** recorder + inference using threads with a live **Rich** dashboard.
* `graphs_gen.py` — generate timing visualizations from CSV logs to compare sequential vs parallel behavior.
* `Time_data/` — CSV logs for sequential and parallel runs (created automatically).
* `KWS_waves/` — where timeline plots are saved (you can change this).
* `requirements.txt` — Python dependencies.
* `folder_struc.txt` — your repo folder layout (kept in repo).

This markdown explains setup, how to run sequential and parallel modes, where to find CSV logs, how to generate the timing plots, and how to interpret them. It also includes recommended tips and troubleshooting notes.

---

## Quick summary (which to use when)

* **Sequential (`microphone.py`)** — easiest, deterministic. Good for sanity checks and training model compatibility. Records 1 s → blocks until inference finishes → writes CSV row (record + inference timestamps).
* **Parallel (`microphone_parallel.py`)** — recorder runs continuously in fixed chunks and inference runs concurrently on previously recorded chunks. Use this to measure and reduce end-to-end latency and to visualize concurrency. Provides a live textual + tabular dashboard via `rich`.

---

## Setup

1. Clone repository (or place code files somewhere):

   ```bash
   git clone https://github.com/raghavendrajha119/Real-Time-KWS.git
   cd Real-Time-KWS
   ```

2. Create a Python virtual environment (recommended) and install dependencies:

   ```bash
   python3.10 -m venv env310         # Using 3.10 model compatible
   source .env310/bin/activate          # Linux / macOS
   .env310\Scripts\activate             # Windows PowerShell

   pip install -r requirements.txt
   ```

3. Ensure the following files are present:

   * `cnn.h5` — your trained Keras model (or update `--keras_file_path` argument)
   * `labels.txt` — label names, one per line (e.g. `_silence_`, `_unknown_`, `on`, `off`, `stop`)
   * `folder_struc.txt` — optional, describes project layout

4. Folders:

   * `Time_data/` — will be created automatically (contains `events_seq.csv` and `events_par.csv`).
   * `KWS_waves/` — created when you generate plots (or you can create it manually).

---

## Run Sequential Demo (blocking)

This replicates the original single-threaded behavior (record 1s → compute MFCC → predict → repeat).

**Command**

```bash
python microphone.py --keras_file_path cnn.h5 --labels labels.txt
```

**Notes**

* Default sample rate = `16000`, chunk duration = `1.0s`. Change via flags if needed.
* Each chunk appends a row to `Time_data/events_seq.csv`:

  ```
  chunk_id, record_start, record_end, infer_start, infer_end, label, score
  ```

  All times are epoch seconds (floating point). Use these for plotting.

**Stop**

* Say the `stop` keyword (if recognized by your model), or press `Ctrl+C`.

---

## Run Parallel Demo (recorder + inference + dashboard)

This runs the recorder and inference in separate threads. The recorder writes temporary WAVs and queues them for inference. The inference thread processes queued files and updates a live dashboard (via `rich`).

**Command**

```bash
python microphone_parallel.py --keras_file_path cnn.h5 --labels labels.txt
```

**Important flags**

* `--wav_base` (default `microphone`) — base filename prefix used for temporary WAVs.
* `--duration` (default `1.0`) — chunk duration (seconds). Lowering duration increases chunk rate.
* `--device` — numeric audio input device index (use `python -m sounddevice` to list devices).

**Logs**

* Recorder writes a CSV row for each recorded chunk to `Time_data/events_par.csv`:

  ```
  chunk_id, record_start, record_end, queued_time, infer_start, infer_end, label, score
  ```

  * `record_start`, `record_end` — when the recorder began/finished capturing that chunk.
  * `queued_time` — when the WAV was pushed to the inference queue.
  * `infer_start`, `infer_end`, `label`, `score` — inference timings and results (filled by inference thread).

**Live dashboard**

* The dashboard shows: chunk id, timestamp, status (`Recording`, `Saved`, `Inferred`, `Error`), prediction and confidence.
* It helps visually map recorder activity vs inference.

**Stop**

* Say the `stop` keyword (if recognized), or press `Ctrl+C`. The inference thread drains the queue before shutdown (so last queued items are processed). If you configured `stop` detection to set the stop flag immediately, recorder may stop soon afterwards — the implementation attempts to finish queued work first.

---

## Generate timing plots (graphs)

Use `graphs_gen.py` to visualize CSV records and produce timeline PNGs. The script visually shows recording intervals (blue), inference intervals (orange) and queue moments (vertical `Q`).

**Example usage**

1. Sequential plot:

   ```bash
   python graphs_gen.py seq Time_data/events_seq.csv KWS_waves/seq_timeline.png
   ```

2. Parallel plot:

   ```bash
   python graphs_gen.py par Time_data/events_par.csv KWS_waves/par_timeline.png
   ```

3. Quick helper (create `KWS_waves/` and generate both if you have both CSVs):

   ```bash
   mkdir -p KWS_waves
   python graphs_gen.py seq Time_data/events_seq.csv KWS_waves/seq_timeline.png
   python graphs_gen.py par Time_data/events_par.csv KWS_waves/par_timeline.png
   ```

**What you’ll see**

* **Blue rectangles** — the recording duration for each chunk (start → end).
* **Orange rectangles** — the inference duration for each chunk (infer start → infer end).
* **Vertical `Q` marker** (parallel mode) — the moment the chunk was enqueued for inference.
* X-axis is seconds relative to earliest logged event.

**Interpretation**

* If bars do **not overlap** in the sequential plot, recording blocks inference (as expected).
* In the parallel plot, **overlap** between a blue recording bar for a later chunk and an orange inference bar for an earlier chunk demonstrates concurrency (recorder and inference working in parallel).
* Long gaps between `queued_time` and `infer_start` indicate queue/backlog problems (model too slow or queue too small).

---

## Folder structure (example)

`folder_struc.txt` should already match your layout, e.g.:

```
.
├── microphone.py                 # sequential script
├── microphone_parallel.py        # parallel script + rich dashboard
├── graphs_gen.py              # graphs generator
├── requirements.txt
├── cnn.h5                        # trained model
├── labels.txt
├── Time_data/                    # CSV logs (auto-created)
│   ├── events_seq.csv
│   └── events_par.csv
├── KWS_waves/                    # generated PNGs
└── folder_struc.txt
```

---

## requirements.txt (recommended)

Make sure your `requirements.txt` contains at least:

```
tensorflow
numpy
sounddevice
scipy
rich
matplotlib
librosa  # optional if you also generate audio / spectrogram visuals
```

Install with:

```bash
pip install -r requirements.txt
```

*Note*: TensorFlow may print GPU driver warnings if you do not have CUDA/TensorRT installed — this is normal and nonfatal.

---
