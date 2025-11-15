#!/usr/bin/env python3
"""
Parallel real-time KWS using TF audio_ops, with enhanced live logging using rich.

This script performs real-time keyword spotting using a pre-trained
TensorFlow model. Audio is continuously captured from the microphone
in fixed-duration chunks and processed in parallel threads for
efficient inference.

Features:
- Parallel audio recording and inference threads.
- Live Rich dashboard for monitoring status.
- Keyword-based action responses ("on", "off", "stop").
- Automatic cleanup of temporary WAV files.

Usage:
    python microphone_parallel.py --keras_file_path cnn.h5 --labels labels.txt --device 6
"""
import argparse
import os
import time
import threading
import queue
import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box

import csv
from datetime import datetime

# --------- Globals ----------
CHUNK_QUEUE = queue.Queue(maxsize=8)    # Shared buffer for recorded audio chunks
STOP_FLAG = threading.Event()   # Signal to stop all threads
MODEL_LOCK = threading.Lock()   # Prevent Concurrent model.predict() calls

# --------- Setup console UI ----------
console = Console()  # Track state of each audio chunk
chunk_status = {}  # {chunk_id: {"status": str, "prediction": str, "score": float, "timestamp": str}}
chunk_lock = threading.Lock()   # Synchronization for chunk_status dict

# --------- Save in CSV ----------
LOG_DIR = "Time_data"
os.makedirs(LOG_DIR, exist_ok=True)
EVENTS_PAR = os.path.join(LOG_DIR, "events_par.csv")
if not os.path.exists(EVENTS_PAR):
    with open(EVENTS_PAR, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id","record_start","record_end","queued_time","infer_start","infer_end","label","score"])

# --------- TF Audio helpers ----------
def load_wav_file(wav_filename, desired_samples):
    """
    Load and decode a WAV file using TensorFlow's audio ops.

    Args:
        wav_filename (str): Path to the WAV file.
        desired_samples (int): Number of samples to read from the file.

    Returns:
        tuple: (decoded_audio: tf.Tensor, sample_rate: tf.Tensor)
    """
    wav_file = tf.io.read_file(wav_filename)
    decoded_wav = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=desired_samples)
    return decoded_wav.audio, decoded_wav.sample_rate

def calculate_mfcc(audio_signal, audio_sample_rate, window_size, window_stride, num_mfcc):
    """
    Compute MFCC (Mel-Frequency Cepstral Coefficients) features from audio.

    Args:
        audio_signal (tf.Tensor): Audio waveform.
        audio_sample_rate (tf.Tensor): Sampling rate of audio.
        window_size (int): Window size in samples.
        window_stride (int): Step size between windows.
        num_mfcc (int): Number of MFCC coefficients to compute.

    Returns:
        tf.Tensor: MFCC feature tensor.
    """
    spectrogram = audio_ops.audio_spectrogram(
        input=audio_signal,
        window_size=window_size,
        stride=window_stride,
        magnitude_squared=True
    )
    mfcc_features = audio_ops.mfcc(spectrogram, audio_sample_rate, dct_coefficient_count=num_mfcc)
    return mfcc_features

def load_labels(filename):
    """
    Load label names from a text file.

    Args:
        filename (str): Path to labels.txt file.

    Returns:
        list[str]: List of label strings.
    """
    with open(filename, "r") as f:
        return f.read().splitlines()

# --------- Recorder Thread ----------
def recorder_thread(wav_basepath, sample_rate, duration, device=None):
    """
    Continuously record microphone audio and enqueue chunks for inference.

    Args:
        wav_basepath (str): Prefix for saving temporary WAV files.
        sample_rate (int): Recording sampling rate.
        duration (float): Duration of each chunk (in seconds).
        device (int, optional): Microphone device index.

    Behavior:
        - Records fixed-duration chunks.
        - Saves each chunk as a WAV file.
        - Pushes (chunk_id, file_path) to CHUNK_QUEUE.
    """
    if device is not None:
        try:
            sd.default.device = device
        except Exception as e:
            console.print(f"[yellow][WARN][/yellow] Could not set device {device}: {e}")

    idx = 0
    while not STOP_FLAG.is_set():
        idx += 1
        timestamp = time.strftime("%H:%M:%S")
        chunk_name = f"chunk_{idx}"
        with chunk_lock:
            chunk_status[chunk_name] = {"status": "🎙️ Recording", "prediction": "-", "score": "-", "timestamp": timestamp}

        rec_start=time.time()
        try:
            # Capture audio
            data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            rec_end=time.time()
            data = data.reshape(-1)
            
            # Save to disk temporarily
            filename = f"{wav_basepath}_{int(time.time()*1000)}_{idx}.wav"
            write(filename, sample_rate, data)
            
            # Send to inference queue
            queued_time=time.time()
            CHUNK_QUEUE.put_nowait((chunk_name, filename))
            
            # Update status
            with chunk_lock:
                chunk_status[chunk_name]["status"] = "📁 Saved"
                
        except queue.Full:
            with chunk_lock:
                chunk_status[chunk_name]["status"] = "⚠️ Dropped"
        except Exception as e:
            rec_end=time.time()
            console.print(f"[red][ERROR][/red] Recording failed: {e}")
            with chunk_lock:
                chunk_status[chunk_name]["status"] = "❌ Error"
                
        # log recorder record timestamps + queued time
        with open(EVENTS_PAR, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([chunk_name, rec_start, rec_end, queued_time, "", "", "", ""])

# --------- Inference Thread ----------
def inference_thread(keras_path, labels_path, sample_rate, window_size_ms, window_stride_ms, dct_count):
    """
    Continuously perform model inference on recorded audio chunks.

    Args:
        keras_path (str): Path to the trained Keras model file (.h5).
        labels_path (str): Path to the labels file.
        sample_rate (int): Sampling rate for decoding audio.
        window_size_ms (float): Window size (milliseconds) for MFCC.
        window_stride_ms (float): Window stride (milliseconds) for MFCC.
        dct_count (int): Number of DCT coefficients for MFCC.

    Behavior:
        - Loads the model and labels.
        - Fetches WAV files from CHUNK_QUEUE.
        - Extracts MFCCs and predicts the keyword.
        - Updates live dashboard.
        - Responds to specific keywords ("on", "off", "stop").
    """
    model = tf.keras.models.load_model(keras_path)
    labels = load_labels(labels_path)
    
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    
    console.print(f"[green][{time.strftime('%H:%M:%S')}] Model loaded for inference.[/green]")

    while not STOP_FLAG.is_set() or not CHUNK_QUEUE.empty():
        try:
            chunk_name, filename = CHUNK_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue
        inf_start=time.time()
        try:
            # Extract features from the recorder chunk
            decoded, sample_tf = load_wav_file(filename, sample_rate)
            mfcc_tf = calculate_mfcc(decoded, sample_tf, window_size_samples, window_stride_samples, dct_count)
            x = tf.reshape(mfcc_tf, [1, -1])
            
            # Thread-safe model inference
            with MODEL_LOCK:
                preds = model.predict(x, verbose=0)

            top = preds[0].argmax()
            human_label = labels[top]
            score = preds[0][top]
            inf_end = time.time()

            # --- CSV logging for inference stage ---
            with open(EVENTS_PAR, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([chunk_name, "", "", "", inf_start, inf_end, human_label, score])
                
            # Update UI data
            with chunk_lock:
                chunk_status[chunk_name]["status"] = "✅ Inferred"
                chunk_status[chunk_name]["prediction"] = human_label
                chunk_status[chunk_name]["score"] = f"{score:.3f}"

            # Action Hooks
            if human_label == "on":
                print("\t+#@+#@+#@+       Welcome     +#@+#@+#@+")
            elif human_label == "off":
                print("\t+#@+#@+#@+      Thank you     +#@+#@+#@+")
            elif human_label == "stop":
                print("\t+#@+#@+#@+   have a Nice day   +#@+#@+#@+")
                STOP_FLAG.set()

        except Exception as e:
            inf_end = time.time()
            
            with chunk_lock:
                chunk_status[chunk_name]["status"] = "❌ Error"
            console.print(f"[red][ERROR][/red] Inference failed for {filename}: {e}")
        finally:
            # Clean the temporary files post inferencing
            try:
                os.remove(filename)
            except Exception:
                pass

# --------- Rich Table Builder ----------
def build_table():
    """
    Build the live dashboard table for Rich console.

    Returns:
        rich.table.Table: Formatted dashboard table object.
    """
    table = Table(title="🎧 Real-time Keyword Spotting Dashboard", box=box.ROUNDED, title_style="bold cyan")
    table.add_column("Chunk ID", justify="center")
    table.add_column("Timestamp", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Prediction", justify="center")
    table.add_column("Confidence", justify="center")

    with chunk_lock:
        for k, v in list(chunk_status.items())[-15:]:  # show last 15 chunks
            table.add_row(k, v["timestamp"], v["status"], v["prediction"], str(v["score"]))
    return table

# --------- Main Entrypoint ----------
def main():
    """
    Entry point for the KWS system.
    Initializes arguments, threads, and dashboard.
    """
    parser = argparse.ArgumentParser(description="Parallel real-time Keyword Spotting with TensorFlow")
    parser.add_argument('--keras_file_path', required=True, help="Path to trained Keras model (.h5)")
    parser.add_argument('--labels', required=True, help="Path to labels text file")
    parser.add_argument('--wav_base', default='microphone', help="Base filename prefix for recordings")
    parser.add_argument('--device', type=int, default=None, help="Audio input device index")
    parser.add_argument('--sample_rate', type=int, default=16000, help="Audio sample rate (Hz)")
    parser.add_argument('--window_size_ms', type=float, default=40.0, help="Window size in ms for MFCC")
    parser.add_argument('--window_stride_ms', type=float, default=20.0, help="Window stride in ms for MFCC")
    parser.add_argument('--dct_coefficient_count', type=int, default=10, help="Number of DCT coefficients for MFCC")
    parser.add_argument('--duration', type=float, default=1.0, help="Chunk duration in seconds")
    args = parser.parse_args()

    # --- Pre-run validation ---
    if not os.path.exists(args.keras_file_path):
        raise SystemExit(f"Model not found: {args.keras_file_path}")
    if not os.path.exists(args.labels):
        raise SystemExit(f"Labels file not found: {args.labels}")

    # --- Thread setup ---
    rec_thread = threading.Thread(
        target=recorder_thread,
        args=(args.wav_base, args.sample_rate, args.duration, args.device),
        daemon=True
    )
    inf_thread = threading.Thread(
        target=inference_thread,
        args=(args.keras_file_path, args.labels, args.sample_rate,
              args.window_size_ms, args.window_stride_ms, args.dct_coefficient_count),
        daemon=True
    )

    console.print(f"\n[cyan][{time.strftime('%H:%M:%S')}] Starting parallel recorder + inference "
                  f"(sr={args.sample_rate}Hz, dur={args.duration}s)[/cyan]")

    rec_thread.start()
    inf_thread.start()

    # --- Live Dashboard ---
    with Live(build_table(), refresh_per_second=2, console=console) as live:
        try:
            while not STOP_FLAG.is_set():
                live.update(build_table())
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\n[bold red]Stopping manually...[/bold red]")
            STOP_FLAG.set()

    rec_thread.join(timeout=1.0)
    inf_thread.join(timeout=1.0)

    console.print(f"[green][{time.strftime('%H:%M:%S')}] Program terminated cleanly.[/green]")


if __name__ == "__main__":
    main()