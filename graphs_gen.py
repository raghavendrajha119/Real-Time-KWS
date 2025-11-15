#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import sys
import numpy as np
import os
from datetime import datetime

# ----------------------------------------------
# Helper: Read Sequential CSV (events_seq.csv)
# ----------------------------------------------
def read_seq(csvfile):
    rows = []
    with open(csvfile, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "chunk": row["chunk_id"],
                "rec_start": float(row["record_start"]) if row["record_start"] else None,
                "rec_end": float(row["record_end"]) if row["record_end"] else None,
                "inf_start": float(row["infer_start"]) if row["infer_start"] else None,
                "inf_end": float(row["infer_end"]) if row["infer_end"] else None,
                "label": row.get("label",""),
                "score": float(row["score"]) if row.get("score") else None
            })
    return rows

# ----------------------------------------------
# Helper: Read Parallel CSV (events_par.csv)
# Mixed rows → must merge by chunk_id
# ----------------------------------------------
def read_par(csvfile):
    chunks = {}
    with open(csvfile, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row["chunk_id"]
            if cid not in chunks:
                chunks[cid] = {
                    "rec_start": None, "rec_end": None, "queued": None,
                    "inf_start": None, "inf_end": None,
                    "label": None, "score": None
                }

            # Fill fields only when present
            if row["record_start"]:
                chunks[cid]["rec_start"] = float(row["record_start"])
            if row["record_end"]:
                chunks[cid]["rec_end"] = float(row["record_end"])
            if row.get("queued_time"):
                chunks[cid]["queued"] = float(row["queued_time"])
            if row.get("infer_start"):
                chunks[cid]["inf_start"] = float(row["infer_start"])
            if row.get("infer_end"):
                chunks[cid]["inf_end"] = float(row["infer_end"])
            if row.get("label"):
                chunks[cid]["label"] = row["label"]
            if row.get("score"):
                try:
                    chunks[cid]["score"] = float(row["score"])
                except:
                    pass

    # Convert to list for plotting, sorted by recording time
    lst = [{"chunk": k, **v} for k, v in chunks.items()]
    lst.sort(key=lambda x: x["rec_start"] if x["rec_start"] else 0.0)
    return lst

# ----------------------------------------------------------
# Plotting Function (Timeline): Recording + Queue + Inference
# ----------------------------------------------------------
def plot_rows(rows, out_png="timeline.png", title="Timeline"):

    # Extract all timestamps to normalize X-axis
    times = []
    for r in rows:
        for key in ("rec_start", "rec_end", "inf_start", "inf_end", "queued"):
            val = r.get(key)
            if val:
                times.append(val)

    if not times:
        print("❌ No timestamps found. Cannot plot.")
        return

    t0 = min(times)  # Start time offset

    # Y positions for each chunk
    ys = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.5)))

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (seconds, relative)", fontsize=12)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["chunk"] for r in rows])
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # -------------------------
    # Draw bars for each chunk
    # -------------------------
    for i, r in enumerate(rows):
        y = ys[i]

        # Blue bar = Recording interval
        if r.get("rec_start") and r.get("rec_end"):
            rs = r["rec_start"] - t0
            re = r["rec_end"] - t0
            ax.add_patch(
                patches.Rectangle((rs, y - 0.25), re - rs, 0.4, color='tab:blue', alpha=0.7)
            )
            ax.text(rs, y + 0.22, "record", fontsize=8, color='tab:blue')

        # Queue marker (Parallel only)
        if r.get("queued"):
            q = r["queued"] - t0
            ax.plot([q, q], [y - 0.4, y + 0.4], color='black', linewidth=1)
            ax.text(q, y - 0.35, "Q", fontsize=7, color='black')

        # Orange bar = Inference interval
        if r.get("inf_start") and r.get("inf_end"):
            is_ = r["inf_start"] - t0
            ie = r["inf_end"] - t0
            ax.add_patch(
                patches.Rectangle((is_, y - 0.25), ie - is_, 0.4, color='tab:orange', alpha=0.7)
            )
            ax.text(
                is_, y - 0.42,
                f"pred:{r.get('label','')[:8]} ({r.get('score','')})",
                fontsize=7, color='tab:orange'
            )

    ax.set_ylim(-1, len(rows))
    ax.set_xlim(0, max(times) - t0 + 0.2)
    plt.tight_layout()

    # Save plot
    plt.savefig(out_png, dpi=200)
    print(f"✅ Saved timeline plot → {out_png}")

# --------------------------------------
# Main Entrypoint
# --------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python plot_timeline.py <seq|par> <csv_filename> [output_png]")
        sys.exit(1)

    mode = sys.argv[1]          # seq or par
    csv_name = sys.argv[2]      # filename only

    # Auto-paths
    csv_path = os.path.join("Time_data", csv_name)

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    # Output folder
    out_folder = "KWS_waves"
    os.makedirs(out_folder, exist_ok=True)

    # Output PNG path
    out_png = os.path.join(out_folder,
                           sys.argv[3] if len(sys.argv) > 3 else f"{mode}_timeline.png")

    # Read CSV
    rows = read_seq(csv_path) if mode == "seq" else read_par(csv_path)

    # Plot & save
    plot_rows(rows, out_png=out_png, title=f"{mode.upper()} TIMELINE")

"""
---------------------------
How to Interpret the Plot
---------------------------

BLUE BAR  = Recording duration for each chunk.  
ORANGE BAR = Model inference time for that chunk.  
BLACK "Q" LINE = Moment the chunk was placed in inference queue (parallel only).  

Sequential Case (seq):
- Recording and inference NEVER overlap.
- The timeline looks like:
  
  Record → Infer  
  Record → Infer  
  ... (strict alternation)

Parallel Case (par):
- Recording of chunk N overlaps with inference of chunk N-1.
- This overlap visually proves parallelism.
- Queue markers show when chunks become available.

This visualization is ideal for:
✔ Medium articles  
✔ Demonstrating latency improvements  
✔ Understanding pipeline behavior  
✔ Debugging inference queue bottlenecks  
"""
