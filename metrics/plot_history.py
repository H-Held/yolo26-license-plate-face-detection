"""Render metrics_history.csv as a PNG chart (accuracy across releases).

Usage: python metrics/plot_history.py
Reads metrics/metrics_history.csv, writes metrics/accuracy_history.png.
No external services — local matplotlib rendering only.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "metrics_history.csv")
OUT_PATH = os.path.join(HERE, "accuracy_history.png")


def main():
    versions, map50, face_recall, plate_recall = [], [], [], []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            versions.append(row["version"])
            map50.append(float(row["map50"]))
            face_recall.append(float(row["face_recall"]))
            plate_recall.append(float(row["license_plate_recall"]))

    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=150)
    ax.plot(versions, map50, marker="o", color="#2563eb", label="mAP@50")
    ax.plot(versions, face_recall, marker="o", color="#16a34a", label="face recall")
    ax.plot(versions, plate_recall, marker="o", color="#ea580c", label="license-plate recall")

    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Model accuracy across releases (test split)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
