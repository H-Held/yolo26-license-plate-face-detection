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
    versions, model_sizes, map50 = [], [], []
    face_recall, face_precision = [], []
    plate_recall, plate_precision = [], []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            versions.append(row["version"])
            model_sizes.append(row["model_size"])
            map50.append(float(row["map50"]))
            face_recall.append(float(row["face_recall"]))
            face_precision.append(float(row["face_precision"]))
            plate_recall.append(float(row["license_plate_recall"]))
            plate_precision.append(float(row["license_plate_precision"]))

    labels = [f"{v}\n({s})" for v, s in zip(versions, model_sizes)]

    fig, (ax_recall, ax_prec) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150, sharey=True)

    ax_recall.plot(labels, map50, marker="o", color="#2563eb", label="mAP@50")
    ax_recall.plot(labels, face_recall, marker="o", color="#16a34a", label="face recall")
    ax_recall.plot(labels, plate_recall, marker="o", color="#ea580c", label="license-plate recall")
    ax_recall.set_title("Recall / mAP")

    ax_prec.plot(labels, face_precision, marker="o", color="#16a34a", label="face precision")
    ax_prec.plot(labels, plate_precision, marker="o", color="#ea580c", label="license-plate precision")
    ax_prec.set_title("Precision")

    for ax in (ax_recall, ax_prec):
        ax.set_ylim(0, 1)
        ax.set_xlabel("version (model size)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    ax_recall.set_ylabel("score")
    fig.suptitle("Model accuracy across releases (test split)")
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
