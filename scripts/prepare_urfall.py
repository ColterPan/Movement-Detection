"""Download and prepare the UR Fall Detection Dataset (camera 0) for ``motionsense eval``.

Dataset: B. Kwolek, M. Kepski, "Human fall detection on embedded platform using depth maps
and wireless accelerometer", 2014. http://fenix.ur.edu.pl/~mkepski/ds/uf.html
Check the dataset page for its terms of use before redistributing anything.

The cam0 videos show depth (left) and RGB (right) side by side; this script keeps the RGB half
and writes ``labels.csv`` with the fall onset time taken from the official per-frame labels
(label 0 = falling, 1 = lying, -1 = not lying).

Usage:
    python scripts/prepare_urfall.py --out %LOCALAPPDATA%/motionsense/datasets/urfall
    motionsense eval --labels %LOCALAPPDATA%/motionsense/datasets/urfall/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import cv2

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
FALLS = [f"fall-{i:02d}" for i in range(1, 31)]
ADLS = [f"adl-{i:02d}" for i in range(1, 41)]
FPS = 30.0


def download(url: str, dest: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS download: {url}")
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    # nosec B310: only https:// URLs reach this point (checked above)
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:  # nosec B310
        fh.write(resp.read())
    tmp.replace(dest)


def crop_rgb(src: Path, dest: Path) -> int:
    """Keep the right (RGB) half of each frame. Returns the number of frames written."""
    if dest.exists() and dest.stat().st_size > 0:
        return -1
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    writer = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = frame[:, w // 2 :]
        if writer is None:
            writer = cv2.VideoWriter(
                str(dest), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w - w // 2, h)
            )
        writer.write(rgb)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    return n


def fall_onsets(labels_csv: Path) -> dict[str, float]:
    """First frame labelled 0 (falling) per sequence, in seconds."""
    onsets: dict[str, float] = {}
    with labels_csv.open(newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            seq, frame, label = row[0], int(row[1]), int(row[2])
            if label == 0 and seq not in onsets:
                onsets[seq] = (frame - 1) / FPS
    return onsets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--falls", type=int, default=30, help="how many fall sequences (max 30)")
    parser.add_argument(
        "--adls", type=int, default=40, help="how many daily-activity sequences (max 40)"
    )
    args = parser.parse_args()

    raw = args.out / "raw"
    clips = args.out / "clips"
    raw.mkdir(parents=True, exist_ok=True)
    clips.mkdir(parents=True, exist_ok=True)

    labels_csv = raw / "urfall-cam0-falls.csv"
    download(BASE + "urfall-cam0-falls.csv", labels_csv)
    onsets = fall_onsets(labels_csv)

    rows: list[tuple[str, int, str]] = []
    for name in FALLS[: args.falls] + ADLS[: args.adls]:
        src = raw / f"{name}-cam0.mp4"
        try:
            download(f"{BASE}{name}-cam0.mp4", src)
        except OSError as exc:
            print(f"skip {name}: {exc}", file=sys.stderr)
            continue
        crop_rgb(src, clips / f"{name}.mp4")
        is_fall = name.startswith("fall")
        onset = onsets.get(name)
        rows.append(
            (
                f"clips/{name}.mp4",
                int(is_fall),
                f"{onset:.3f}" if is_fall and onset is not None else "",
            )
        )
        print(f"ready {name} fall={is_fall} onset={onset}")

    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video", "has_fall", "fall_time_s"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} clips -> {args.out / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
