"""Match visual_task's per-trial gaze-correct sidecar CSVs (see
trial_events.py) against one mx-eye recording's frames.csv, producing a
single combined marker CSV: one row per matched trial, with the exact
video frame index and time (in seconds into the video) the gaze-correct
moment falls on.

This does not modify the recording in any way - it is a read-only,
non-destructive pass that produces a new, separate marker file. A future
mx-eye GUI video view can read this file directly to show markers on a
timeline; for now it's also just a plain CSV you can open in a
spreadsheet.

How the matching works
-----------------------
Every sidecar CSV has a CLOCK_REALTIME timestamp_ns from visual_task's own
machine. Every row of mx-eye's frames.csv has both an acquisition_ns
(CLOCK_REALTIME, the same clock base - directly comparable ONLY because
chrony keeps the two machines' clocks synchronized; see
AGENTS.md "Clock synchronization" in the mx_eye repo) and an
acquisition_monotonic_ns (the video file's own internal frame timing
base, NOT directly comparable to timestamp_ns on its own). For each
sidecar event, this script finds the frames.csv row whose acquisition_ns
is closest to the event's timestamp_ns, and reports that row's
video_index directly - no interpolation, no estimation beyond "nearest
frame by real-clock timestamp".

Usage
-----
    python make_video_markers.py \\
        --trial-data-dir /path/to/visual_task/data/<animal>/<date>/<session>/visual_tracing \\
        --frames-csv /path/to/mx_eye/recordings/<recording-folder>/frames.csv \\
        --output markers.csv \\
        [--fps 30] [--max-skew-ms 200]

--fps is only used to additionally report a human-readable
video_seconds column (video_index / fps); if omitted, video_seconds is
left blank and only the frame index is reported (which is always exact,
regardless of fps).

--max-skew-ms (default 200) is a sanity check: if the closest frame's
acquisition_ns is further than this from the event's timestamp_ns, the
match is still reported, but flagged in the skew_ms column so an
unusually large value (e.g. due to clock sync drift, or the event
genuinely falling outside the recording's time range) is easy to spot
rather than silently trusted.
"""

from __future__ import annotations

import argparse
import csv
from bisect import bisect_left
from pathlib import Path


def load_frames_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_trial_sidecars(trial_data_dir: Path, event_name: str) -> list[dict[str, str]]:
    """Read <event_name>.csv from the given directory - one shared file
    per event type per session/stage, with one row per trial that fired
    that event (see trial_events.py's _write_trial_event: the filename
    deliberately excludes trial_id, so every trial in a session
    accumulates into this one file via DataLogger's own append-mode CSV
    writing)."""
    path = trial_data_dir / f"{event_name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_nearest_frame(
    frame_acquisition_ns: list[int], target_ns: int
) -> tuple[int, int]:
    """Returns (index_into_frame_acquisition_ns, that_frame's_acquisition_ns)
    for the frame whose acquisition_ns is closest to target_ns.
    frame_acquisition_ns must already be sorted ascending (frames.csv rows
    are written in acquisition order, so this holds as-is)."""
    pos = bisect_left(frame_acquisition_ns, target_ns)
    candidates = []
    if pos > 0:
        candidates.append(pos - 1)
    if pos < len(frame_acquisition_ns):
        candidates.append(pos)
    best = min(candidates, key=lambda i: abs(frame_acquisition_ns[i] - target_ns))
    return best, frame_acquisition_ns[best]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial-data-dir", required=True, type=Path)
    parser.add_argument("--frames-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-skew-ms", type=float, default=200.0)
    parser.add_argument(
        "--event",
        default="gaze_correct",
        help="Which sidecar event to match: the <event> in trial_<id>_<event>.csv "
        "(e.g. gaze_correct, or target_shown for pre-calibration use - see "
        "trial_events.py's module docstring). Default: gaze_correct.",
    )
    args = parser.parse_args()

    frame_rows = load_frames_csv(args.frames_csv)
    if not frame_rows:
        raise SystemExit(f"No rows found in {args.frames_csv}")
    frame_acquisition_ns = [int(row["acquisition_ns"]) for row in frame_rows]

    sidecar_rows = load_trial_sidecars(args.trial_data_dir, args.event)
    if not sidecar_rows:
        raise SystemExit(
            f"No rows found in {args.trial_data_dir / f'{args.event}.csv'} "
            f"(file missing or empty)"
        )

    output_rows = []
    for sidecar in sidecar_rows:
        event_ns = int(sidecar["timestamp_ns"])
        frame_pos, frame_ns = find_nearest_frame(frame_acquisition_ns, event_ns)
        video_index = int(frame_rows[frame_pos]["video_index"])
        skew_ms = (frame_ns - event_ns) / 1e6

        video_seconds = video_index / args.fps if args.fps else ""

        output_rows.append(
            {
                "trial_id": sidecar["trial_id"],
                "event": sidecar["event"],
                "timestamp_ns": event_ns,
                "video_index": video_index,
                "video_seconds": video_seconds,
                "skew_ms": round(skew_ms, 2),
                "target_x": sidecar.get("target_x", ""),
                "target_y": sidecar.get("target_y", ""),
            }
        )

        if abs(skew_ms) > args.max_skew_ms:
            print(
                f"warning: trial {sidecar['trial_id']} matched frame is "
                f"{skew_ms:.1f} ms away from the event timestamp "
                f"(> --max-skew-ms={args.max_skew_ms}) - check clock sync "
                f"(chrony) between the two machines, or whether this event "
                f"genuinely falls outside the recording's time range."
            )

    fieldnames = [
        "trial_id",
        "event",
        "timestamp_ns",
        "video_index",
        "video_seconds",
        "skew_ms",
        "target_x",
        "target_y",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} marker(s) to {args.output}")


if __name__ == "__main__":
    main()
