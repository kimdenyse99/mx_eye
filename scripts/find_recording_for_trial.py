"""Given several mx-eye recordings (each with its own frames.csv) and a
visual_task sidecar event file (see trial_events.py), determine which
recording actually covers each trial's event timestamp - useful when
recording was stopped and restarted multiple times during what was really
one continuous visual_task session, so there's no reliable 1:1 mapping
between "a session" and "a recording".

A recording's coverage is simply the real-clock time span of its own
frames.csv: from its first row's acquisition_ns to its last row's
acquisition_ns. A trial's event "belongs to" whichever recording's span
contains its timestamp_ns (with a small configurable tolerance for an
event that falls just outside a recording's first/last frame, since a
recording's actual frame rate means its very first or last frame is
rarely exactly aligned with a trial's logged moment).

This is read-only and purely informational: it doesn't move, rename, or
alter any files. Its job is to add one more column (recording) to the
sidecar data, which make_video_markers.py and join_calibration_dataset.py
can then use directly (see their --recordings-root option) instead of
needing a single, manually-chosen --frames-csv/--tracking-csv path.

Usage
-----
    python find_recording_for_trial.py \\
        --recordings-root /path/to/mx_eye/recordings \\
        --trial-data-dir /path/to/visual_task/data/<animal>/<date>/<session>/<stage> \\
        --event target_shown \\
        --output trial_to_recording.csv \\
        [--tolerance-ms 500]

--recordings-root is the folder CONTAINING multiple recording folders
(each with its own frames.csv), not one specific recording folder itself
- e.g. mx_eye's configured recording.directory, not one timestamped
subfolder of it.

--tolerance-ms (default 500) is how far outside a recording's own first/
last frame timestamp an event may still fall and be considered covered by
that recording - a generous default given a video's frame interval is
typically well under 100ms; events further outside than this are reported
as unmatched rather than silently assigned to the nearest recording
regardless of how far off it actually is.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def recording_span(frames_csv: Path) -> tuple[int, int] | None:
    """Returns (first_acquisition_ns, last_acquisition_ns) for one
    recording's frames.csv, or None if the file is missing/empty."""
    if not frames_csv.exists():
        return None
    with frames_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    ns_values = [int(row["acquisition_ns"]) for row in rows]
    return min(ns_values), max(ns_values)


def find_recordings(recordings_root: Path) -> dict[Path, tuple[int, int]]:
    """Scan recordings_root for subdirectories containing a frames.csv,
    returning {recording_dir: (first_ns, last_ns)} for each one found."""
    spans: dict[Path, tuple[int, int]] = {}
    for entry in sorted(recordings_root.iterdir()):
        if not entry.is_dir():
            continue
        frames_csv = entry / "frames.csv"
        span = recording_span(frames_csv)
        if span is not None:
            spans[entry] = span
    return spans


def load_trial_sidecars(trial_data_dir: Path, event_name: str) -> list[dict[str, str]]:
    path = trial_data_dir / f"{event_name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recordings-root", required=True, type=Path)
    parser.add_argument("--trial-data-dir", required=True, type=Path)
    parser.add_argument("--event", default="target_shown")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance-ms", type=float, default=500.0)
    args = parser.parse_args()

    spans = find_recordings(args.recordings_root)
    if not spans:
        raise SystemExit(
            f"No recording folders with a frames.csv found under {args.recordings_root}"
        )
    print(f"Found {len(spans)} recording(s):")
    for recording_dir, (first_ns, last_ns) in spans.items():
        duration_s = (last_ns - first_ns) / 1e9
        print(f"  {recording_dir.name}: {duration_s:.1f}s of footage")

    sidecar_rows = load_trial_sidecars(args.trial_data_dir, args.event)
    if not sidecar_rows:
        raise SystemExit(
            f"No rows found in {args.trial_data_dir / f'{args.event}.csv'} "
            f"(file missing or empty)"
        )

    tolerance_ns = args.tolerance_ms * 1e6
    output_rows = []
    unmatched = 0
    for sidecar in sidecar_rows:
        event_ns = int(sidecar["timestamp_ns"])
        matched_dir = None
        for recording_dir, (first_ns, last_ns) in spans.items():
            if first_ns - tolerance_ns <= event_ns <= last_ns + tolerance_ns:
                matched_dir = recording_dir
                break

        output_rows.append(
            {
                "trial_id": sidecar["trial_id"],
                "event": sidecar["event"],
                "timestamp_ns": event_ns,
                "target_x": sidecar.get("target_x", ""),
                "target_y": sidecar.get("target_y", ""),
                "recording": matched_dir.name if matched_dir else "",
                "recording_path": str(matched_dir) if matched_dir else "",
            }
        )
        if matched_dir is None:
            unmatched += 1
            print(
                f"warning: trial {sidecar['trial_id']}'s event at {event_ns} "
                f"doesn't fall within any recording's span (even with "
                f"--tolerance-ms={args.tolerance_ms}) - this trial may have "
                f"happened while recording was stopped."
            )

    fieldnames = [
        "trial_id",
        "event",
        "timestamp_ns",
        "target_x",
        "target_y",
        "recording",
        "recording_path",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nWrote {len(output_rows)} row(s) to {args.output}")
    if unmatched:
        print(f"{unmatched} trial(s) could not be matched to any recording.")


if __name__ == "__main__":
    main()
