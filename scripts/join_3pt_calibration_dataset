"""Join visual_task's per-trial sidecar event CSVs (see trial_events.py)
against BOTH an mx-eye recording's frames.csv and its tracking.csv at
once, producing a single combined CSV: one row per trial, with the known
target position, the matched camera frame index/seconds, and the raw
pupil-minus-CR gaze reading mx-eye measured at that same real-world
moment - everything needed to build a calibration dataset (e.g. for
visual_3pt_calibration's three distinct target positions) in one place,
without needing to cross-reference two separate outputs by trial_id
yourself.

This supersedes running make_video_markers.py and
join_gaze_calibration_data.py separately when you want both kinds of
information together; those two scripts still exist independently for
when you only need one or the other.

Matching method
-----------------
For each sidecar event's timestamp_ns, independently find the frames.csv
row and the tracking.csv row whose own acquisition_ns is closest (all
three are the same raw CLOCK_REALTIME clock base - directly comparable
only because chrony keeps the two machines' clocks synchronized; see
AGENTS.md "Clock synchronization" in the mx_eye repo). No interpolation
beyond "nearest sample by real-clock timestamp", and the two matches
(video frame, tracking sample) are found independently of each other -
they are not required to be the exact same underlying camera frame,
since frames.csv and tracking.csv are not guaranteed to have a row for
every single frame in lockstep (tracking can skip frames; recording can
too, independently).

Usage
-----
    python join_calibration_dataset.py \\
        --trial-data-dir /path/to/visual_task/data/<animal>/<date>/<session>/<stage> \\
        --frames-csv /path/to/mx_eye/recordings/<recording-folder>/frames.csv \\
        --tracking-csv /path/to/mx_eye/recordings/<recording-folder>/tracking.csv \\
        --output calibration_dataset.csv \\
        --event target_shown \\
        [--fps 30] [--max-skew-ms 200] [--require-valid]

--event selects which sidecar to read (trial_<id>_<event>.csv); default
target_shown, the right choice for from-scratch, no-prior-calibration
data collection (see trial_events.py's module docstring for why).

--require-valid drops any trial whose matched tracking.csv row's flags do
not include VALID, or whose raw x/y are not finite (NaN) - i.e. mx-eye
itself failed to get a usable pupil/CR reading on that sample. Off by
default so you can see everything, including failed samples, during
initial inspection.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from bisect import bisect_left
from pathlib import Path


def solve_offset(
    raw_x: float,
    raw_y: float,
    target_x: float,
    target_y: float,
    screen_width: float,
    screen_height: float,
    gain_x: float,
    gain_y: float,
    rotation_deg: float,
) -> tuple[float, float]:
    """Solve for the offset_x/offset_y a mx_eye CalibrationProfile would
    need so that applying its transform (see calibration.py's
    CalibrationProfile.apply) to (raw_x, raw_y) lands exactly on
    (target_x, target_y), given the other already-fixed transform
    parameters (gain, rotation, screen size).

    This mirrors CalibrationProfile.apply's own formula exactly (same
    center-relative, rotate-then-offset order), just solved in reverse
    for offset_x/offset_y instead of forward for screen_x/screen_y:

        dx = raw_x * gain_x
        dy = raw_y * gain_y
        screen_x = center_x + cos(theta)*dx - sin(theta)*dy + offset_x
        screen_y = center_y + sin(theta)*dx + cos(theta)*dy + offset_y

    With only one (raw, target) pair, this solves offset_x/offset_y
    exactly for THIS one point - it says nothing about gain_x/gain_y or
    rotation_deg themselves, which this function takes as given
    (whatever is already configured, e.g. 90 degrees for a rotated
    physical mount) rather than also fitting. Determining gain
    independently needs at least a second, different known point (see
    visual_3pt_calibration's three positions).
    """
    center_x = screen_width / 2
    center_y = screen_height / 2
    dx = raw_x * gain_x
    dy = raw_y * gain_y
    theta = math.radians(rotation_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    offset_x = target_x - center_x - (cosine * dx - sine * dy)
    offset_y = target_y - center_y - (sine * dx + cosine * dy)
    return offset_x, offset_y

# Must match mx_eye_protocol.data_frame.TrackingFlags.VALID's bit value.
# Duplicated here (rather than importing mx_eye_protocol) so this script
# has zero dependency on either the mx_eye or visual_task codebases and
# can be run from anywhere with only the standard library.
TRACKING_FLAG_VALID = 1


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_trial_sidecars(trial_data_dir: Path, event_name: str) -> list[dict[str, str]]:
    """Find every trial_<id>_<event_name>.csv in the given directory and
    return their rows, one dict per matched trial (each sidecar is a
    one-data-row CSV, per trial_events.py)."""
    pattern = re.compile(rf"^trial_(\d+)_{re.escape(event_name)}\.csv$")
    rows = []
    for entry in sorted(trial_data_dir.iterdir()):
        match = pattern.match(entry.name)
        if not match:
            continue
        with entry.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
        if row is None:
            continue
        rows.append(row)
    return rows


def find_nearest(sorted_ns: list[int], target_ns: int) -> int:
    pos = bisect_left(sorted_ns, target_ns)
    candidates = []
    if pos > 0:
        candidates.append(pos - 1)
    if pos < len(sorted_ns):
        candidates.append(pos)
    return min(candidates, key=lambda i: abs(sorted_ns[i] - target_ns))


def is_valid_sample(row: dict[str, str]) -> bool:
    try:
        flags = int(row["flags"])
        x, y = float(row["x"]), float(row["y"])
    except (KeyError, ValueError):
        return False
    return bool(flags & TRACKING_FLAG_VALID) and math.isfinite(x) and math.isfinite(y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trial-data-dir", required=True, type=Path)
    parser.add_argument("--frames-csv", required=True, type=Path)
    parser.add_argument("--tracking-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--event", default="target_shown")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-skew-ms", type=float, default=200.0)
    parser.add_argument("--require-valid", action="store_true")
    parser.add_argument(
        "--screen-width", type=float, default=1024.0,
        help="Must match the CalibrationProfile's own screen_width (mx-eye default: 1024).",
    )
    parser.add_argument(
        "--screen-height", type=float, default=600.0,
        help="Must match the CalibrationProfile's own screen_height (mx-eye default: 600).",
    )
    parser.add_argument(
        "--gain-x", type=float, default=1.0,
        help="This animal's current gain_x from mx-eye's Calibration window (default: 1.0, unset).",
    )
    parser.add_argument(
        "--gain-y", type=float, default=1.0,
        help="This animal's current gain_y from mx-eye's Calibration window (default: 1.0, unset).",
    )
    parser.add_argument(
        "--rotation-deg", type=float, default=0.0,
        help="This animal's current rotation_deg from mx-eye's Calibration window (default: 0.0).",
    )
    args = parser.parse_args()

    frame_rows = load_csv(args.frames_csv)
    if not frame_rows:
        raise SystemExit(f"No rows found in {args.frames_csv}")
    frame_ns = [int(row["acquisition_ns"]) for row in frame_rows]

    tracking_rows = load_csv(args.tracking_csv)
    if not tracking_rows:
        raise SystemExit(f"No rows found in {args.tracking_csv}")
    tracking_ns = [int(row["acquisition_ns"]) for row in tracking_rows]

    sidecar_rows = load_trial_sidecars(args.trial_data_dir, args.event)
    if not sidecar_rows:
        raise SystemExit(
            f"No trial_*_{args.event}.csv files found in {args.trial_data_dir}"
        )

    output_rows = []
    skipped_invalid = 0
    offsets_by_target: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for sidecar in sidecar_rows:
        event_ns = int(sidecar["timestamp_ns"])

        frame_idx = find_nearest(frame_ns, event_ns)
        video_index = int(frame_rows[frame_idx]["video_index"])
        frame_skew_ms = (frame_ns[frame_idx] - event_ns) / 1e6
        video_seconds = video_index / args.fps if args.fps else ""

        track_idx = find_nearest(tracking_ns, event_ns)
        tracking_row = tracking_rows[track_idx]
        tracking_skew_ms = (tracking_ns[track_idx] - event_ns) / 1e6

        valid = is_valid_sample(tracking_row)
        if args.require_valid and not valid:
            skipped_invalid += 1
            continue

        target_x_str = sidecar.get("target_x", "")
        target_y_str = sidecar.get("target_y", "")
        raw_x_str = tracking_row.get("x", "")
        raw_y_str = tracking_row.get("y", "")

        offset_x = offset_y = ""
        try:
            offset_x, offset_y = solve_offset(
                float(raw_x_str),
                float(raw_y_str),
                float(target_x_str),
                float(target_y_str),
                args.screen_width,
                args.screen_height,
                args.gain_x,
                args.gain_y,
                args.rotation_deg,
            )
            if not (math.isfinite(offset_x) and math.isfinite(offset_y)):
                offset_x = offset_y = ""
            else:
                offsets_by_target.setdefault(
                    (target_x_str, target_y_str), []
                ).append((offset_x, offset_y))
                offset_x, offset_y = round(offset_x, 3), round(offset_y, 3)
        except (ValueError, TypeError):
            # target_x/y or raw x/y missing/empty/NaN for this row - leave
            # offset blank rather than guessing or crashing the whole run.
            pass

        output_rows.append(
            {
                "trial_id": sidecar["trial_id"],
                "event": sidecar["event"],
                "timestamp_ns": event_ns,
                "target_x": target_x_str,
                "target_y": target_y_str,
                "video_index": video_index,
                "video_seconds": video_seconds,
                "frame_skew_ms": round(frame_skew_ms, 2),
                "raw_x": raw_x_str,
                "raw_y": raw_y_str,
                "pupil_x": tracking_row.get("pupil_x", ""),
                "pupil_y": tracking_row.get("pupil_y", ""),
                "cr_x": tracking_row.get("cr_x", ""),
                "cr_y": tracking_row.get("cr_y", ""),
                "sample_valid": valid,
                "tracking_skew_ms": round(tracking_skew_ms, 2),
                "offset_x": offset_x,
                "offset_y": offset_y,
            }
        )

        for label, skew_ms in (
            ("frame", frame_skew_ms),
            ("tracking", tracking_skew_ms),
        ):
            if abs(skew_ms) > args.max_skew_ms:
                print(
                    f"warning: trial {sidecar['trial_id']}'s matched {label} sample "
                    f"is {skew_ms:.1f} ms away from the event timestamp "
                    f"(> --max-skew-ms={args.max_skew_ms}) - check clock sync "
                    f"(chrony) between the two machines, or whether this event "
                    f"genuinely falls outside the recording's time range."
                )

    fieldnames = [
        "trial_id",
        "event",
        "timestamp_ns",
        "target_x",
        "target_y",
        "video_index",
        "video_seconds",
        "frame_skew_ms",
        "raw_x",
        "raw_y",
        "pupil_x",
        "pupil_y",
        "cr_x",
        "cr_y",
        "sample_valid",
        "tracking_skew_ms",
        "offset_x",
        "offset_y",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} row(s) to {args.output}")
    if args.require_valid and skipped_invalid:
        print(f"Skipped {skipped_invalid} trial(s) with no valid tracking sample nearby.")

    if offsets_by_target:
        print(
            "\nOffset solved per row above assumes gain_x={}, gain_y={}, "
            "rotation_deg={}, screen {}x{} (pass --gain-x/--gain-y/"
            "--rotation-deg/--screen-width/--screen-height to match this "
            "animal's actual CalibrationProfile if different from these "
            "defaults).".format(
                args.gain_x,
                args.gain_y,
                args.rotation_deg,
                args.screen_width,
                args.screen_height,
            )
        )
        print("\nAverage offset_x/offset_y per distinct target position:")
        for (target_x_str, target_y_str), values in offsets_by_target.items():
            mean_x = sum(v[0] for v in values) / len(values)
            mean_y = sum(v[1] for v in values) / len(values)
            print(
                f"  target ({target_x_str}, {target_y_str}), "
                f"{len(values)} trial(s): "
                f"offset_x={mean_x:.3f}, offset_y={mean_y:.3f}"
            )


if __name__ == "__main__":
    main()
