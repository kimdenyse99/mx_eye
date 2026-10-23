"""Background-subtraction motion/presence detection, for motion-triggered
recording (see RecordingConfig.motion_triggered_recording).

This answers "has something (an animal) entered the field of view," not
just "did anything change between two consecutive frames." A pure
frame-to-frame difference would stop registering an animal that has
entered and then holds still - which is exactly the case this needs to
handle well, since a resting/sleeping animal is still a subject worth
recording. Background subtraction instead maintains a running estimate of
the empty-scene background and flags any pixel that differs meaningfully
from it as foreground, so a stationary animal keeps reading as foreground
for as long as it remains in frame, not just during the moment it moved.

The background model is explicitly NOT a single frame captured once: it is
an adaptive running estimate (OpenCV's MOG2), continuously nudged toward
whatever the camera currently sees, at a slow, configurable rate. This
matters for two reasons:

- It absorbs genuine gradual change in the real empty scene (slow lighting
  drift through the day, a shifted shadow) without that being
  misinterpreted as sustained motion forever.
- Left unchecked, it would equally absorb an animal that stays still long
  enough, eventually reading the animal itself as "background" - which
  would defeat the point of this detector. See `learning_enabled` below:
  learning is meant to be paused for as long as something recognizes the
  animal as present (currently: this detector's own foreground signal,
  until a future RFID-based presence signal replaces or augments it -
  see the module docstring in service.py's motion-handling section).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionResult:
    foreground_percent: float
    is_motion: bool


class MotionDetector:
    """One instance per tracking_worker run; call update() once per frame.

    Not thread/process-shared - tracking_worker owns one instance and
    reports only the resulting percentage (a plain float) through shared
    stats, rather than sharing the background model itself across
    processes.
    """

    def __init__(
        self,
        threshold_percent: float,
        *,
        history: int = 500,
        var_threshold: float = 16.0,
        default_learning_rate: float = -1.0,
        warmup_frames: int = 15,
    ) -> None:
        """
        threshold_percent: foreground_percent at or above this counts as
            motion (see MotionResult.is_motion).
        history: number of frames MOG2 uses to build its background
            model; higher is slower to adapt but more stable.
        var_threshold: MOG2's own per-pixel variance threshold for
            classifying a pixel as foreground - this is a detection
            sensitivity knob, separate from threshold_percent (which
            instead gates the *aggregate* foreground area).
        default_learning_rate: passed to cv2.BackgroundSubtractorMOG2.apply
            as its learningRate when learning is enabled. -1 lets OpenCV
            choose an automatic rate based on `history`; a small positive
            value (e.g. 0.001-0.01) can be used for finer manual control.
        warmup_frames: a freshly-constructed MOG2 has no background model
            at all, so its very first apply() call has nothing to compare
            the frame against and reads as ~100% foreground - a false
            motion spike that has nothing to do with anything actually
            entering the scene. For this many initial frames, update()
            unconditionally feeds the background model (ignoring
            learning_enabled) and always reports is_motion=False, so the
            model gets a real chance to learn the empty scene before
            foreground_percent is trusted for anything. Set to 0 to
            disable warmup (not recommended at startup).
        """
        self._threshold_percent = threshold_percent
        self._default_learning_rate = default_learning_rate
        self._warmup_frames = warmup_frames
        self._frames_seen = 0
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self._total_pixels: int | None = None

    def update(self, frame, *, learning_enabled: bool) -> MotionResult:
        """frame: a decoded BGR image (same array tracking already uses -
        this does not re-decode or duplicate any capture-side work).
        learning_enabled: whether this frame may update the background
            model. Pass False while something (currently: this detector's
            own prior is_motion result; later: an RFID presence signal)
            considers an animal to be present, so a motionless animal is
            never silently absorbed into the background. Passing False
            does not disable foreground detection itself - the frame is
            still compared against the existing (frozen) background.
            Ignored during the initial warmup_frames (see __init__)."""
        if frame.ndim == 1:
            # 1-D encoded (e.g. MJPEG) frames are not supported directly;
            # the caller is expected to pass the same decoded image it
            # already uses for tracking.
            raise ValueError("MotionDetector requires a decoded image, not encoded bytes.")

        if self._total_pixels is None:
            self._total_pixels = frame.shape[0] * frame.shape[1]

        warming_up = self._frames_seen < self._warmup_frames
        self._frames_seen += 1

        if warming_up:
            # Force-feed the background model regardless of
            # learning_enabled, and never report motion from a frame the
            # model hasn't had a real chance to learn from yet. The mask
            # itself is intentionally discarded here - only the model's
            # internal state (updated as a side effect of apply())
            # matters during warmup, not this frame's foreground reading.
            self._subtractor.apply(frame, learningRate=-1.0)
            return MotionResult(foreground_percent=0.0, is_motion=False)

        learning_rate = self._default_learning_rate if learning_enabled else 0.0
        mask = self._subtractor.apply(frame, learningRate=learning_rate)

        # MOG2 output is 0 (background) or 255 (foreground) with
        # detectShadows=False; count nonzero directly rather than
        # thresholding again.
        foreground_pixels = int(np.count_nonzero(mask))
        foreground_percent = 100.0 * foreground_pixels / self._total_pixels

        return MotionResult(
            foreground_percent=foreground_percent,
            is_motion=foreground_percent >= self._threshold_percent,
        )