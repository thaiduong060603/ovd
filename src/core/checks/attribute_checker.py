"""
attribute_checker.py — Deterministic image-based attribute checking.

Used AFTER NanoOWL detection/tracking to filter tracks by visual attributes
(color, etc.) before the Rule Engine evaluates them.

The tracker continues tracking ALL objects internally.
Only the list passed to RuleEngine is filtered.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# HSV color ranges
# Each entry: list of (lower_hsv, upper_hsv) pairs — multiple ranges cover
# hue-wrap-around (e.g. red spans 0-10 and 160-180).
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    "yellow": [
        (np.array([18,  80,  80]), np.array([35, 255, 255])),
    ],
    "orange": [
        (np.array([10,  80,  80]), np.array([18, 255, 255])),
    ],
    "red": [
        (np.array([0,   80,  80]), np.array([10, 255, 255])),
        (np.array([160, 80,  80]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([100, 80,  80]), np.array([130, 255, 255])),
    ],
    "green": [
        (np.array([40,  60,  60]), np.array([80, 255, 255])),
    ],
    "white": [
        (np.array([0,    0, 180]), np.array([180, 40, 255])),
    ],
    "black": [
        (np.array([0,    0,   0]), np.array([180, 255, 50])),
    ],
}


class AttributeChecker:
    """
    Post-filter: verifies track attributes (e.g. color) on cropped bbox regions.

    Usage:
        checker = AttributeChecker()
        eval_tracks = checker.filter_tracks(frame, tracks, attribute_checks)
        # Pass eval_tracks (not tracks) to RuleEngineV1.evaluate()
    """

    def __init__(self, min_color_ratio: float = 0.12):
        """
        Args:
            min_color_ratio: fraction of bbox pixels that must match the
                             target color to pass.  0.12 = 12% (permissive,
                             accounts for partial visibility / lighting).
        """
        self.min_color_ratio = min_color_ratio

    # ── Public API ────────────────────────────────────────────────────────────

    def filter_tracks(
        self,
        frame:   np.ndarray,
        tracks:  List,                      # List[Track]
        checks:  List[Dict[str, Any]],
    ) -> List:
        """
        Return only the tracks that pass ALL applicable attribute checks.

        Args:
            frame:   current BGR frame
            tracks:  ByteTracker output
            checks:  list of check dicts from GeneratedRule.attribute_checks, e.g.
                     [{"class_name": "a photo of helmet",
                       "attribute": "color", "value": "yellow"}]
        """
        if not checks:
            return tracks
        return [t for t in tracks if self._passes_all(frame, t, checks)]

    def check_color(
        self,
        frame:        np.ndarray,
        bbox:         np.ndarray,
        target_color: str,
    ) -> bool:
        """
        Check whether the dominant color inside bbox matches target_color.

        Args:
            frame:        BGR numpy array
            bbox:         [x1, y1, x2, y2] pixel coordinates
            target_color: color name key (see _COLOR_RANGES)

        Returns:
            True if ≥ min_color_ratio of pixels match.
        """
        key = target_color.lower().strip()
        if key not in _COLOR_RANGES:
            return True  # unknown color → pass by default

        crop = self._safe_crop(frame, bbox)
        if crop is None or crop.size == 0:
            return False

        hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        if total == 0:
            return False

        mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)
        for lower, upper in _COLOR_RANGES[key]:
            mask |= cv2.inRange(hsv, lower, upper)

        return (np.count_nonzero(mask) / total) >= self.min_color_ratio

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _passes_all(
        self,
        frame:  np.ndarray,
        track,
        checks: List[Dict[str, Any]],
    ) -> bool:
        for check in checks:
            if not self._applies_to_track(check.get("class_name", ""), track.class_name):
                continue  # this check targets a different class

            attr  = check.get("attribute", "").lower()
            value = check.get("value", "")

            if attr == "color":
                if not self.check_color(frame, track.bbox, value):
                    return False
            # Extend here for future attribute types (size, shape, …)

        return True

    @staticmethod
    def _applies_to_track(check_class: str, track_class: str) -> bool:
        """
        Match ignoring the 'a photo of ' prefix so that both
        'helmet' and 'a photo of helmet' match track class 'a photo of helmet'.
        """
        def _strip(s: str) -> str:
            return s.lower().replace("a photo of ", "").strip()

        c = _strip(check_class)
        t = _strip(track_class)
        return c in t or t in c

    @staticmethod
    def _safe_crop(
        frame: np.ndarray,
        bbox:  np.ndarray,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
