"""
nanoowl_detector.py — NanoOWL detector wrapper for OVD Watchdog pipeline.

Wraps nanoowl.OwlPredictor to match the YOLOWorldDetector interface:
    detector.detect(frame_bgr, prompts, frame_id) -> List[Detection]

Requirements:
    pip install nanoowl
    TensorRT engine: ~/owlvit_image_encoder.engine

Build engine once on Jetson:
    python3 -m nanoowl.build_image_encoder_engine \
        ~/owlvit_image_encoder.engine \
        --model_name google/owlvit-base-patch32
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.models.detection import Detection


_DEFAULT_ENGINE = str(Path.home() / "owlvit_image_encoder.engine")


class NanoOWLDetector:
    """
    NanoOWL (OWL-ViT + TensorRT) detector.

    Drop-in replacement for YOLOWorldDetector within the OVD pipeline.
    Text encodings are cached and only recomputed when prompts change.
    """

    def __init__(
        self,
        model_name:      str   = "google/owlvit-base-patch32",
        engine_path:     str   = _DEFAULT_ENGINE,
        box_threshold:   float = 0.10,
        text_threshold:  float = 0.10,   # kept for interface parity; NanoOWL uses single threshold
        device:          str   = "cuda",
        nms_iou:         float = 0.45,
    ):
        self.model_name     = model_name
        self.engine_path    = engine_path
        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold
        self.device         = device
        self.nms_iou        = nms_iou

        self._predictor      = None
        self._last_prompts:  List[str] = []
        self._text_encodings = None

        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self) -> None:
        from nanoowl.owl_predictor import OwlPredictor
        print(f"[NanoOWL] Loading predictor …")
        print(f"[NanoOWL]   model  : {self.model_name}")
        print(f"[NanoOWL]   engine : {self.engine_path}")
        self._predictor = OwlPredictor(
            self.model_name,
            image_encoder_engine=self.engine_path,
        )
        print("[NanoOWL] Predictor ready.")

    # ── Text encoding cache ───────────────────────────────────────────────────

    def _get_text_encodings(self, prompts: List[str]):
        """Re-encode only when prompts differ from last call."""
        if prompts != self._last_prompts:
            print(f"[NanoOWL] Encoding prompts: {prompts}")
            self._text_encodings = self._predictor.encode_text(prompts)
            self._last_prompts   = list(prompts)
        return self._text_encodings

    # ── Main interface ────────────────────────────────────────────────────────

    def detect(
        self,
        image:    np.ndarray,
        prompts:  List[str],
        frame_id: int = 0,
    ) -> List[Detection]:
        """
        Run NanoOWL inference on a BGR frame.

        Args:
            image:    BGR numpy array (from cv2)
            prompts:  text prompts, e.g. ["a photo of person", "a photo of helmet"]
            frame_id: frame index (metadata only)

        Returns:
            List[Detection] compatible with ByteTracker.update()
        """
        from PIL import Image as PILImage

        if self._predictor is None or not prompts:
            return []

        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        text_enc = self._get_text_encodings(prompts)

        output = self._predictor.predict(
            image          = pil_img,
            text           = prompts,
            text_encodings = text_enc,
            threshold      = self.box_threshold,
            pad_square     = True,
        )

        if len(output.boxes) == 0:
            return []

        boxes_raw  = output.boxes.detach().cpu().numpy()
        scores_raw = output.scores.detach().cpu().numpy()
        label_idxs = output.labels.detach().cpu().numpy()

        boxes_raw, scores_raw, label_idxs = self._nms(boxes_raw, scores_raw, label_idxs)

        h, w = image.shape[:2]
        detections: List[Detection] = []
        for box, score, idx in zip(boxes_raw, scores_raw, label_idxs):
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w,  int(box[2]))
            y2 = min(h,  int(box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            label = prompts[int(idx)]
            detections.append(Detection(
                bbox        = np.array([x1, y1, x2, y2], dtype=np.float32),
                confidence  = float(score),
                class_name  = label,
                prompt_used = label,
                frame_id    = frame_id,
            ))

        return detections

    def detect_with_interval(
        self,
        image:              np.ndarray,
        prompts:            List[str],
        frame_id:           int,
        detection_interval: int,
    ) -> Tuple[bool, List[Detection]]:
        """Run detection only every N frames (interface parity with YOLOWorldDetector)."""
        if frame_id % detection_interval != 0:
            return False, []
        return True, self.detect(image, prompts, frame_id)

    def warmup(self, image_size: Tuple[int, int] = (640, 640)) -> None:
        """Warm up TensorRT kernels with a dummy frame."""
        dummy = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        self.detect(dummy, ["a photo of person"], frame_id=0)
        print("[NanoOWL] Warmup complete.")

    # ── NMS ───────────────────────────────────────────────────────────────────

    def _nms(
        self,
        boxes:  np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return boxes, scores, labels
        try:
            boxes_xywh = [
                [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
                for b in boxes
            ]
            idxs = cv2.dnn.NMSBoxes(
                boxes_xywh,
                scores.tolist(),
                score_threshold=0.0,
                nms_threshold=self.nms_iou,
            )
            if len(idxs) == 0:
                return np.array([]), np.array([]), np.array([])
            idxs = idxs.flatten()
            return boxes[idxs], scores[idxs], labels[idxs]
        except Exception:
            return boxes, scores, labels
