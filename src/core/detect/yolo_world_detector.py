
import numpy as np
import cv2
from typing import List, Tuple
from pathlib import Path

from src.models.detection import Detection


def _parse_prompts(prompt_str: str) -> List[str]:
    """
    'person . helmet . hard hat . safety helmet .'
    → ['person', 'helmet', 'hard hat', 'safety helmet']
    """
    parts = [p.strip().rstrip(".").strip() for p in prompt_str.split(".")]
    return [p for p in parts if p]


class YOLOWorldDetector:
    def __init__(
        self,
        model_path: str = "models/yolov8s-world.pt",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25, 
        device: str = "cuda",
        nms_iou: float = 0.45,
    ):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold   
        self.device = device
        self.nms_iou = nms_iou

        self._last_classes: List[str] = []

        print(f"Loading YOLO-World on {device}...")
        self._load_model(model_path)
        print(f"✓ YOLO-World loaded: {Path(model_path).name}")

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics not install. Run: pip install ultralytics"
            )

        path = Path(model_path)


        if not path.exists():
            model_name = path.name
            print(f"  Model {model_name} not found, downloading...")
            self._model = YOLO(model_name)  
            path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = Path(model_name)
            if downloaded.exists() and not path.exists():
                downloaded.rename(path)
                print(f"  ✓ Saved to {path}")
        else:
            self._model = YOLO(str(path))

        # Set device
        self._model.to(self.device)
        try:
            if hasattr(self._model.model, 'txt_feats'):
                pass  
            self._model.model.float()  # force FP32
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────
    # Public API 
    # ──────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        prompts: List[str],
        frame_id: int = 0,
    ) -> List[Detection]:
        if image is None or image.size == 0:
            return []

        classes: List[str] = []
        for p in prompts:
            if " . " in p or p.endswith(" ."):
                classes.extend(_parse_prompts(p))
            else:
                classes.append(p.strip())
        classes = [c for c in classes if c]

        if not classes:
            return []

        if classes != self._last_classes:
            self._model.set_classes(classes)
            self._last_classes = classes

        prompt_text = " . ".join(classes) + " ."

        try:
            results = self._model.predict(
                source=image,
                conf=self.box_threshold,
                iou=self.nms_iou,
                device=self.device,
                verbose=False,   
            )
        except Exception as e:
            print(f"Warning: YOLO-World detection failed on frame {frame_id}: {e}")
            return []

        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return []

        boxes_data = results[0].boxes
        h, w = image.shape[:2]

        for i in range(len(boxes_data)):
            conf = float(boxes_data.conf[i].cpu())
            if conf < self.box_threshold:
                continue

            # xyxy pixel coords
            x1, y1, x2, y2 = boxes_data.xyxy[i].cpu().numpy()
            x1 = int(np.clip(x1, 0, w))
            y1 = int(np.clip(y1, 0, h))
            x2 = int(np.clip(x2, 0, w))
            y2 = int(np.clip(y2, 0, h))

            if x2 <= x1 or y2 <= y1:
                continue

            cls_idx = int(boxes_data.cls[i].cpu())
            class_name = classes[cls_idx] if cls_idx < len(classes) else "unknown"

            detections.append(Detection(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                confidence=conf,
                class_name=class_name,
                prompt_used=prompt_text,
                frame_id=frame_id,
            ))

        return detections

    def detect_with_interval(
        self,
        image: np.ndarray,
        prompts: List[str],
        frame_id: int,
        detection_interval: int,
    ) -> Tuple[bool, List[Detection]]:
        if frame_id % detection_interval == 0:
            return True, self.detect(image, prompts, frame_id)
        return False, []

    def warmup(self, image_size: Tuple[int, int] = (640, 640)):
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        self.detect(dummy, ["person"], frame_id=-1)
        print("  ✓ GPU warmed up")
