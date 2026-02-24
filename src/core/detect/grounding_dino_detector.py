"""
GroundingDINO detector implementation
"""
import torch
import numpy as np
from typing import List, Tuple
import cv2
from PIL import Image
import warnings

from src.models.detection import Detection

warnings.filterwarnings('ignore')


class GroundingDINODetector:
    """
    GroundingDINO-based open-vocabulary object detector
    """
    
    def __init__(
        self,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GroundingDINO detector
        
        Args:
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            device: Device to run on (cuda/cpu)
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        
        print(f"Loading GroundingDINO on {device}...")
        
        try:
            from groundingdino.util.inference import load_model, load_image, predict
            from groundingdino.util import get_tokenlizer
            
            # Load model with auto-download
            self.model = load_model(
                model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                model_checkpoint_path="models/groundingdino_swint_ogc.pth",
                device=device
            )
            
            self.predict_fn = predict
            self.load_image_fn = load_image
            
            print("✓ GroundingDINO loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GroundingDINO: {e}")
    
    def detect(
        self,
        image: np.ndarray,
        prompts: List[str],
        frame_id: int = 0
    ) -> List[Detection]:
        """
        Detect objects in image using text prompts
        
        Args:
            image: Input image (BGR format from OpenCV)
            prompts: List of text prompts (e.g., ["person", "car"])
            frame_id: Current frame number
        
        Returns:
            List of Detection objects
        """
        if image is None or image.size == 0:
            return []
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        all_detections = []
        
        # Join prompts with . separator
        prompt_text = " . ".join(prompts) + " ."
        
        try:
            # Use load_image to convert PIL to tensor properly
            import tempfile
            import os
            
            # Save temporarily to use load_image function
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                image_pil.save(tmp_path)
            
            try:
                # Load image using GroundingDINO's function
                image_source, image_transformed = self.load_image_fn(tmp_path)
                
                # Run detection
                boxes, logits, phrases = self.predict_fn(
                    model=self.model,
                    image=image_transformed,
                    caption=prompt_text,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )
                
                # Get original image dimensions
                h, w = image.shape[:2]
                
                # Convert to Detection objects
                for box, confidence, phrase in zip(boxes, logits, phrases):
                    # GroundingDINO returns normalized coords [cx, cy, w, h]
                    cx, cy, box_w, box_h = box.cpu().numpy()
                    
                    # Convert to pixel coordinates [x1, y1, x2, y2]
                    x1 = int((cx - box_w / 2) * w)
                    y1 = int((cy - box_h / 2) * h)
                    x2 = int((cx + box_w / 2) * w)
                    y2 = int((cy + box_h / 2) * h)
                    
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    detection = Detection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        confidence=float(confidence),
                        class_name=phrase.strip(),
                        prompt_used=prompt_text,
                        frame_id=frame_id
                    )
                    
                    all_detections.append(detection)
            
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        except Exception as e:
            print(f"Warning: Detection failed on frame {frame_id}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return all_detections
    
    def detect_with_interval(
        self,
        image: np.ndarray,
        prompts: List[str],
        frame_id: int,
        detection_interval: int
    ) -> Tuple[bool, List[Detection]]:
        """
        Detect only on specified intervals
        
        Args:
            image: Input image
            prompts: Text prompts
            frame_id: Current frame number
            detection_interval: Run detection every N frames
        
        Returns:
            (should_detect, detections) tuple
        """
        should_detect = (frame_id % detection_interval == 0)
        
        if should_detect:
            detections = self.detect(image, prompts, frame_id)
            return True, detections
        else:
            return False, []