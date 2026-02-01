"""
Bird detection using YOLOv8 with NCNN backend
"""
import logging
from pathlib import Path
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


class BirdDetector:
    def __init__(self, model_path=None, confidence_threshold=None):
        self.model_path = model_path or config.MODEL_PATH
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.model = None
    
    def load(self):
        """Load the NCNN model"""
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run 'python setup_model.py' first to download and export the model."
            )
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info(f"Model loaded. Confidence threshold: {self.confidence_threshold}")
    
    def detect_birds(self, frame):
        """
        Run detection on a frame and return bird detections.
        
        Args:
            frame: numpy array (RGB image)
        
        Returns:
            list of dicts with keys: confidence, bbox (x1, y1, x2, y2)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        birds = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Check if it's a bird (COCO class 14)
                if class_id == config.BIRD_CLASS_ID:
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    birds.append({
                        "confidence": confidence,
                        "bbox": (int(x1), int(y1), int(x2), int(y2))
                    })
                    logger.debug(f"Bird detected: confidence={confidence:.2f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        
        return birds
    
    def has_bird(self, frame):
        """Quick check if any bird is in frame"""
        return len(self.detect_birds(frame)) > 0
