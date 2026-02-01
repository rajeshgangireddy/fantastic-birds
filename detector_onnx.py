"""
Lightweight YOLOv8 bird detector using OpenCV DNN.
No ultralytics/PyTorch required - works on Raspberry Pi!
"""
import logging
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# COCO class names (80 classes) - bird is index 14
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class BirdDetectorONNX:
    """
    YOLOv8 bird detector using OpenCV DNN backend.
    Lightweight - no PyTorch/ultralytics needed!
    """
    
    def __init__(self, model_path=None, confidence_threshold=None):
        self.model_path = model_path or str(Path(config.MODEL_PATH).parent / "yolov8n.onnx")
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.net = None
        self.input_size = (640, 640)
    
    def load(self):
        """Load the ONNX model with OpenCV DNN"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. "
                "Run 'python export_onnx.py' on your dev machine first, "
                "then copy models/yolov8n.onnx to the Pi."
            )
        
        logger.info(f"Loading ONNX model from {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        
        # Use CPU backend (OpenCV will auto-optimize for ARM)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        logger.info(f"Model loaded. Confidence threshold: {self.confidence_threshold}")
    
    def _preprocess(self, frame):
        """Preprocess frame for YOLOv8"""
        # Get original dimensions
        self.orig_h, self.orig_w = frame.shape[:2]
        
        # YOLOv8 expects RGB, but OpenCV DNN blobFromImage handles BGR->RGB
        # Letterbox resize to 640x640
        blob = cv2.dnn.blobFromImage(
            frame, 
            scalefactor=1/255.0,
            size=self.input_size,
            swapRB=True,  # BGR to RGB
            crop=False
        )
        return blob
    
    def _postprocess(self, outputs):
        """
        Postprocess YOLOv8 outputs.
        YOLOv8 output shape: (1, 84, 8400) where 84 = 4 bbox + 80 classes
        """
        birds = []
        
        # Get output and transpose: (1, 84, 8400) -> (8400, 84)
        output = outputs[0]
        if output.shape[0] == 1:
            output = output[0]
        output = output.T  # (8400, 84)
        
        # Calculate scale factors for letterbox
        scale_x = self.orig_w / self.input_size[0]
        scale_y = self.orig_h / self.input_size[1]
        scale = max(scale_x, scale_y)
        
        # Padding offsets for letterbox
        pad_x = (self.input_size[0] * scale - self.orig_w) / 2
        pad_y = (self.input_size[1] * scale - self.orig_h) / 2
        
        for detection in output:
            # First 4 values are bbox (cx, cy, w, h), rest are class scores
            cx, cy, w, h = detection[:4]
            class_scores = detection[4:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # Filter by confidence and bird class (14)
            if confidence < self.confidence_threshold:
                continue
            
            if class_id != config.BIRD_CLASS_ID:
                continue
            
            # Convert from center format to corner format and scale to original image
            x1 = int((cx - w/2) * scale - pad_x)
            y1 = int((cy - h/2) * scale - pad_y)
            x2 = int((cx + w/2) * scale - pad_x)
            y2 = int((cy + h/2) * scale - pad_y)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, self.orig_w))
            y1 = max(0, min(y1, self.orig_h))
            x2 = max(0, min(x2, self.orig_w))
            y2 = max(0, min(y2, self.orig_h))
            
            birds.append({
                "confidence": float(confidence),
                "bbox": (x1, y1, x2, y2)
            })
        
        # Apply Non-Maximum Suppression
        if birds:
            birds = self._nms(birds, iou_threshold=0.45)
        
        return birds
    
    def _nms(self, detections, iou_threshold=0.45):
        """Non-Maximum Suppression"""
        if not detections:
            return []
        
        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # OpenCV 4.x returns indices directly
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        
        return [detections[i] for i in indices]
    
    def detect_birds(self, frame):
        """
        Run detection on a frame and return bird detections.
        
        Args:
            frame: numpy array (RGB or BGR image)
        
        Returns:
            list of dicts with keys: confidence, bbox (x1, y1, x2, y2)
        """
        if self.net is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Preprocess
        blob = self._preprocess(frame)
        
        # Run inference
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        # Postprocess
        birds = self._postprocess([outputs])
        
        for bird in birds:
            logger.debug(
                f"Bird detected: confidence={bird['confidence']:.2f}, "
                f"bbox={bird['bbox']}"
            )
        
        return birds
    
    def has_bird(self, frame):
        """Quick check if any bird is in frame"""
        return len(self.detect_birds(frame)) > 0


# Alias for compatibility with existing code
BirdDetector = BirdDetectorONNX
