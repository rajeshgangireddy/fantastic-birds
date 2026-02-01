"""
Script to download YOLOv8n and export to NCNN format for Raspberry Pi
Run this script once before starting the bird detector.
"""
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_model():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    ncnn_model_path = models_dir / "yolov8n_ncnn_model"
    
    if ncnn_model_path.exists():
        logger.info(f"NCNN model already exists at {ncnn_model_path}")
        return ncnn_model_path
    
    logger.info("Downloading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    logger.info("Exporting to NCNN format (this may take a few minutes)...")
    export_path = model.export(format="ncnn")
    
    # Move to models directory if needed
    exported = Path(export_path)
    if exported.parent != models_dir:
        target = models_dir / exported.name
        exported.rename(target)
        logger.info(f"Moved model to {target}")
        return target
    
    logger.info(f"Model exported successfully to {export_path}")
    return export_path


def test_model():
    """Quick test to verify model works"""
    import numpy as np
    
    logger.info("Testing model with dummy image...")
    model = YOLO("models/yolov8n_ncnn_model")
    
    # Create a dummy image (640x480 RGB)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    results = model(dummy_frame, verbose=False)
    logger.info(f"Model test successful! Detected {len(results[0].boxes)} objects in test image.")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("YOLOv8n NCNN Model Setup")
    logger.info("=" * 50)
    
    setup_model()
    test_model()
    
    logger.info("=" * 50)
    logger.info("Setup complete! You can now run: python main.py")
    logger.info("=" * 50)
