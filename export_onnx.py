"""
Export YOLOv8n to ONNX format for use with OpenCV DNN.
Run this on your dev machine (NOT on Raspberry Pi).
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx():
    from ultralytics import YOLO
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    onnx_path = models_dir / "yolov8n.onnx"
    
    if onnx_path.exists():
        logger.info(f"ONNX model already exists at {onnx_path}")
        return onnx_path
    
    logger.info("Downloading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    logger.info("Exporting to ONNX format...")
    export_path = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=12,  # Good compatibility
    )
    
    # Move to models directory if needed
    exported = Path(export_path)
    if exported.parent != models_dir:
        target = models_dir / "yolov8n.onnx"
        exported.rename(target)
        logger.info(f"Moved model to {target}")
        return target
    
    logger.info(f"Model exported to {export_path}")
    return export_path


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("YOLOv8n ONNX Export (run on dev machine)")
    logger.info("=" * 50)
    
    path = export_to_onnx()
    
    logger.info("=" * 50)
    logger.info(f"Done! Copy {path} to your Raspberry Pi")
    logger.info("=" * 50)
