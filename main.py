"""
Main application loop for bird detection
"""
import os
import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path

import cv2

import config
from camera import Camera
from notifier import HomeAssistantNotifier

# Use ONNX detector (lightweight, no PyTorch) or Ultralytics
if config.USE_ONNX_DETECTOR:
    from detector_onnx import BirdDetector
else:
    from detector import BirdDetector

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    logger.info("Shutdown signal received, stopping...")
    running = False


def setup_image_directory():
    """Create image directory if it doesn't exist"""
    if config.SAVE_IMAGES:
        path = Path(config.IMAGES_DIR)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Image directory: {path}")
        return path
    return None


def cleanup_old_images(images_dir):
    """Remove oldest images if exceeding MAX_IMAGES"""
    if not config.SAVE_IMAGES or config.MAX_IMAGES <= 0:
        return
    
    images = sorted(images_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime)
    
    while len(images) > config.MAX_IMAGES:
        oldest = images.pop(0)
        oldest.unlink()
        logger.debug(f"Deleted old image: {oldest}")


def save_detection_image(frame, birds, images_dir):
    """Save frame with detection boxes drawn"""
    if not config.SAVE_IMAGES or images_dir is None:
        return None
    
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes
    for bird in birds:
        x1, y1, x2, y2 = bird["bbox"]
        confidence = bird["confidence"]
        
        # Draw rectangle
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Bird: {confidence:.0%}"
        cv2.putText(frame_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bird_{timestamp}.jpg"
    filepath = images_dir / filename
    cv2.imwrite(str(filepath), frame_bgr)
    logger.info(f"Saved detection image: {filepath}")
    
    # Cleanup old images
    cleanup_old_images(images_dir)
    
    return filepath


def main():
    global running
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 50)
    logger.info("Starting Bird Detector")
    logger.info("=" * 50)
    logger.info(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD:.0%}")
    logger.info(f"Notification cooldown: {config.NOTIFICATION_COOLDOWN_SECONDS}s")
    logger.info(f"Camera resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    
    # Setup
    images_dir = setup_image_directory()
    
    # Initialize components
    camera = Camera()
    detector = BirdDetector()
    notifier = HomeAssistantNotifier()
    
    # Test Home Assistant connection
    logger.info("Testing Home Assistant connection...")
    if not notifier.test_connection():
        logger.error("Failed to connect to Home Assistant. Check your URL and token.")
        sys.exit(1)
    
    # Load model
    logger.info("Loading detection model...")
    detector.load()
    
    # Start camera
    logger.info("Starting camera...")
    camera.start()
    
    last_notification_time = 0
    frame_count = 0
    
    logger.info("Bird detector running. Press Ctrl+C to stop.")
    
    try:
        while running:
            # Capture frame
            frame = camera.capture_frame()
            frame_count += 1
            
            # Run detection
            birds = detector.detect_birds(frame)
            
            if birds:
                current_time = time.time()
                time_since_last = current_time - last_notification_time
                
                # Get best detection (highest confidence)
                best_bird = max(birds, key=lambda x: x["confidence"])
                logger.info(f"Bird detected! Count: {len(birds)}, Best confidence: {best_bird['confidence']:.0%}")
                
                # Check cooldown
                if time_since_last >= config.NOTIFICATION_COOLDOWN_SECONDS:
                    # Save image
                    image_path = save_detection_image(frame, birds, images_dir)
                    
                    # Send notification
                    notifier.notify_bird_detected(
                        confidence=best_bird["confidence"],
                        image_url=str(image_path) if image_path else None
                    )
                    last_notification_time = current_time
                else:
                    remaining = config.NOTIFICATION_COOLDOWN_SECONDS - time_since_last
                    logger.debug(f"Cooldown active, {remaining:.0f}s remaining")
            
            # Log progress periodically
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
            
            # Small delay to prevent CPU overload
            time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    
    finally:
        logger.info("Shutting down...")
        camera.stop()
        logger.info("Bird detector stopped.")


if __name__ == "__main__":
    main()
