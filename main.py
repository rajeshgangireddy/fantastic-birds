"""
Main application loop for bird detection
"""
import gc
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
from streaming import StreamingServer

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
    
    # Frame is already BGR from picamera2 (RGB888 format is actually BGR byte order)
    frame_bgr = frame.copy()
    
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
    streamer = StreamingServer(port=config.STREAMING_PORT) if config.STREAMING_ENABLED else None
    
    # Test Home Assistant connection
    logger.info("Testing Home Assistant connection...")
    if not notifier.test_connection():
        logger.error("Failed to connect to Home Assistant. Check your URL and token.")
        sys.exit(1)
    
    # Load model
    logger.info("Loading detection model...")
    detector.load()
    
    # Start streaming server (if enabled)
    if streamer:
        streamer.start()
    
    # Start camera
    logger.info("Starting camera...")
    camera.start()
    
    last_notification_time = 0
    frame_count = 0
    last_detections = None  # Cache detections for streaming overlay
    frame_interval = 1.0 / config.STREAMING_FPS if config.STREAMING_ENABLED else 0.1
    last_gc_time = time.time()
    
    logger.info(f"Streaming FPS: {config.STREAMING_FPS}, Detection every {config.DETECTION_INTERVAL} frames")
    logger.info("Bird detector running. Press Ctrl+C to stop.")
    
    try:
        while running:
            loop_start = time.time()
            
            # Capture frame
            frame = camera.capture_frame()
            frame_count += 1
            
            # Run detection only every N frames (detection is slow on Pi)
            birds = None
            if frame_count % config.DETECTION_INTERVAL == 0:
                birds = detector.detect_birds(frame)
                if birds:
                    last_detections = birds  # Cache for overlay
                else:
                    last_detections = None
            
            # Update streaming buffer (show cached detections on all frames)
            if streamer:
                streamer.update_frame(frame, last_detections)
            
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
            if frame_count % 2000 == 0:
                logger.info(f"Processed {frame_count} frames")
            
            # Periodic garbage collection to prevent memory buildup
            if time.time() - last_gc_time > 300:  # Every 5 minutes
                gc.collect()
                last_gc_time = time.time()
            
            # Maintain target frame rate
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    
    finally:
        logger.info("Shutting down...")
        if streamer:
            streamer.stop()
        camera.stop()
        logger.info("Bird detector stopped.")


if __name__ == "__main__":
    main()
