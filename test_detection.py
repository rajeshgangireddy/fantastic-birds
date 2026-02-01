"""
Test script to verify bird detection with a static image.
Usage:
    python test_detection.py <image_path>
    python test_detection.py  # Downloads a sample bird image
"""
import sys
import logging
from pathlib import Path

import cv2
import requests
import numpy as np

import config

# Use ONNX detector (lightweight, no PyTorch) or Ultralytics
if config.USE_ONNX_DETECTOR:
    from detector_onnx import BirdDetector
else:
    from detector import BirdDetector

from notifier import HomeAssistantNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_sample_bird_image():
    """Download a sample bird image for testing"""
    # Using a public domain bird image from Unsplash
    url = "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=640"
    
    logger.info("Downloading sample bird image...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BirdDetector/1.0)"}
    response = requests.get(url, timeout=30, headers=headers)
    response.raise_for_status()
    
    # Convert to numpy array
    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Save locally
    sample_path = Path("test_bird.jpg")
    cv2.imwrite(str(sample_path), img)
    logger.info(f"Saved sample image to {sample_path}")
    
    return sample_path


def test_detection(image_path):
    """Test bird detection on an image"""
    logger.info(f"Loading image: {image_path}")
    
    # Load image (OpenCV loads as BGR)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.error(f"Failed to load image: {image_path}")
        return False
    
    # Convert to RGB (YOLO expects RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    logger.info(f"Image size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
    
    # Initialize detector
    logger.info("Loading model...")
    detector = BirdDetector()
    detector.load()
    
    # Run detection
    logger.info(f"Running detection (confidence threshold: {config.CONFIDENCE_THRESHOLD:.0%})...")
    birds = detector.detect_birds(img_rgb)
    
    if birds:
        logger.info(f"✅ Detected {len(birds)} bird(s)!")
        for i, bird in enumerate(birds, 1):
            logger.info(f"   Bird {i}: confidence={bird['confidence']:.1%}, bbox={bird['bbox']}")
        
        # Draw boxes and save result
        for bird in birds:
            x1, y1, x2, y2 = bird["bbox"]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Bird: {bird['confidence']:.0%}"
            cv2.putText(img_bgr, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        output_path = Path("test_result.jpg")
        cv2.imwrite(str(output_path), img_bgr)
        logger.info(f"Saved annotated image to {output_path}")
        
        return True
    else:
        logger.warning("❌ No birds detected in image")
        return False


def test_home_assistant():
    """Test Home Assistant connection"""
    logger.info("Testing Home Assistant connection...")
    logger.info(f"   URL: {config.HA_URL}")
    logger.info(f"   Mobile service: {config.HA_MOBILE_NOTIFY_SERVICE}")
    
    notifier = HomeAssistantNotifier()
    
    if notifier.test_connection():
        logger.info("✅ Home Assistant connection successful!")
        return True
    else:
        logger.error("❌ Home Assistant connection failed!")
        logger.error("   Check HA_URL and HA_TOKEN in config.py")
        return False


def test_notification():
    """Send a test notification"""
    logger.info("Sending test notification...")
    
    notifier = HomeAssistantNotifier()
    
    # Send test notification
    dashboard_ok = notifier.send_persistent_notification(
        "This is a test notification from Bird Detector",
        title="🐦 Test Notification"
    )
    
    mobile_ok = notifier.send_mobile_notification(
        "This is a test notification from Bird Detector",
        title="🐦 Test Notification"
    )
    
    if dashboard_ok:
        logger.info("✅ Dashboard notification sent!")
    else:
        logger.error("❌ Dashboard notification failed")
    
    if mobile_ok:
        logger.info("✅ Mobile notification sent!")
    else:
        logger.error("❌ Mobile notification failed")
    
    return dashboard_ok and mobile_ok


def main():
    print("=" * 60)
    print("Bird Detector - Test Suite")
    print("=" * 60)
    
    # Determine image path
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)
    else:
        image_path = download_sample_bird_image()
    
    print("\n" + "=" * 60)
    print("TEST 1: Model & Bird Detection")
    print("=" * 60)
    detection_ok = test_detection(image_path)
    
    print("\n" + "=" * 60)
    print("TEST 2: Home Assistant Connection")
    print("=" * 60)
    ha_ok = test_home_assistant()
    
    if ha_ok:
        print("\n" + "=" * 60)
        print("TEST 3: Send Test Notification")
        print("=" * 60)
        response = input("Send a test notification to Home Assistant? [y/N]: ")
        if response.lower() == 'y':
            test_notification()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"   Detection: {'✅ PASS' if detection_ok else '❌ FAIL'}")
    print(f"   HA Connection: {'✅ PASS' if ha_ok else '❌ FAIL'}")
    
    if detection_ok and ha_ok:
        print("\n🎉 All tests passed! You're ready to run: python main.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()
