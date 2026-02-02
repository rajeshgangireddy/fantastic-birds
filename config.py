"""
Configuration for Bird Detector
Edit these values or set environment variables
"""
import os

# =============================================================================
# HOME ASSISTANT CONFIGURATION
# =============================================================================
# Your Home Assistant URL (e.g., http://192.168.1.100:8123)
HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123")

# Long-lived access token from HA (Profile -> Security -> Create Token)
HA_TOKEN = os.getenv("HA_TOKEN", "YOUR_LONG_LIVED_ACCESS_TOKEN")

# Mobile app notification service name (check HA -> Developer Tools -> Services)
# Usually: notify.mobile_app_<your_phone_name>
HA_MOBILE_NOTIFY_SERVICE = os.getenv("HA_MOBILE_NOTIFY_SERVICE", "notify.mobile_app_pixel_8_pro")

# =============================================================================
# DETECTION CONFIGURATION
# =============================================================================
# Confidence threshold for bird detection (0.0 - 1.0)
# Lower = more detections (more false positives), Higher = fewer detections
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.2"))

# Cooldown between notifications in seconds (to avoid spam)
NOTIFICATION_COOLDOWN_SECONDS = int(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", "60"))

# COCO class ID for bird
BIRD_CLASS_ID = 14

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================
# Camera resolution (lower = faster inference on Pi 3)
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))

# =============================================================================
# IMAGE SAVING CONFIGURATION
# =============================================================================
# Save images when bird detected
SAVE_IMAGES = os.getenv("SAVE_IMAGES", "true").lower() == "true"

# Directory to save captured bird images
IMAGES_DIR = os.getenv("IMAGES_DIR", "/home/pi/bird_images")

# Maximum number of images to keep (oldest deleted first, 0 = unlimited)
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "100"))

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Path to the ONNX model (lightweight, works on Pi)
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.onnx")

# Use ONNX detector (True) or Ultralytics NCNN (False)
# ONNX is recommended for Raspberry Pi - no PyTorch needed!
USE_ONNX_DETECTOR = os.getenv("USE_ONNX_DETECTOR", "true").lower() == "true"

# =============================================================================
# STREAMING CONFIGURATION (for Home Assistant Generic Camera)
# =============================================================================
# Enable MJPEG streaming server
STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "true").lower() == "true"

# Port for the streaming server
STREAMING_PORT = int(os.getenv("STREAMING_PORT", "8081"))

# Target FPS for streaming (higher = smoother but more CPU/bandwidth)
# Recommended: 15-25 for smooth video, 5-10 for low bandwidth
STREAMING_FPS = int(os.getenv("STREAMING_FPS", "20"))

# How often to run detection (in frames). Detection is slow on Pi.
# 1 = every frame (slow), 5 = every 5th frame (faster streaming)
# At 20 FPS with DETECTION_INTERVAL=5, detection runs 4 times/second
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", "10"))

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
