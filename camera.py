"""
Pi Camera v2 capture module using picamera2
"""
import logging
from picamera2 import Picamera2

import config

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, width=None, height=None):
        self.width = width or config.CAMERA_WIDTH
        self.height = height or config.CAMERA_HEIGHT
        self.picam2 = None
    
    def start(self):
        """Initialize and start the camera"""
        logger.info(f"Starting camera at {self.width}x{self.height}")
        self.picam2 = Picamera2()
        
        # Configure for still capture
        # Note: RGB888 format actually outputs BGR byte order on Pi
        camera_config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        logger.info("Camera started successfully")
    
    def capture_frame(self):
        """Capture and return a single frame as numpy array (BGR byte order)"""
        if self.picam2 is None:
            raise RuntimeError("Camera not started. Call start() first.")
        return self.picam2.capture_array()
    
    def stop(self):
        """Stop the camera"""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
            logger.info("Camera stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
