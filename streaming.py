"""
Simple MJPEG streaming server for Home Assistant integration.

Serves the latest frame (with detection annotations) at /video_feed
Add to Home Assistant as a Generic Camera with:
  still_image_url: http://<pi-ip>:8081/frame.jpg
  stream_source: http://<pi-ip>:8081/video_feed
"""
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2

import config

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe buffer for sharing frames between detection loop and streamer."""
    
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._last_update = 0
    
    def update(self, frame_rgb, detections=None):
        """
        Update the buffer with a new frame.
        
        Args:
            frame_rgb: RGB numpy array from camera
            detections: Optional list of bird detections to draw
        """
        # Convert RGB to BGR for OpenCV encoding
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw detection boxes if provided
        if detections:
            for bird in detections:
                x1, y1, x2, y2 = bird["bbox"]
                confidence = bird["confidence"]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Bird: {confidence:.0%}"
                cv2.putText(frame_bgr, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode to JPEG
        _, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        with self._lock:
            self._frame = jpeg.tobytes()
            self._last_update = time.time()
    
    def get_frame(self):
        """Get the latest JPEG frame (thread-safe)."""
        with self._lock:
            return self._frame
    
    def get_age(self):
        """Get seconds since last frame update."""
        with self._lock:
            if self._last_update == 0:
                return float('inf')
            return time.time() - self._last_update


# Global frame buffer - shared between main loop and HTTP server
frame_buffer = FrameBuffer()


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MJPEG streaming."""
    
    def log_message(self, format, *args):
        """Suppress default logging (too noisy)."""
        pass
    
    def do_GET(self):
        if self.path == '/video_feed':
            self._handle_mjpeg_stream()
        elif self.path == '/frame.jpg':
            self._handle_single_frame()
        elif self.path == '/health':
            self._handle_health()
        else:
            self.send_error(404, "Not found. Use /video_feed, /frame.jpg, or /health")
    
    def _handle_mjpeg_stream(self):
        """Stream MJPEG frames continuously."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            while True:
                frame = frame_buffer.get_frame()
                if frame is None:
                    # No frame yet, send a placeholder
                    time.sleep(0.1)
                    continue
                
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
                
                # Use configured FPS
                time.sleep(1.0 / config.STREAMING_FPS)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected - normal behavior
            pass
    
    def _handle_single_frame(self):
        """Return a single JPEG frame (for HA still_image_url)."""
        frame = frame_buffer.get_frame()
        
        if frame is None:
            self.send_error(503, "No frame available yet")
            return
        
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(frame))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(frame)
    
    def _handle_health(self):
        """Health check endpoint."""
        age = frame_buffer.get_age()
        status = "ok" if age < 5 else "stale"
        body = f'{{"status": "{status}", "frame_age_seconds": {age:.1f}}}'.encode()
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


class StreamingServer:
    """MJPEG streaming server that runs in a background thread."""
    
    def __init__(self, host='0.0.0.0', port=8081):
        self.host = host
        self.port = port
        self._server = None
        self._thread = None
    
    def start(self):
        """Start the streaming server in a background thread."""
        self._server = HTTPServer((self.host, self.port), StreamHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Streaming server started at http://{self.host}:{self.port}")
        logger.info(f"  - MJPEG stream: http://<this-ip>:{self.port}/video_feed")
        logger.info(f"  - Single frame: http://<this-ip>:{self.port}/frame.jpg")
        logger.info(f"  - Health check: http://<this-ip>:{self.port}/health")
    
    def stop(self):
        """Stop the streaming server."""
        if self._server:
            self._server.shutdown()
            self._thread.join(timeout=2)
            logger.info("Streaming server stopped")
    
    def update_frame(self, frame_rgb, detections=None):
        """
        Update the frame buffer with new data.
        Call this from your main detection loop.
        
        Args:
            frame_rgb: RGB numpy array from camera
            detections: Optional list of bird detections
        """
        frame_buffer.update(frame_rgb, detections)
