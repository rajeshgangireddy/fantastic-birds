# 🐦 Fantastic Birds - Bird Detection for Raspberry Pi

Detect birds using YOLOv8n on a Raspberry Pi 3 with Pi Camera v2, and send notifications to Home Assistant.

## Features

- **Real-time bird detection** using YOLOv8n (nano) model
- **NCNN backend** optimized for ARM/Raspberry Pi
- **Home Assistant integration** - dashboard + mobile push notifications
- **Image capture** - saves snapshots when birds are detected
- **Configurable** - confidence threshold, cooldown, etc.
- **Auto-start** - runs as a systemd service

## Hardware Requirements

- Raspberry Pi 3 (or newer)
- Pi Camera v2
- Network connection to Home Assistant

## Quick Start

### 1. Clone and Install

```bash
# On your Raspberry Pi
cd ~
git clone <your-repo-url> fantastic-birds
cd fantastic-birds

# Install dependencies
pip3 install -r requirements.txt
```

### 2. Setup the Model

```bash
# Download YOLOv8n and export to NCNN format
python3 setup_model.py
```

This downloads the pre-trained YOLOv8n model (which already knows how to detect birds from COCO dataset) and exports it to NCNN format for optimal performance on Raspberry Pi.

### 3. Configure Home Assistant

1. **Get a Long-Lived Access Token:**
   - Go to your Home Assistant → Profile → Security
   - Scroll to "Long-Lived Access Tokens" → Create Token
   - Copy the token (you'll only see it once!)

2. **Find your mobile notification service:**
   - Go to Developer Tools → Services
   - Search for `notify.mobile_app_`
   - Note your service name (e.g., `notify.mobile_app_pixel_7`)

3. **Edit configuration:**

   Option A: Edit `config.py` directly:
   ```python
   HA_URL = "http://192.168.1.100:8123"  # Your HA URL
   HA_TOKEN = "your_long_lived_token_here"
   HA_MOBILE_NOTIFY_SERVICE = "notify.mobile_app_your_phone"
   ```

   Option B: Use environment variables (recommended for systemd):
   ```bash
   export HA_URL="http://192.168.1.100:8123"
   export HA_TOKEN="your_token"
   export HA_MOBILE_NOTIFY_SERVICE="notify.mobile_app_your_phone"
   ```

### 4. Test Run

```bash
python3 main.py
```

Point your camera at a bird (or a picture of a bird) to test!

### 5. Setup as a Service (Auto-start)

```bash
# Edit the service file with your settings
nano bird-detector.service

# Copy to systemd
sudo cp bird-detector.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable bird-detector
sudo systemctl start bird-detector

# Check status
sudo systemctl status bird-detector

# View logs
journalctl -u bird-detector -f
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `HA_URL` | `http://homeassistant.local:8123` | Home Assistant URL |
| `HA_TOKEN` | - | Long-lived access token |
| `HA_MOBILE_NOTIFY_SERVICE` | `notify.mobile_app_phone` | Mobile notification service |
| `CONFIDENCE_THRESHOLD` | `0.2` (20%) | Detection confidence threshold |
| `NOTIFICATION_COOLDOWN_SECONDS` | `60` | Seconds between notifications |
| `CAMERA_WIDTH` | `640` | Camera capture width |
| `CAMERA_HEIGHT` | `480` | Camera capture height |
| `SAVE_IMAGES` | `true` | Save images on detection |
| `IMAGES_DIR` | `/home/pi/bird_images` | Where to save images |
| `MAX_IMAGES` | `100` | Max images to keep (0=unlimited) |
| `LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
fantastic-birds/
├── main.py              # Main application loop
├── config.py            # Configuration
├── camera.py            # Pi Camera capture
├── detector.py          # YOLOv8 bird detection
├── notifier.py          # Home Assistant notifications
├── setup_model.py       # Model download/export script
├── requirements.txt     # Python dependencies
├── bird-detector.service # Systemd service file
└── models/              # NCNN model (created by setup_model.py)
    └── yolov8n_ncnn_model/
```

## Performance Notes

- **Expected FPS on Pi 3:** ~0.5-2 FPS (YOLOv8n with NCNN)
- Lower resolution = faster inference
- The 100ms delay between frames helps prevent CPU overload
- Monitor temperature: `vcgencmd measure_temp`

## Troubleshooting

### Camera not working
```bash
# Test camera
libcamera-hello
# Check if camera is enabled
sudo raspi-config  # Interface Options → Camera
```

### Home Assistant connection fails
- Verify HA_URL is reachable from Pi: `curl http://your-ha-ip:8123/api/`
- Check token is valid (no extra spaces/newlines)
- Ensure HA is not blocking the Pi's IP

### Model not found
```bash
# Re-run model setup
python3 setup_model.py
```

### Service won't start
```bash
# Check logs
journalctl -u bird-detector -e
# Verify paths in service file match your setup
```

## License

MIT
A WIP framework to detect birds - optimised for Raspberry Pi 
