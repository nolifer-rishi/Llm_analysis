# Antigravity — Search & Rescue Tank Robot

> Phase 1: Chassis & Mobility — Android Wi-Fi joystick → Raspberry Pi 4 → BTS7960 motor driver → Tank chassis

## Project Layout

```
hackathon v1/
├── pi_controller/              # Raspberry Pi 4 code
│   ├── motor_controller.py     # WebSocket server + GPIO motor driver
│   ├── requirements.txt        # Python dependencies
│   └── install.sh              # Pi setup script
├── android_controller/         # Android app (Kotlin + Jetpack Compose)
│   ├── app/src/main/
│   │   ├── java/com/antigravity/controller/
│   │   │   ├── MainActivity.kt
│   │   │   ├── ui/             # Joystick, ControlScreen, ConnectionScreen
│   │   │   ├── network/        # WebSocketManager
│   │   │   └── viewmodel/      # DriveViewModel
│   │   ├── res/values/themes.xml
│   │   └── AndroidManifest.xml
│   ├── build.gradle.kts
│   └── settings.gradle.kts
├── README.md                   # This file
└── WIRING_GUIDE.md             # Hardware wiring reference
```

## Architecture

```
┌──────────────┐    WebSocket    ┌──────────────┐    GPIO/PWM    ┌──────────────┐
│   Android    │───────────────→│  Raspberry   │──────────────→│  BTS7960 x2  │
│   Joystick   │  ws://IP:8765  │   Pi 4 (2GB) │   pigpio      │  Motor Driver │
│   App        │←───────────────│              │               │  → 4x Motors  │
│              │   Telemetry    │              │               │              │
└──────────────┘                └──────────────┘               └──────────────┘
     Phone                       12V Battery                    Tank Chassis
    Hotspot ←─── Wi-Fi ────────→  + 5V Buck
```

## Quick Start

### 1. Raspberry Pi Setup

```bash
# SSH into the Pi
ssh pi@<pi-ip-address>

# Clone/copy files to Pi
mkdir -p ~/antigravity/pi_controller
# (copy pi_controller/ files here)

# Run setup
cd ~/antigravity/pi_controller
chmod +x install.sh
sudo ./install.sh

# Start the controller
python3 motor_controller.py
```

### 2. Android App

1. Open `android_controller/` in **Android Studio**
2. Sync Gradle → **Run** on your phone
3. Enter the Pi's IP address → **Connect**
4. Drive with the joystick!

### 3. Testing Without Hardware

```bash
# On any PC with Python 3.8+
pip install websockets
python motor_controller.py --dry-run

# From another terminal, test with websocat or browser
# Send: {"type":"drive","x":0.5,"y":1.0,"timestamp":0}
```

## Protocol

| Message | Direction | Description |
|---------|-----------|-------------|
| `{"type":"drive","x":0,"y":0,"timestamp":0}` | App → Pi | Joystick position (-1 to 1) |
| `{"type":"estop"}` | App → Pi | Emergency stop |
| `{"type":"estop_release"}` | App → Pi | Release E-STOP |
| `{"type":"telemetry",...}` | Pi → App | Motor speeds, uptime, status |

## Safety Features

- **Watchdog**: Motors auto-stop if no command received for 500ms
- **E-STOP**: Immediate motor stop, ignores all drive commands until released
- **Graceful shutdown**: SIGINT/SIGTERM cleanup zeroes all GPIO pins
- **Connection loss**: Wi-Fi dropout → watchdog → motors stop

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Can't connect from app | Check Pi IP (`hostname -I`), confirm phone hotspot is 2.4GHz |
| pigpio daemon error | Run `sudo pigpiod` before starting controller |
| Motors don't move | Check wiring per WIRING_GUIDE.md, confirm GNDs are tied |
| Jerky movement | Reduce `--max-speed` to 180, increase ramp step in code |
| One side doesn't work | Check the specific BTS7960 module's EN pins (must be HIGH) |
