# Real-Time Hand Gesture-Based Image Puzzle Game Using Computer Vision

A real-time interactive puzzle game controlled entirely by hand gestures, built with Python, OpenCV, and MediaPipe. The player uses a webcam to capture an image, which is then divided into a grid of tiles. The tiles are shuffled and the player reassembles them using pinch and drag gestures, without touching any keyboard or mouse.

---

## Features

- Contactless control using hand gesture recognition
- Real-time image capture directly from the webcam as the puzzle source
- 3x3 sliding tile puzzle with shuffle and lock-in mechanics
- Pinch gesture debouncing to prevent accidental tile drops
- Metal gesture hold-timer to prevent accidental game resets
- Particle effects and animated score popups on correct tile placement
- Minimalist black and white UI theme
- FPS counter, move counter, and progress bar

---

## Gestures

| Gesture | Action |
|---|---|
| Move open hand | Reposition the capture frame |
| Pinch (index + thumb) | Resize frame / grab a tile |
| Hold pinch (0.8 seconds) | Capture image and start puzzle |
| Release pinch over a tile | Swap tiles |
| Metal gesture (index + pinky up) held 0.6 seconds | Reset game |

---

## Project Structure

```
gesture-puzzle/
├── main.py          # Entry point, run this file
├── constants.py     # All colors, thresholds, and settings
├── drawing.py       # Drawing utilities, visual effects, Particle class
├── game.py          # PuzzleGame class, game logic, main loop
└── requirements.txt
```

---

## Requirements

- Python 3.9 or higher
- Webcam

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/chaidenfoanto/cv-puzzle-game.git
cd cv-puzzle-game
```

**2. Create and activate a virtual environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Program

```bash
python main.py
```

Press `Q` to quit the game at any time.

---

## Configuration

All adjustable settings are located in `constants.py`. No other file needs to be modified for basic configuration.

| Constant | Default | Description |
|---|---|---|
| `GRID_SIZE` | `3` | Puzzle grid size (3 = 3x3) |
| `PINCH_THRESHOLD` | `45` | Pixel distance to register a pinch |
| `PINCH_RELEASE_DEBOUNCE` | `6` | Frames before pinch release is confirmed |
| `METAL_HOLD_REQUIRED` | `0.6` | Seconds to hold metal gesture for reset |
| `CAPTURE_HOLD_TIME` | `0.8` | Seconds to hold pinch to capture image |
| `CAP_WIDTH` / `CAP_HEIGHT` | `1280 x 720` | Camera resolution |

---

## Dependencies

| Package | Version |
|---|---|
| opencv-python | >= 4.8.0 |
| mediapipe | >= 0.10.0 |
| numpy | >= 1.24.0 |

---

## License

This project is intended for educational purposes.
