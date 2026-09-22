# ball-tracker

Tracks the ball and players in soccer video, then detects passes and possession and writes an annotated video.

> **Status:** layers 1, 2, 3 and 6 work (input, detection, Kalman ball tracking, overlay). Event detection is next.

On a 30 s 1080p broadcast clip (RTX 3060 Ti): 900 frames in 59 s, ball detected in 75% of frames and tracked in 100% (the Kalman filter predicts through gaps up to 1.0 s), ~20 players per frame.

## Pipeline

1. **Input**: OpenCV reads the mp4 one frame at a time.
2. **Detection**: two models. YOLOv8m (COCO) finds players, with IDs from ByteTrack. A soccer-specific YOLO11n fine-tune finds the ball, because COCO's `sports ball` class only managed 22% on this footage.
3. **Ball tracking**: a hand-written Kalman filter (constant velocity) smooths the ball position. It predicts through dropouts and rejects detections that are too far from the prediction (gating).
4. **Events**: the player closest to the ball holds possession. When possession moves from player A to player B, that counts as a pass.
5. **Aggregation**: total passes and possession seconds per player.
6. **Visualization**: player boxes with IDs, a ball dot and trail, the possessor highlighted, and live counters, written to a new mp4.

## Setup

Requires Python 3.10–3.13 (PyTorch has no 3.14 wheels) and an NVIDIA GPU with CUDA.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

Then fetch the ball weights (5 MB, not in the repo):

```bash
curl -L -o ball.pt https://huggingface.co/martinjolif/yolo-football-ball-detection/resolve/main/yolo-football-ball-detection.pt
```

## Usage

```bash
python tracker.py input.mp4 out.mp4
python test_tracker.py
```

## Known limitations

- **No teams.** Every possession change counts as a pass, interceptions included.
- **ID switches.** Player IDs can swap during occlusions, which creates fake passes.
- **Third-party ball weights.** `ball.pt` has unknown training data and failure modes.
- **Pixel distances.** Thresholds depend on camera zoom. There is no pitch calibration.
