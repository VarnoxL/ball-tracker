# ball-tracker

Tracks the ball and players in soccer video, then detects passes and possession and writes an annotated video.

> **Status:** Planned, not implemented yet.

## Pipeline

1. **Input**: OpenCV reads the mp4 one frame at a time.
2. **Detection**: YOLOv8 (COCO pretrained) finds players (`person`) and the ball (`sports ball`). Players get IDs from ByteTrack.
3. **Ball tracking**: a hand-written Kalman filter (constant velocity) smooths the ball position. It predicts through dropouts and rejects detections that are too far from the prediction (gating).
4. **Events**: the player closest to the ball holds possession. When possession moves from player A to player B, that counts as a pass.
5. **Aggregation**: total passes and possession seconds per player.
6. **Visualization**: player boxes with IDs, a ball dot and trail, the possessor highlighted, and live counters, written to a new mp4.

## Setup

Requires Python 3.10+ and an NVIDIA GPU with CUDA.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## Usage

```bash
python tracker.py input.mp4 out.mp4
python test_tracker.py
```

## Known limitations

- **No teams.** Every possession change counts as a pass, interceptions included.
- **ID switches.** Player IDs can swap during occlusions, which creates fake passes.
- **Weak ball detection** on wide shots, because the COCO model isn't trained on soccer.
- **Pixel distances.** Thresholds depend on camera zoom. There is no pitch calibration.
