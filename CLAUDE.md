# CLAUDE.md

## 1. Project overview
A computer-vision pipeline that tracks the ball and players in soccer video and detects passes and possession, using YOLOv8 detection and a hand-written Kalman filter. It's a personal project to show depth in CV and algorithms: the tracking and event-detection layers are hand-written on purpose, and everything else uses libraries.

**Status:** not implemented yet. Only the README and this file exist.

## 2. Pipeline
Each frame goes through layers 1–3. Layers 4–6 work on the resulting time series.

| # | Layer | Input → Output | Built with |
|---|-------|----------------|------------|
| 1 | Input | mp4 → frames + `fps`, width, height | `cv2.VideoCapture` |
| 2 | Detection | frame → player boxes with IDs, best ball box | ultralytics YOLOv8 + ByteTrack |
| 3 | Ball tracking | ball box (or none) → smoothed `(x, y)` per frame | **hand-written** Kalman filter |
| 4 | Events | ball series + players → possession series, pass events | **hand-written** pure functions |
| 5 | Aggregation | events → pass count, possession seconds per player | `collections.Counter` |
| 6 | Visualization | frames + everything above → annotated mp4 | `cv2` drawing + `VideoWriter` |

## 3. File map
```
tracker.py        all 6 layers, in pipeline order, one section per layer
test_tracker.py   assert-based checks for the Kalman filter + events (synthetic data, no video)
requirements.txt  ultralytics, opencv-python, numpy (torch is installed separately with CUDA)
README.md         user-facing docs
```
Where things live inside `tracker.py`:
- **Tuning constants** at the top: `CONF_BALL`, `IMGSZ`, `GATE_PX`, `MAX_MISSED`, `POSSESS_PX`, `MIN_HOLD`, `TRAIL`
- **`Kalman` class**: ball state `[x, y, vx, vy]`, `predict()` / `update(z)`
- **`possessor(ball_xy, players)`**: the ID of the player nearest the ball, or `None`
- **`detect_events(possession_series, fps)`**: a list of `(frame, "pass", from_id, to_id)`
- **`main()`**: CLI, the frame loop, drawing, and the stats printout

Keep it as one file until it passes about 400 lines. Then split it along the layer lines (`detect.py`, `kalman.py`, `events.py`, `viz.py`).

## 4. Commands
```bash
# setup (once)
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True

# run
python tracker.py input.mp4 out.mp4

# test
python test_tracker.py
```
The YOLO weights (`yolov8m.pt`) download automatically on the first run. Weights and videos are gitignored.

## 5. Conventions and constraints
- **Coordinates:** pixel space `(x, y)`, origin at the top-left, never normalized.
- **Player position:** the foot point, `((x1 + x2) / 2, y2)`, which is the bottom-center of the box. The box center isn't used.
- **Frame index:** 0-based. Time in seconds is `frame / fps`.
- **Frame rate:** always read `fps` from the video. Never hardcode 30.
- **Lost ball:** the ball series holds `None` while the Kalman track is lost. Downstream code must handle `None`.
- **Tuning values** live only in the constants at the top of `tracker.py`. Never put magic numbers inline.
- **Event logic stays pure:** it takes series in and returns lists out. No cv2 and no I/O, so it can be tested without video.
- **Kalman math:** don't change `F`, `H`, `Q`, `R` or the gating without updating §6.
- **Simplifications** are marked with a `# ponytail:` comment that names the limit and how to upgrade.
- **Git:** never add a `Co-Authored-By` line or any Claude attribution to commits or PRs.

## 6. Design decisions
- **Kalman filter over moving-average smoothing:** it predicts through detection dropouts (occlusion, blur) and gives a predicted position to gate against. Smoothing does neither.
- **Constant-velocity model:** at 25–30 fps, gravity and bounces barely change velocity from one frame to the next, and the correction step absorbs the rest. Move to constant acceleration only if the trails visibly lag.
- **Gating (`GATE_PX`):** COCO's `sports ball` class fires on heads and the penalty spot. A detection far from the prediction is treated as noise.
- **YOLOv8 via ultralytics (not v5):** one pip install, a maintained API and built-in ByteTrack. Swapping in newer weights is a one-line change.
- **ByteTrack for players, the hand-written Kalman for the ball:** players aren't the focus of the project, so their tracking comes from the library. The ball is the algorithmic showcase.
- **Possession = nearest foot point within `POSSESS_PX`, held for `MIN_HOLD` frames:** the hold threshold stops the ball rolling past a player from counting as possession.
- **Pass = confirmed possessor changes from A to B:** it's simple and testable. The user's original idea (direction change + distance) is a later refinement.
- **`imgsz=1280`:** the ball is only a few pixels wide in broadcast shots, and the default 640 misses it.

## 7. Known limitations / TODOs
- [ ] **No teams:** every possession change counts as a pass, interceptions included. Fix: k-means on jersey colors.
- [ ] **ID switches:** ByteTrack swaps player IDs in occlusions, which creates fake passes. Fix: `botsort.yaml` with ReID.
- [ ] **Weak ball detection** on wide shots with COCO weights. Fix: fine-tune on a soccer-ball dataset.
- [ ] **Pixel thresholds depend on camera zoom.** Fix: a pitch homography to work in meters.
- [ ] **Camera cuts and replays** reset nothing yet, so tracks jump. Fix: detect scene cuts and reset the Kalman filter.
- [ ] **Single camera only.** No multi-angle fusion.
