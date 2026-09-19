# HANDOFF

Written 2026-09-19, at the end of step 1. Read this plus `CLAUDE.md` and you have everything.

---

## 1. Where the project stands

| Layer | What it does | Status |
|---|---|---|
| 1 Input | read mp4 frame by frame | **done** |
| 2 Detection | YOLO finds players + ball | **done** |
| 3 Ball tracking | Kalman filter over ball positions | **not started** — Daniel writes this |
| 4 Events | possession + pass rules | **not started** — Daniel writes this |
| 5 Aggregation | pass count, possession seconds | not started |
| 6 Visualization | overlay + mp4 out | **done** (raw detections, no smoothing) |

Split agreed up front: Claude writes the plumbing (1, 2, 5, 6), Daniel hand-writes the algorithmic layers (3, 4), because those are the parts worth defending in an interview.

The pipeline runs end to end today. It draws raw detections, so the ball dot flickers and the trail jumps across the pitch during dropouts. That is intentional: the dropouts have to be visible before anything hides them.

## 2. Measured results (the numbers that matter)

Full run on `clip.mp4` — 30.03 s, 1908×1080, 29.97 fps, 900 frames, RTX 3060 Ti:

```
900 frames in 58.8s (15.3 fps)
ball detected in 678/900 frames (75.3%)
longest gap: 31 frames (1.0s)
players/frame: avg 20.0
```

Read them this way:
- **75.3%** — one frame in four has no ball measurement. That is what the Kalman filter's predict step is for.
- **31 frames / 1.0 s** — the worst dropout in this clip. `MAX_MISSED` must exceed ~35 or the track dies mid-play.
- **15.3 fps** — processing speed, not video speed. A 30 s clip costs ~1 minute per run. Two models are ~2 fps of that; the rest is `imgsz=1280`.
- **20.0 players/frame** — includes the referee, the two coaches on the touchline and anyone in the technical area. There is no "is this a player" filter.

## 3. Files

```
tracker.py        160 lines — layers 1, 2, 6
test_tracker.py    14 lines — asserts for the pure helpers
requirements.txt            — ultralytics, opencv-python, numpy
CLAUDE.md                   — project context: pipeline, conventions, design decisions
README.md                   — user-facing setup and usage
HANDOFF.md                  — this file
clip.mp4          gitignored — the 30 s test clip
out.mp4           gitignored — last render
ball.pt           gitignored — soccer ball weights, 5.3 MB
.venv/            gitignored — Python 3.13 environment
```

### `tracker.py`, section by section

- **L15-25, tuning knobs.** Every magic number lives here, by convention (CLAUDE.md §5). `PLAYER_MODEL`, `BALL_MODEL`, `IMGSZ=1280`, `CONF_BALL=0.20`, `CONF_PLAYER=0.35`, `TRAIL=30`, `PERSON_CLS=0`, plus two BGR colors.
- **L29 `foot_point(box)`** → `((x1+x2)/2, y2)`, the bottom-center. A player's ground position is at their feet; the box center floats near the chest and would skew every ball-to-player distance in layer 4.
- **L35 `center(box)`** → box midpoint, used for the ball.
- **L40 `best_ball(dets)`** → highest-confidence detection as `(x, y, conf)`, or `None` for an empty list. One ball per frame.
- **L54 `detect_players(model, frame)`** → `[(track_id, box), ...]`. Calls `model.track(persist=True, tracker="bytetrack.yaml", classes=[0])`. `.track` rather than `.predict` is what yields IDs across frames; `persist=True` tells ByteTrack this frame follows the last one. Boxes without an ID are dropped.
- **L73 `detect_ball(model, frame)`** → `(x, y, conf)` or `None`. Plain `.predict` on the single-class model, then `best_ball`.
- **L83 `draw(...)`** → player rectangles with the ID above each, a filled circle on the ball with its confidence, `cv2.polylines` over the trail deque, and a header line reading `frame N  players N  ball Y/-`.
- **L103 `main()`** → argparse, `VideoCapture`, the frame loop, `VideoWriter` (`mp4v`), a progress line every 30 frames, and the summary block above.

`fps`, width and height always come from the file. Nothing assumes 30 fps.

## 4. Environment, exactly

- **Python 3.13**, not the system default. The default is 3.14 and **PyTorch has no 3.14 wheels**, which is the single most likely thing to waste an hour if forgotten. The venv was created with `py -3.13 -m venv .venv`.
- **torch 2.6.0+cu124**, installed from `https://download.pytorch.org/whl/cu124`. The plain PyPI wheel on Windows is CPU-only and runs roughly 10× slower.
- **ultralytics 8.4.156**, **opencv-python 5.0.0.93**. Ultralytics auto-installed `lap` on the first `.track` call.
- **GPU**: RTX 3060 Ti, 8 GB, driver 610.47. `torch.cuda.is_available()` is True.
- **yt-dlp and ffmpeg** came from winget. Their exes are under `%LOCALAPPDATA%\Microsoft\WinGet\Packages\...`, and the PATH aliases only resolve in a shell started after the install.

### Rebuild from scratch
```powershell
cd C:\Users\danie\Desktop\Projects\ball-tracker
py -3.13 -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -r requirements.txt
curl -L -o ball.pt https://huggingface.co/martinjolif/yolo-football-ball-detection/resolve/main/yolo-football-ball-detection.pt
.venv\Scripts\python test_tracker.py
.venv\Scripts\python tracker.py clip.mp4 out.mp4
```
`yolov8m.pt` downloads itself on the first run. `clip.mp4` is not in the repo; section 6 says how it was made.

## 5. The models

| File | What | Detects | Origin |
|---|---|---|---|
| `yolov8m.pt` | Ultralytics COCO model, 80 classes | players, via `person` | auto-downloaded |
| `ball.pt` | YOLO11n fine-tune, 1 class | the ball | [martinjolif/yolo-football-ball-detection](https://huggingface.co/martinjolif/yolo-football-ball-detection) |

**Neither model was trained here.** `ball.pt` is a third party's weights file, reported mAP50 0.891, downloaded with explicit approval because a `.pt` is executable code from a stranger.

### Why two models: the measurement that forced it

The first full run used COCO's `sports ball` class and found the ball in **22.3%** of frames, with a **5.6 s** worst gap. Sampled frames showed the ball plainly visible and undetected, so it was the model's ceiling, not a bug. A sweep over 300 frames:

| Setting | Ball found |
|---|---|
| yolov8m, imgsz 1280, conf 0.05 | 27.3% |
| yolov8m, imgsz 1920, conf 0.05 | 26.7% |
| yolov8x, imgsz 1920, conf 0.05 | 27.3% |
| **ball.pt, imgsz 1280, conf 0.20** | **80.0%** |
| ball.pt, imgsz 1280, conf 0.05 | 85.0% (and 42/300 frames with >1 box) |

A larger model and a larger input changed nothing, which is the signal that tuning is exhausted: COCO learned large close-up balls, not a 10-pixel ball on grass. `conf=0.20` was chosen over `0.05` because the extra 5% comes with double the multi-box frames, and those extra boxes are false positives that a filter must then reject.

Players stayed on COCO because `person` detection was already finding 20–23 people per frame.

## 6. The test clip

```powershell
yt-dlp --download-sections "*1:18:00-1:18:30" --force-keyframes-at-cuts `
  -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]" --recode mp4 `
  -o clip.mp4 "https://www.youtube.com/watch?v=YsWzugAnsBw"
```
Real Madrid vs Barcelona, Supercopa de España, from the 60:53 mark. One continuous wide tactical shot, no cuts, no replays, both boxes of players visible. Good clip criteria: 20-30 s, 1080p, one unbroken wide shot. Highlight reels are useless here, because a cut every three seconds teleports the ball.

## 7. Open problems

1. **Player ID churn (the big one).** ByteTrack IDs reach the 1000s by frame 750 with ~22 people on the pitch, so tracks are being dropped and re-created constantly. Possession-per-player stats will fragment across dozens of IDs for the same person. Fix candidates: `botsort.yaml` with ReID instead of ByteTrack, a lower `CONF_PLAYER`, or tuning ByteTrack's `track_buffer`. Measure it first — count distinct IDs per run and treat that number as the metric.
2. **No teams.** Nothing distinguishes Madrid from Barcelona, so a possession change is indistinguishable from an interception. Fix: k-means on jersey pixels inside each box.
3. **Non-players counted.** The referee and both coaches are detected as players. They stand still, so `POSSESS_PX` mostly ignores them, but they pollute `players/frame`.
4. **`ball.pt` is a black box.** Unknown training data, unknown failure modes, and it can't be cited as your own work. Fine-tuning a replacement on the Roboflow [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) dataset (663 images, 4 classes) takes 1-2 hours on the 3060 Ti and produces real metrics to talk about.
5. **Pixel thresholds.** Every distance is in pixels, so a zoom change silently rescales the rules. A pitch homography would put them in meters.
6. **No detection cache.** Each tuning iteration re-runs YOLO for ~60 s. Deliberate for now; when event-rule tuning starts to dominate, dump detections to JSON once and replay them in under a second.

## 8. Next step, with spec

**Layer 3, the Kalman filter — Daniel writes this.**

State `x = [px, py, vx, vy]`, constant velocity, one time step per frame (`dt = 1`, work in frames, convert to seconds only when printing).

```
F = [[1,0,1,0],    H = [[1,0,0,0],
     [0,1,0,1],         [0,1,0,0]]
     [0,0,1,0],
     [0,0,0,1]]
```

Per frame: `predict()` first, unconditionally. Then `update(z)` only if a detection exists **and** lies within `GATE_PX` of the predicted position — that gate is what rejects a false ball on a penalty spot. After `MAX_MISSED` consecutive frames with no accepted measurement, declare the track lost and let the next detection start a fresh one.

Starting values to tune from, given the measurements above: `GATE_PX ≈ 80`, `MAX_MISSED ≈ 35` (the worst real gap was 31). `Q` small (the ball's velocity is fairly steady), `R` moderate (detections jitter by a few pixels).

Output contract for layer 4: one entry per frame, `(x, y, measured: bool)` or `None` while lost. Downstream code must handle `None`.

Tests to write in `test_tracker.py`, no framework, just asserts:
1. A ball moving in a straight line, with frames 10-20 blanked, stays within a few px of truth through the gap.
2. A detection 500 px away from the prediction is rejected and does not move the state.
3. After `MAX_MISSED + 1` empty frames, the track reports lost.

Then layer 4: `possessor(ball_xy, players)` using `foot_point` and `POSSESS_PX`, and `detect_events(series, fps)` with the `MIN_HOLD` debounce, emitting `(frame, "pass", from_id, to_id)`.

## 9. Conventions worth not re-deriving

- Pixel space `(x, y)`, top-left origin, never normalized.
- Player position = `foot_point`, never the box center.
- Frame index is 0-based; seconds are `frame / fps`.
- All tuning values live in the constants block, never inline.
- Event logic stays pure — series in, lists out, no cv2, no I/O — so it is testable without video.
- Simplifications get a plain comment naming the limit and the upgrade path. No tool or plugin names in comments.
- Commits carry no `Co-Authored-By` line and no AI attribution.
- Keep `tracker.py` as one file until ~400 lines, then split along layer lines (`detect.py`, `kalman.py`, `events.py`, `viz.py`).

## 10. Git

Private repo, `github.com/VarnoxL/ball-tracker`, branch `main`.

```
903782b Add video input, YOLO detection, and overlay rendering
df0d863 Add CLAUDE.md with pipeline, conventions, and design decisions
e5fb538 Initial commit: README and .gitignore
```

Git warns `LF will be replaced by CRLF` on every commit. Harmless on Windows.
