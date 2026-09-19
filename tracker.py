"""Soccer ball + player tracking. Layers 1, 2 and 6: input, detection, overlay.

Layers 3 (Kalman) and 4 (events) are not here yet — see CLAUDE.md.
Usage: python tracker.py clip.mp4 out.mp4
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# --- tuning knobs (CLAUDE.md 5: every magic number lives here) ---
PLAYER_MODEL = "yolov8m.pt"   # COCO 'person' is plenty for players
BALL_MODEL = "ball.pt"        # soccer fine-tune; COCO 'sports ball' only hit 22% here
IMGSZ = 1280          # the ball is only a few px wide, so 640 misses it
CONF_BALL = 0.20
CONF_PLAYER = 0.35
TRAIL = 30            # frames of ball trail to draw
PERSON_CLS = 0

BALL_COLOR = (0, 255, 255)
PLAYER_COLOR = (0, 200, 0)


# --- pure helpers (no cv2, so test_tracker.py can import them) ---
def foot_point(box):
    """Bottom-center of a box: where the player meets the ground."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def best_ball(dets):
    """Highest-confidence ball detection as (x, y, conf), or None.

    One ball per frame, best confidence wins. Gating against a predicted
    position comes with the Kalman filter in step 2.
    """
    if not dets:
        return None
    box, conf = max(dets, key=lambda d: d[1])
    x, y = center(box)
    return (x, y, conf)


# --- layer 2: detection ---
def detect_players(model, frame):
    """Frame in, list of (track_id, box) out. IDs come from ByteTrack."""
    res = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLS],
        imgsz=IMGSZ,
        conf=CONF_PLAYER,
        verbose=False,
    )[0]
    if res.boxes is None or res.boxes.id is None:
        return []
    return [
        (tid, box)
        for box, tid in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist())
    ]


def detect_ball(model, frame):
    """Frame in, (x, y, conf) or None out. Single-class soccer-ball model."""
    res = model.predict(frame, imgsz=IMGSZ, conf=CONF_BALL, verbose=False)[0]
    if res.boxes is None:
        return None
    # ball track IDs unused; the Kalman filter replaces them in step 2
    return best_ball(list(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())))


# --- layer 6: visualization ---
def draw(frame, players, ball, trail, idx):
    for tid, box in players:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_COLOR, 2)
        cv2.putText(frame, str(tid), (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLAYER_COLOR, 2)

    if len(trail) > 1:
        cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, BALL_COLOR, 2)
    if ball is not None:
        x, y, conf = ball
        cv2.circle(frame, (int(x), int(y)), 8, BALL_COLOR, -1)
        cv2.putText(frame, f"{conf:.2f}", (int(x) + 12, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 1)

    cv2.putText(frame, f"frame {idx}  players {len(players)}  ball {'Y' if ball else '-'}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# --- layer 1 + main loop ---
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)  # always from the file, never hardcoded
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.input}: {w}x{h} @ {fps:.2f}fps, {total} frames")

    player_model = YOLO(PLAYER_MODEL)
    ball_model = YOLO(BALL_MODEL)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail = deque(maxlen=TRAIL)

    idx = hits = player_total = gap = longest_gap = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        players = detect_players(player_model, frame)
        ball = detect_ball(ball_model, frame)
        if ball is not None:
            hits += 1
            gap = 0
            trail.append((ball[0], ball[1]))
        else:
            gap += 1
            longest_gap = max(longest_gap, gap)
        player_total += len(players)

        draw(frame, players, ball, trail, idx)
        out.write(frame)

        idx += 1
        if idx % 30 == 0:
            print(f"  {idx}/{total} frames", end="\r", flush=True)

    cap.release()
    out.release()

    elapsed = time.time() - start
    print(f"\n{idx} frames in {elapsed:.1f}s ({idx / elapsed:.1f} fps)")
    print(f"ball detected in {hits}/{idx} frames ({100 * hits / max(idx, 1):.1f}%)")
    print(f"longest gap: {longest_gap} frames ({longest_gap / fps:.1f}s)")
    print(f"players/frame: avg {player_total / max(idx, 1):.1f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
