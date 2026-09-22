"""Soccer ball + player tracking. Layers 1, 2, 3 and 6: input, detection,
Kalman ball tracking, overlay.

Layer 4 (events) is not here yet — see CLAUDE.md.
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

# Kalman ball filter. Units are px and frames (dt = 1).
GATE_PX = 80          # max distance from the prediction for a detection to count
GATE_GROW_PX = 10     # gate widens this much per missed frame while coasting
MAX_MISSED = 35       # coast this many frames before the track is lost; worst real gap was 31
REACQUIRE_N = 3       # this many gated-out detections in a row, each within GATE_PX
                      # of the last, means a kick: restart the track on them
KF_Q = 1.0            # process noise: random acceleration variance, px^2/frame^4
KF_R = 9.0            # measurement noise variance, px^2 (detections jitter ~3 px)
KF_P0_VEL = 400.0     # initial velocity variance, (px/frame)^2: a new track's speed is unknown

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

    One ball per frame, best confidence wins. Gating against the predicted
    position happens in BallKalman.step.
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
    # ball track IDs unused; BallKalman does the tracking
    return best_ball(list(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())))


# --- layer 3: ball tracking ---
class BallKalman:
    """Constant-velocity Kalman filter over the ball, state [px, py, vx, vy].

    step(z) per frame, z = (x, y) or None. Returns (x, y, measured) or None
    while there is no track.
    """

    F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    # random-acceleration noise for dt = 1: G G^T * q with G = [1/2, 1/2, 1, 1]
    Q = KF_Q * np.array([[0.25, 0, 0.5, 0],
                         [0, 0.25, 0, 0.5],
                         [0.5, 0, 1, 0],
                         [0, 0.5, 0, 1]])
    R = KF_R * np.eye(2)

    def __init__(self):
        self.x = None   # None = no track
        self.P = None
        self.missed = 0
        self.cand = None   # last gated-out detection
        self.cand_n = 0    # how many in a row agree with each other

    def start(self, z):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([KF_R, KF_R, KF_P0_VEL, KF_P0_VEL])
        self.missed = 0
        self.cand, self.cand_n = None, 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = np.asarray(z, dtype=float) - self.H @ self.x   # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def in_gate(self, z):
        # Euclidean pixel gate, widened while coasting because the prediction
        # drifts. Upgrade path: Mahalanobis distance on S, which scales with P.
        gate = GATE_PX + GATE_GROW_PX * self.missed
        return np.hypot(z[0] - self.x[0], z[1] - self.x[1]) <= gate

    def step(self, z):
        if self.x is not None:
            self.predict()
            if z is not None and self.in_gate(z):
                self.update(z)
                self.missed = 0
                self.cand, self.cand_n = None, 0
                return (float(self.x[0]), float(self.x[1]), True)
            if z is not None:
                # A kick breaks constant velocity, so the real ball lands outside
                # the gate. False positives jump around; a kicked ball forms a chain.
                near = self.cand is not None and np.hypot(
                    z[0] - self.cand[0], z[1] - self.cand[1]) <= GATE_PX
                self.cand, self.cand_n = z, self.cand_n + 1 if near else 1
                if self.cand_n >= REACQUIRE_N:
                    self.start(z)
                    return (float(z[0]), float(z[1]), True)
            self.missed += 1
            if self.missed <= MAX_MISSED:
                return (float(self.x[0]), float(self.x[1]), False)
            self.x = None   # lost; a detection this same frame restarts below
        if z is None:
            return None
        self.start(z)
        return (float(z[0]), float(z[1]), True)


# --- layer 6: visualization ---
def draw(frame, players, track, trail, idx):
    for tid, box in players:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_COLOR, 2)
        cv2.putText(frame, str(tid), (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLAYER_COLOR, 2)

    if len(trail) > 1:
        cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, BALL_COLOR, 2)
    state = "-"
    if track is not None:
        x, y, measured = track
        # filled = measured this frame, ring = predicted through a dropout
        cv2.circle(frame, (int(x), int(y)), 8, BALL_COLOR, -1 if measured else 2)
        state = "Y" if measured else "P"

    cv2.putText(frame, f"frame {idx}  players {len(players)}  ball {state}",
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
    kf = BallKalman()

    idx = hits = player_total = gap = longest_gap = 0
    tracked = rejected = lost_gap = longest_lost = 0
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
        else:
            gap += 1
            longest_gap = max(longest_gap, gap)
        player_total += len(players)

        track = kf.step(ball[:2] if ball else None)
        if ball is not None and not (track and track[2]):
            rejected += 1
        if track is not None:
            tracked += 1
            lost_gap = 0
            trail.append(track[:2])
        else:
            lost_gap += 1
            longest_lost = max(longest_lost, lost_gap)
            trail.clear()   # no line jumping across the pitch on re-acquire

        draw(frame, players, track, trail, idx)
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
    print(f"ball tracked in {tracked}/{idx} frames ({100 * tracked / max(idx, 1):.1f}%), "
          f"{rejected} detections gated out, longest untracked run {longest_lost} frames")
    print(f"players/frame: avg {player_total / max(idx, 1):.1f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
