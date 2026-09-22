"""Checks for the pure helpers and the Kalman filter in tracker.py.
Run: python test_tracker.py"""

from tracker import MAX_MISSED, REACQUIRE_N, BallKalman, best_ball, center, foot_point

# feet, not the middle of the body
assert foot_point((10, 20, 30, 80)) == (20, 80)
assert center((10, 20, 30, 80)) == (20, 50)

# highest confidence wins, and the result is the box center
assert best_ball([((0, 0, 10, 10), 0.2), ((100, 100, 110, 110), 0.9)]) == (105, 105, 0.9)
assert best_ball([((0, 0, 10, 10), 0.2)]) == (5, 5, 0.2)
assert best_ball([]) is None

# Kalman: straight line at (5, 2) px/frame, frames 10-20 blanked.
# The filter coasts on its velocity estimate and stays near the truth.
kf = BallKalman()
for t in range(40):
    truth = (100 + 5 * t, 200 + 2 * t)
    x, y, measured = kf.step(None if 10 <= t <= 20 else truth)
    assert measured == (not 10 <= t <= 20)
    if t >= 10:
        assert abs(x - truth[0]) < 3 and abs(y - truth[1]) < 3, (t, x, y, truth)

# Kalman: a detection 500 px from the prediction is gated out and leaves
# the state exactly where a pure prediction would put it.
kf, ref = BallKalman(), BallKalman()
for t in range(10):
    kf.step((100 + 5 * t, 200))
    ref.step((100 + 5 * t, 200))
ref.predict()
x, y, measured = kf.step((600 + 5 * 10, 200))
assert not measured
assert (x, y) == (ref.x[0], ref.x[1])
assert kf.missed == 1

# Kalman: coasts for MAX_MISSED empty frames, then reports lost,
# and the next detection starts a fresh track at that point.
kf = BallKalman()
kf.step((100, 100))
for _ in range(MAX_MISSED):
    assert kf.step(None) is not None
assert kf.step(None) is None
assert kf.step((900, 500)) == (900, 500, True)

# Kalman: a kick. Ball sits still, then moves at 30 px/frame from 110 px away.
# Gated out at first, then the chain of agreeing detections restarts the track.
kf = BallKalman()
for _ in range(10):
    kf.step((500, 500))
for k in range(REACQUIRE_N):
    x, y, measured = kf.step((610 + 30 * k, 500))
    assert measured == (k == REACQUIRE_N - 1)
assert (x, y) == (610 + 30 * (REACQUIRE_N - 1), 500)

# Kalman: scattered false positives never form a chain, so the track holds.
kf = BallKalman()
for _ in range(10):
    kf.step((500, 500))
for fp in [(100, 100), (1500, 900), (100, 900)]:
    assert kf.step(fp)[2] is False
assert kf.step((500, 500))[2] is True

print("ok")
