"""Checks for the pure helpers in tracker.py. Run: python test_tracker.py"""

from tracker import best_ball, center, foot_point

# feet, not the middle of the body
assert foot_point((10, 20, 30, 80)) == (20, 80)
assert center((10, 20, 30, 80)) == (20, 50)

# highest confidence wins, and the result is the box center
assert best_ball([((0, 0, 10, 10), 0.2), ((100, 100, 110, 110), 0.9)]) == (105, 105, 0.9)
assert best_ball([((0, 0, 10, 10), 0.2)]) == (5, 5, 0.2)
assert best_ball([]) is None

print("ok")
