import argparse
import sys

import cv2
import numpy as np


FEATURE_PARAMS = dict(
    maxCorners  = 100,
    qualityLevel = 0.3,
    minDistance  = 7,
    blockSize    = 7
)

LK_PARAMS = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

COLORS = np.random.randint(0, 255, (100, 3))


def load_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: cannot open '{path}'", file=sys.stderr)
        sys.exit(1)
    return cap


def detect_features(gray):
    return cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)


def track_points(prev_gray, curr_gray, prev_pts):
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **LK_PARAMS
    )
    return curr_pts, status


def draw_points(frame, pts):
    for i, pt in enumerate(pts):
        x, y = pt.ravel().astype(int)
        cv2.circle(frame, (x, y), 4, COLORS[i % 100].tolist(), -1)
    return frame


def draw_trajectories(mask, prev_pts, curr_pts):
    for i, (p, c) in enumerate(zip(prev_pts, curr_pts)):
        x0, y0 = p.ravel().astype(int)
        x1, y1 = c.ravel().astype(int)
        cv2.line(mask, (x0, y0), (x1, y1), COLORS[i % 100].tolist(), 2)
    return mask


def process_video(path: str):
    cap = load_video(path)

    ret, first = cap.read()
    if not ret:
        print("ERROR: cannot read first frame", file=sys.stderr)
        return

    prev_gray  = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    prev_pts   = detect_features(prev_gray)
    traj_mask  = np.zeros_like(first)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_pts, status = track_points(prev_gray, curr_gray, prev_pts)

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_curr) < 5:
            prev_gray = curr_gray.copy()
            prev_pts  = detect_features(prev_gray)
            traj_mask = np.zeros_like(frame)
            continue

        traj_mask = draw_trajectories(traj_mask, good_prev, good_curr)
        output    = draw_points(frame.copy(), good_curr)
        output    = cv2.add(output, traj_mask)

        cv2.putText(output, f"tracking: {len(good_curr)} pts", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("optical flow", output)

        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break

        prev_gray = curr_gray.copy()
        prev_pts  = good_curr.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    process_video(p.parse_args().video)


if __name__ == "__main__":
    main()
