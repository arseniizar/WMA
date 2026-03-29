import cv2
import numpy as np
import argparse
import sys


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    return p.parse_args()


def red_mask(hsv):
    m1 = cv2.inRange(hsv, np.array([0,   120,  70], dtype=np.uint8),
                     np.array([10,  255, 255], dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array([160, 120,  70], dtype=np.uint8),
                     np.array([180, 255, 255], dtype=np.uint8))
    combined = cv2.bitwise_or(m1, m2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    return combined


def pick_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def contour_props(cnt, min_area=500):
    if cv2.contourArea(cnt) < min_area:
        return None
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    _, r = cv2.minEnclosingCircle(cnt)
    return cx, cy, int(r)


def overlay_bar(frame, dev, fw):
    fh      = frame.shape[0]
    bh      = 16
    by      = fh - 50
    x0      = 10
    x1      = fw - 10
    total_w = x1 - x0
    mid     = x0 + total_w // 2

    cv2.rectangle(frame, (x0, by), (x1, by + bh), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, by), (x1, by + bh), (130, 130, 130), 1)

    half    = total_w // 2
    clamped = max(min(dev, fw // 2), -(fw // 2))
    fill    = int(abs(clamped) / (fw // 2) * half)

    color = (60, 180, 60) if dev < 0 else (60, 60, 220)
    if dev < 0:
        cv2.rectangle(frame, (mid - fill, by + 2), (mid, by + bh - 2), color, -1)
    else:
        cv2.rectangle(frame, (mid, by + 2), (mid + fill, by + bh - 2), color, -1)

    cv2.line(frame, (mid, by - 2), (mid, by + bh + 2), (200, 200, 200), 1)

    tag   = "L" if dev < 0 else "R"
    label = f"dev: {abs(dev)}px {tag}"
    cv2.putText(frame, label, (x0, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def annotate(frame, props, fw, fh):
    cx, cy, r = props
    dev  = cx - fw // 2
    side = "L" if dev < 0 else "R"

    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 140, 255), -1)

    cv2.line(frame, (fw // 2, fh - 70), (fw // 2, fh - 20), (160, 160, 160), 1)

    info = f"x={cx} y={cy}  dev={dev}px {side}"
    tw, _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0], None
    cv2.putText(frame, info, (10, fh - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    overlay_bar(frame, dev, fw)


def annotate_mask(display, cnt, cx, cy, r):
    cv2.drawContours(display, [cnt], -1, (180, 180, 180), 2)
    cv2.circle(display, (cx, cy), r, (120, 120, 120), 1)
    cv2.circle(display, (cx, cy), 3, (255, 255, 255), -1)


def run():
    args = get_args()
    cap  = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: '{args.video}'")
        sys.exit(1)

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask",     cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        mask   = red_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

        vis  = frame.copy()
        mvis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cnt = pick_contour(mask)
        if cnt is not None:
            props = contour_props(cnt)
            if props is not None:
                cx, cy, r = props
                annotate(vis, props, fw, fh)
                annotate_mask(mvis, cnt, cx, cy, r)
        else:
            cv2.putText(vis, "not detected", (10, fh - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 220), 1, cv2.LINE_AA)

        cv2.imshow("Tracking", vis)
        cv2.imshow("Mask",     mvis)

        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
