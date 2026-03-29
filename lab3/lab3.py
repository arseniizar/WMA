import argparse
import sys

import cv2
import numpy as np


def build_detector(method: str):
    m = method.upper()
    if m == "ORB":
        return cv2.ORB_create(nfeatures=2000)
    if m == "BRISK":
        return cv2.BRISK_create()
    if m == "SIFT":
        return cv2.SIFT_create()
    raise ValueError(f"unknown method: {method}")


def build_matcher(method: str):
    m = method.upper()
    if m in ("ORB", "BRISK"):
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def extract(gray: np.ndarray, detector):
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs


def good_matches(ref_desc: np.ndarray, frm_desc: np.ndarray, matcher, ratio: float = 0.75):
    if ref_desc is None or frm_desc is None:
        return []
    if len(ref_desc) < 2 or len(frm_desc) < 2:
        return []
    pairs = matcher.knnMatch(ref_desc, frm_desc, k=2)
    kept = []
    for pair in pairs:
        if len(pair) == 2:
            a, b = pair
            if a.distance < ratio * b.distance:
                kept.append(a)
    return kept


def locate(ref_kps, frm_kps, matches, ref_shape, min_matches: int = 15):
    if len(matches) < min_matches:
        return None, None

    src = np.float32([ref_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([frm_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        return None, None

    inliers = int(mask.sum())
    if inliers < min_matches:
        return None, None

    h, w = ref_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)

    area = cv2.contourArea(projected)
    if area < 1000:
        return None, None

    return projected, mask


def draw_box(frame: np.ndarray, corners) -> np.ndarray:
    out = frame.copy()
    if corners is None:
        return out
    pts = np.int32(corners)
    cv2.polylines(out, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def draw_kp_matches(ref_bgr, ref_kps, frm_bgr, frm_kps, matches, limit: int = 50):
    shown = sorted(matches, key=lambda m: m.distance)[:limit]
    return cv2.drawMatches(
        ref_bgr, ref_kps,
        frm_bgr, frm_kps,
        shown, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )


def put(img, text, pos, color=(220, 220, 220)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,  1, cv2.LINE_AA)


def process_image(ref_path: str, test_path: str, method: str):
    ref_bgr  = cv2.imread(ref_path)
    test_bgr = cv2.imread(test_path)

    if ref_bgr is None:
        print(f"ERROR: cannot open {ref_path}", file=sys.stderr); return
    if test_bgr is None:
        print(f"ERROR: cannot open {test_path}", file=sys.stderr); return

    detector = build_detector(method)
    matcher  = build_matcher(method)

    ref_kps,  ref_desc  = extract(cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2GRAY), detector)
    test_kps, test_desc = extract(cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY), detector)

    matches        = good_matches(ref_desc, test_desc, matcher)
    corners, _     = locate(ref_kps, test_kps, matches, ref_bgr.shape[:2])

    result     = draw_box(test_bgr, corners)
    match_vis  = draw_kp_matches(ref_bgr, ref_kps, test_bgr, test_kps, matches)

    found = corners is not None
    put(result, f"matches: {len(matches)}  {'FOUND' if found else 'NOT FOUND'}",
        (10, 30), (0, 255, 0) if found else (0, 0, 220))

    cv2.imshow("reference", ref_bgr)
    cv2.imshow("result",    result)
    cv2.imshow("matches",   match_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(ref_path: str, video_path: str, method: str):
    ref_bgr = cv2.imread(ref_path)
    if ref_bgr is None:
        print(f"ERROR: cannot open {ref_path}", file=sys.stderr); return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}", file=sys.stderr); return

    detector = build_detector(method)
    matcher  = build_matcher(method)

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_kps, ref_desc = extract(ref_gray, detector)

    fps   = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 1 else 20

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frm_kps, frm_desc = extract(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detector)
        matches            = good_matches(ref_desc, frm_desc, matcher)
        corners, _         = locate(ref_kps, frm_kps, matches, ref_gray.shape)

        result = draw_box(frame, corners)
        found  = corners is not None
        put(result, f"matches: {len(matches)}  {'FOUND' if found else 'NOT FOUND'}",
            (10, 30), (0, 255, 0) if found else (0, 0, 220))

        cv2.imshow("tracking", result)
        if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--image",  default=None)
    p.add_argument("--video",  default=None)
    p.add_argument("--method", default="ORB")
    args = p.parse_args()

    if args.image is None and args.video is None:
        print("ERROR: provide --image or --video", file=sys.stderr)
        return 1

    if args.image:
        process_image(args.reference, args.image, args.method)
    if args.video:
        process_video(args.reference, args.video, args.method)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())