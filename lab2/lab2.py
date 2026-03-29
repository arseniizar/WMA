import cv2 as cv
import numpy as np
import glob
import os
import sys


def load_images():
    names = sorted(glob.glob('tray*.jpg'))
    if not names:
        print('no tray images found')
        sys.exit(1)
    return names


def find_tray(img):
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv,
                      np.array([5,  80, 140], np.uint8),
                      np.array([16, 255, 255], np.uint8))

    k = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv.contourArea)


def detect_circles(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    raw = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.2, minDist=26,
        param1=100, param2=55,
        minRadius=12, maxRadius=40
    )
    return raw


def filter_circles(raw, img_gray, tray_cnt):
    if raw is None:
        return []

    kept = []
    for x, y, r in np.round(raw[0]).astype(int):
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv.circle(mask, (x, y), max(r - 4, 1), 255, -1)
        mean_val = cv.mean(img_gray, mask=mask)[0]

        in_tray = (tray_cnt is not None and
                   cv.pointPolygonTest(tray_cnt, (float(x), float(y)), False) >= 0)

        if in_tray and mean_val > 150:
            continue
        kept.append((x, y, r))
    return kept


def size_threshold(circles):
    radii = [r for _, _, r in circles]
    if len(radii) < 2:
        return 24
    z = np.float32(radii).reshape(-1, 1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv.kmeans(z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    return float(centers.mean())


def classify_and_draw(result, circles, tray_cnt, thresh):
    counts = {'small_in': 0, 'small_out': 0, 'big_in': 0, 'big_out': 0}

    for x, y, r in circles:
        big    = r >= thresh
        inside = (tray_cnt is not None and
                  cv.pointPolygonTest(tray_cnt, (float(x), float(y)), False) >= 0)

        if big and inside:
            counts['big_in']    += 1
        elif big:
            counts['big_out']   += 1
        elif inside:
            counts['small_in']  += 1
        else:
            counts['small_out'] += 1

        cv.circle(result, (x, y), r, (0, 255, 0) if big else (255, 0, 0), 2)
        cv.circle(result, (x, y), 3, (255, 255, 255) if inside else (255, 0, 255), -1)

    return counts


def draw_stats(result, tray_area, counts):
    lines = [
        f"Area: {tray_area:.1f}",
        f"Small & In: {counts['small_in']}",
        f"Small & Out: {counts['small_out']}",
        f"Big   & In: {counts['big_in']}",
        f"Big   & Out: {counts['big_out']}",
    ]
    for i, line in enumerate(lines):
        cv.putText(result, line, (20, 30 + i * 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)


def process(image_name):
    img = cv.imread(image_name)
    if img is None:
        print('cannot read:', image_name)
        return None

    result    = img.copy()
    tray_cnt  = find_tray(img)
    tray_area = cv.contourArea(tray_cnt) if tray_cnt is not None else 0.0

    if tray_cnt is not None:
        cv.drawContours(result, [tray_cnt], -1, (0, 255, 0), 3)

    gray    = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    raw     = detect_circles(img)
    circles = filter_circles(raw, gray, tray_cnt)
    thresh  = size_threshold(circles)
    counts  = classify_and_draw(result, circles, tray_cnt, thresh)

    draw_stats(result, tray_area, counts)
    return result


def run():
    os.makedirs('output', exist_ok=True)

    for name in load_images():
        result = process(name)
        if result is None:
            continue

        out = os.path.join('output', os.path.splitext(os.path.basename(name))[0] + '_result.jpg')
        cv.imwrite(out, result)
        cv.imshow('result', result)

        if cv.waitKey(0) & 0xFF == 27:
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    run()