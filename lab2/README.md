# LAB2 — Coin Detection & Tray Analysis

Detects a tray and coins in images, classifies coins by size (small/big) and position (inside/outside tray), and outputs annotated results.

## Usage

Place `tray*.jpg` images in the working directory, then run:

```bash
python lab2.py
```

Results are saved to `output/`. 

## How it works

The tray is found via HSV segmentation (hue 5–16) with morphological cleanup, then its contour area is computed. Coins are detected with Hough circles on a CLAHE-enhanced grayscale image — coins inside the tray with high mean brightness are filtered out as tray surface reflections. The remaining circles are split into small/big using k-means clustering on their radii, then classified by whether their center falls inside the tray contour.