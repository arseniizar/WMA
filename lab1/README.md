# LAB1 — Red Object Detection and Tracking

Detects and tracks a red object in a video using HSV color segmentation and moment-based localization. Displays live deviation from the frame center.

## Demo

## Usage

```bash
python lab1_object_detection.py --video sample.mp4
```

## How it works

Frames are converted to HSV and red is segmented across two hue ranges (0–10 and 160–180). The mask is cleaned with morphological opening and closing, then the largest contour is picked and its centroid computed via image moments. Deviation from the frame center is shown as text and a color-coded bar — green for left, blue for right.

## Windows

| Window | Content |
|--------|---------|
| `Tracking` | Original frame with circle, centroid, and deviation bar |
| `Mask` | Binary HSV mask after morphological cleanup |