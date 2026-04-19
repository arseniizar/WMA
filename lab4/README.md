# LAB4 — Optical Flow Tracking

Tracks feature points across video frames using Shi-Tomasi detection and Lucas-Kanade optical flow. Draws colored motion trajectories for each tracked point.

## Usage

```bash
python lab4.py --video film.mp4
```

## How it works

Good features are detected in the first frame using Shi-Tomasi corner detection (`goodFeaturesToTrack`). Each subsequent frame is compared to the previous one using the Lucas-Kanade pyramidal optical flow method (`calcOpticalFlowPyrLK`), which estimates where each point moved. Only points with a valid tracking status are kept — if fewer than 5 survive, features are re-detected and trajectories reset. Motion paths are accumulated on a persistent mask and overlaid on the video output.
