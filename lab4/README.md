# LAB4 — Optical Flow Tracking

Tracks feature points across video frames using Shi-Tomasi detection and Lucas-Kanade optical flow. Draws colored motion trajectories for each tracked point.

## Demo Video

<img width="640" height="360" alt="ezgif-54a5f7f10b3eae8a" src="https://github.com/user-attachments/assets/3015c447-8b83-4af4-be41-60bbc9423d8b" />


## Usage

```bash
python lab4.py --video film.mp4
```

## How it works

Good features are detected in the first frame using Shi-Tomasi corner detection (`goodFeaturesToTrack`). Each subsequent frame is compared to the previous one using the Lucas-Kanade pyramidal optical flow method (`calcOpticalFlowPyrLK`), which estimates where each point moved. Only points with a valid tracking status are kept — if fewer than 5 survive, features are re-detected and trajectories reset. Motion paths are accumulated on a persistent mask and overlaid on the video output.
