# LAB3 — Object Recognition & Tracking via Descriptors

Detects and tracks a reference object in images and video using local feature descriptors (ORB, BRISK, or SIFT).

## Usage

```bash
# single image
python lab3.py --reference saw1.jpg --image test.jpg --method ORB

# video
python lab3.py --reference saw1.jpg --video sawmovie.mp4 --method ORB
```
## How it works

Keypoints and descriptors are extracted from the reference image and each test frame using the chosen detector. Descriptor pairs are matched with a brute-force matcher and filtered via Lowe's ratio test (threshold 0.75). If enough matches survive (≥15), a homography is estimated with RANSAC — the object is considered found only when the inlier count and projected area both pass a minimum threshold, avoiding false positives in frames where the object is absent. The detected region is drawn as a green polygon; frames with no reliable detection are left unmarked.