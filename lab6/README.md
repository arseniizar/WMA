# LAB6 — YOLO Algorithm Object Detection

Implements object detection, bounding box extraction, filtering, and model fine-tuning utilizing the **YOLOv8**
architecture.

## Requirements

Ensure your environment is set up and install the required library:

```bash
pip install ultralytics opencv-python numpy
```

## Usage Examples

**1. Detect objects in a static image:**

```bash
python lab6.py --model yolov8n.pt --image test.jpg
```

*(Note: If `yolov8n.pt` is not present, the ultralytics library will automatically download it on the first run).*

**2. Detect objects in a video file:**

```bash
python lab6.py --model yolov8n.pt --video my_video.mp4 --confidence 0.6
```

**3. Detect objects using live camera feed:**

```bash
python lab6.py --model yolov8n.pt --camera
```

**4. View Dataset Annotation Help:**

```bash
python lab6.py --model yolov8n.pt --show-annotation-help
```

**5. Fine-Tune (Train) a Custom Model:**
(Make sure you have an exported YOLO dataset from Roboflow or CVAT)

```bash
python lab6.py --model yolov8n.pt --train --train-data dataset/data.yaml --epochs 20
```

**6. Compare Base vs Fine-Tuned Model:**

```bash
python lab6.py --model yolov8n.pt --compare --trained-model runs/detect/train/weights/best.pt --image custom_test.jpg
```