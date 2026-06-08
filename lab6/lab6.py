#!/usr/bin/env python3
"""
LAB6: YOLO Algorithm
Full starter template (exercise version)

Goals:
1. Load the YOLO model.
2. Load an image, video, or camera stream.
3. Perform object detection.
4. Read bounding boxes, classes, and confidence scores.
5. Draw detection results.
6. Prepare data for model fine-tuning.
7. Run YOLO model fine-tuning.
8. Compare the base model and the fine-tuned model.

Instructions:
- Complete the TODO sections one by one.
- First, run detection for a single image.
- Then add support for video and camera.
- Finally, prepare the fine-tuning part.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# --- LOGGER TO SAVE CONSOLE OUTPUT ---
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# TODO 1:
# - Import the YOLO class from the appropriate library
# Example:
# from ultralytics import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' library is not installed.")
    sys.exit(1)


# ============================================================
# TODO 2: Load the base model
# ============================================================

def load_model(model_path: str):
    """
    Loads a YOLO model from file.

    TODO:
    - Create the model object
    - Handle exceptions / loading errors
    - Return the model
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from '{model_path}'. Error: {e}")


# ============================================================
# TODO 3: Load image
# ============================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from file.

    TODO:
    - Use cv2.imread(...)
    - Check whether the image was loaded correctly
    - If not, raise an error or exception
    - Return the image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: cannot load image from '{image_path}'.")
    return image


# ============================================================
# TODO 4: Load video stream
# ============================================================

def load_stream(source: str):
    """
    Loads a video stream.

    TODO:
    - If source == "camera", use the default camera
    - Otherwise, use the path to the video file
    - Use cv2.VideoCapture(...)
    - Check whether the stream was opened correctly
    - Return the VideoCapture object
    """
    src = 0 if source == "camera" else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Error: cannot open video source: '{source}'")
    return cap


# ============================================================
# TODO 5: Run detection
# ============================================================

def run_detection(model, image: np.ndarray):
    """
    Runs the YOLO model on a single image / frame.

    TODO:
    - Call the model on the image
    - Set optional parameters, e.g. verbose=False
    - Return the raw model results
    """
    results = model(image, verbose=False)
    return results[0]


# ============================================================
# TODO 6: Read data from YOLO results
# ============================================================

def parse_results(results) -> List[Dict[str, Any]]:
    """
    Reads detection data from the model output.

    The returned list of detections should contain, for example:
    [
        {
            "box": [x1, y1, x2, y2],
            "class_id": 0,
            "confidence": 0.92
        },
        ...
    ]

    TODO:
    - Iterate through all detected objects
    - Read the bounding box
    - Read class_id
    - Read confidence
    - Return a list of detection dictionaries
    """
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            detections.append({"box": xyxy, "class_id": cls, "confidence": conf})
    return detections


# ============================================================
# TODO 7: Filter detections
# ============================================================

def filter_detections(
        detections: List[Dict[str, Any]],
        confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filters detections based on the confidence threshold.

    TODO:
    - Remove detections with confidence lower than confidence_threshold
    - Optionally: add additional class filtering
    - Return the filtered list of detections
    """
    return [d for d in detections if d["confidence"] >= confidence_threshold]


# ============================================================
# TODO 8: Get class names
# ============================================================

def get_class_names(model) -> Optional[Dict[int, str]]:
    """
    Retrieves class names from the model, if available.

    TODO:
    - Read the mapping index -> class name
    - Return the dictionary or None
    """
    if hasattr(model, 'names'):
        return model.names
    return None


# ============================================================
# TODO 9: Draw detections
# ============================================================

def draw_detections(
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        class_names: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    Draws bounding boxes and labels on the image.

    TODO:
    - Create a copy of the image
    - For each detection, draw a rectangle
    - Add text:
        * class name or class_id
        * confidence
    - Return the output image
    """
    output_image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        conf = det["confidence"]
        cls_id = det["class_id"]
        label = class_names[cls_id] if (class_names and cls_id in class_names) else str(cls_id)
        text = f"{label}: {conf:.2f}"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = max(y1 - 10, 20)
        cv2.putText(output_image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output_image


# ============================================================
# TODO 10: Diagnostic text
# ============================================================

def add_diagnostics(
        image: np.ndarray,
        num_detections: int,
        confidence_threshold: float
) -> np.ndarray:
    """
    Adds diagnostic text to the image.

    TODO:
    - Add the number of detected objects
    - Add the current confidence threshold
    - Return the output image
    """
    output_image = image.copy()
    diag_text = f"Detections: {num_detections} | Conf >= {confidence_threshold:.2f}"
    cv2.putText(output_image, diag_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return output_image


# ============================================================
# TODO 11: Check training data structure
# ============================================================

def check_dataset_structure(data_path: str) -> bool:
    """
    Checks the dataset structure for YOLO training.

    Expected example structure:
    dataset/
        images/
            train/
            val/
        labels/
            train/
            val/
        data.yaml

    TODO:
    - Check for the existence of directories:
        * images/train
        * images/val
        * labels/train
        * labels/val
    - Check for the existence of data.yaml
    - Return True/False
    """
    base_dir = os.path.dirname(data_path) if data_path.endswith('.yaml') else data_path
    required_paths = [
        os.path.join(base_dir, "images", "train"),
        os.path.join(base_dir, "images", "val"),
        os.path.join(base_dir, "labels", "train"),
        os.path.join(base_dir, "labels", "val")
    ]
    for p in required_paths:
        if not os.path.isdir(p):
            print(f"Missing required directory: {p}")
            return False

    yaml_path = data_path if data_path.endswith('.yaml') else os.path.join(base_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        print(f"Missing required file: {yaml_path}")
        return False
    return True


# ============================================================
# TODO 12: Annotation data information
# ============================================================

def print_annotation_info():
    """
    Displays information for the student about preparing training data.

    TODO:
    - Print recommended annotation tools:
        * LabelImg
        * CVAT
        * Makesense.ai
        * Roboflow
    - Remind the YOLO file format:
        class_id x_center y_center width height
    """
    print("\n--- YOLO Annotation Guide ---")
    print("Recommended Annotation Tools:")
    print("  1. LabelImg     - Simple desktop tool (recommended for offline work)")
    print("  2. CVAT         - Advanced web-based tool")
    print("  3. Makesense.ai - Browser-based, no installation required")
    print("  4. Roboflow     - Platform with augmentation and direct YOLO export\n")
    print("YOLO File Format:")
    print("  Each annotation is stored in a .txt file, one line per object.")
    print("  Format: <class_id> <x_center> <y_center> <width> <height>")
    print("  Values must be normalized to the range 0-1.")
    print("  Example: 0 0.523 0.441 0.210 0.317\n")


# ============================================================
# TODO 13: Fine-tune the model
# ============================================================

def fine_tune_model(
        model,
        data_path: str,
        num_epochs: int = 20,
        image_size: int = 640
):
    """
    Runs YOLO model training / fine-tuning.

    TODO:
    - Check the dataset structure
    - Run model training
    - Set parameters:
        * data
        * epochs
        * imgsz
    - Return training information or the path to the best model
    """
    print(f"Starting model fine-tuning with {num_epochs} epochs...")
    model.train(data=data_path, epochs=num_epochs, imgsz=image_size)
    best_model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
    print(f"\nTraining Complete! Best model saved at: {best_model_path}")
    return best_model_path


# ============================================================
# TODO 14: Load fine-tuned model
# ============================================================

def load_fine_tuned_model(model_path: str):
    """
    Loads the model saved after training.

    TODO:
    - Load the resulting model
    - Return the model
    """
    return load_model(model_path)


# ============================================================
# TODO 15: Compare models
# ============================================================

def compare_models(
        base_model,
        fine_tuned_model,
        image: np.ndarray,
        confidence_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compares the base model and the fine-tuned model on the same image.

    TODO:
    - Run detection for the base model
    - Run detection for the fine-tuned model
    - Parse and filter the results
    - Draw both outputs
    - Return the two output images
    """
    results_base = run_detection(base_model, image)
    det_base = filter_detections(parse_results(results_base), confidence_threshold)
    img_base = draw_detections(image, det_base, get_class_names(base_model))
    img_base = add_diagnostics(img_base, len(det_base), confidence_threshold)

    results_ft = run_detection(fine_tuned_model, image)
    det_ft = filter_detections(parse_results(results_ft), confidence_threshold)
    img_ft = draw_detections(image, det_ft, get_class_names(fine_tuned_model))
    img_ft = add_diagnostics(img_ft, len(det_ft), confidence_threshold)

    return img_base, img_ft


# ============================================================
# Image processing
# ============================================================

def process_image(model, image_path: str, confidence_threshold: float):
    """
    Processes a single image.
    """
    image = load_image(image_path)

    results = run_detection(model, image)
    detections = parse_results(results)
    detections = filter_detections(detections, confidence_threshold)

    class_names = get_class_names(model)

    output_image = draw_detections(image, detections, class_names)
    output_image = add_diagnostics(output_image, len(detections), confidence_threshold)

    cv2.imshow("Detection result - image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Video / camera processing
# ============================================================

def process_video(model, source: str, confidence_threshold: float):
    """
    Processes video or camera input.
    """
    cap = load_stream(source)
    class_names = get_class_names(model)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = run_detection(model, frame)
        detections = parse_results(results)
        detections = filter_detections(detections, confidence_threshold)

        output_frame = draw_detections(frame, detections, class_names)
        output_frame = add_diagnostics(output_frame, len(detections), confidence_threshold)

        cv2.imshow("Detection result - video", output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Comparison mode
# ============================================================

def run_comparison(
        base_model,
        fine_tuned_model,
        image_path: str,
        confidence_threshold: float
):
    """
    Runs a comparison of two models on a single image.
    """
    image = load_image(image_path)

    base_image, fine_tuned_image = compare_models(
        base_model,
        fine_tuned_model,
        image,
        confidence_threshold
    )

    cv2.imshow("Base model", base_image)
    cv2.imshow("Fine-tuned model", fine_tuned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Main function
# ============================================================

def main():
    # Activate Dual Logger
    sys.stdout = DualLogger("lab6_log.txt")

    parser = argparse.ArgumentParser(description="LAB6 - YOLO, full starter template")
    parser.add_argument("--model", required=True, help="Path to the base model, e.g. yolov8n.pt")
    parser.add_argument("--image", help="Path to the image")
    parser.add_argument("--video", help="Path to the video file")
    parser.add_argument("--camera", action="store_true", help="Use camera")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")

    parser.add_argument("--train", action="store_true", help="Run model fine-tuning")
    parser.add_argument("--train-data", help="Path to YOLO training data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--trained-model", help="Path to the fine-tuned model")
    parser.add_argument("--compare", action="store_true", help="Compare base and fine-tuned model")
    parser.add_argument("--show-annotation-help", action="store_true", help="Show annotation data information")

    args = parser.parse_args()

    # TODO 16:
    # - Load the base model
    print(f"Loading Base Model: '{args.model}'...")
    base_model = load_model(args.model)

    # TODO 17:
    # - If args.show_annotation_help == True,
    #   call print_annotation_info()
    if args.show_annotation_help:
        print_annotation_info()
        return

    # TODO 18:
    # - If args.train == True:
    #   * check args.train_data
    #   * run model fine-tuning
    #   * print a message when training is complete
    if args.train:
        if not args.train_data:
            print("Error: --train-data is required when using --train")
            return
        if not check_dataset_structure(args.train_data):
            print("Error: Dataset structure is invalid.")
            return
        fine_tune_model(base_model, args.train_data, args.epochs, args.imgsz)
        return

    # TODO 19:
    # - If args.compare == True:
    #   * check whether args.trained_model and args.image are provided
    #   * load the fine-tuned model
    #   * run model comparison
    if args.compare:
        if not args.trained_model or not args.image:
            print("Error: --compare requires both --trained-model and --image to be provided.")
            return
        print(f"Loading Fine-Tuned Model: '{args.trained_model}'...")
        fine_tuned_model = load_fine_tuned_model(args.trained_model)
        print(f"Comparing models on image: '{args.image}'...")
        run_comparison(base_model, fine_tuned_model, args.image, args.confidence)
        return

    # TODO 20:
    # - Add support for:
    #   * args.image
    #   * args.video
    #   * args.camera
    # - If no mode was provided, print an error message
    if args.image:
        print(f"Running detection on image: {args.image}")
        process_image(base_model, args.image, args.confidence)
    elif args.video:
        print(f"Running detection on video: {args.video}")
        process_video(base_model, args.video, args.confidence)
    elif args.camera:
        print("Running detection on camera stream. Press 'q' to quit.")
        process_video(base_model, "camera", args.confidence)
    else:
        print("Error: No execution mode selected.")
        print("Please provide --image, --video, --camera, --train, --compare, or --show-annotation-help")

    print("\nScript completed. Logs saved to 'lab6_log.txt'.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")
        sys.exit(0)
