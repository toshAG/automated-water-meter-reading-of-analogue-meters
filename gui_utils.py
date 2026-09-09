"""
Small stateless helpers used by desktop_app.py — kept separate so the
main GUI file stays focused on layout and event handling.
"""

from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


def draw_overlays(frame_bgr: np.ndarray, result: Optional[dict]) -> np.ndarray:
    """Draws bounding boxes for the odometer region, each pointer dial, and
    individual digits on a copy of the frame. Returns the annotated copy."""
    annotated = frame_bgr.copy()
    if result is None or not result.get("success"):
        return annotated

    ox1, oy1, ox2, oy2 = result["odometer_box"]
    cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 200, 0), 2)
    cv2.putText(annotated, "odometer", (ox1, max(oy1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    for i, (px1, py1, px2, py2) in enumerate(result.get("pointer_boxes", [])):
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 140, 255), 2)
        cv2.putText(annotated, f"dial{i+1}", (px1, max(py1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    for dx1, dy1, dx2, dy2 in result.get("digit_boxes", []):
        cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (255, 200, 0), 1)

    reading_text = f"Reading: {result['reading']}  (class conf: {result['classification_confidence']*100:.1f}%)"
    cv2.putText(annotated, reading_text, (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def crop_to_photoimage(crop: Optional[np.ndarray], target_width: int = 160) -> Optional[ImageTk.PhotoImage]:
    """Renders a small crop (odometer or a pointer dial) as a thumbnail PhotoImage.
    Returns None if the crop is missing or empty (e.g. detection failed)."""
    if crop is None or crop.size == 0:
        return None
    return frame_to_photoimage(crop, target_width)


def frame_to_photoimage(frame_bgr: np.ndarray, target_width: int) -> ImageTk.PhotoImage:
    """Converts a BGR OpenCV frame into a tkinter-displayable PhotoImage,
    scaled to target_width while preserving aspect ratio."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    target_width = max(target_width, 200)
    ratio = target_width / image.width
    target_height = max(int(image.height * ratio), 1)
    image = image.resize((target_width, target_height))
    return ImageTk.PhotoImage(image)


def format_history_entry(result: dict) -> str:
    timestamp = datetime.fromtimestamp(result["timestamp"]).strftime("%H:%M:%S")
    if not result.get("success"):
        return f"{timestamp}  FAILED — {result.get('message', 'detection failed')[:60]}..."
    odometer = "".join(str(d) for d in result["odometer_digits"])
    pointer = "".join(str(d) for d in result["pointer_digits"])
    rotation = result.get("rotation_applied", 0)
    rotation_str = f" rot={rotation}°" if rotation else ""
    return (f"{timestamp}  reading={result['reading']}  odometer={odometer}  "
            f"pointer={pointer}  det={result['detection_confidence']*100:.0f}% "
            f"cls={result['classification_confidence']*100:.0f}%{rotation_str}  "
            f"{result['inference_time_ms']:.0f}ms")
