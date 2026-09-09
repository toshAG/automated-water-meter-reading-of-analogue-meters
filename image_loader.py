"""
Image loading helper for the "Upload Image" feature.

Kept separate from desktop_app.py so the file-dialog / validation logic
doesn't clutter the GUI class.
"""

import os
from typing import Optional

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class ImageLoadError(RuntimeError):
    pass


def load_image_bgr(path: str) -> np.ndarray:
    """Loads an image file into a BGR numpy array (OpenCV convention). Raises ImageLoadError on failure."""
    if not os.path.exists(path):
        raise ImageLoadError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    image = cv2.imread(path)
    if image is None:
        raise ImageLoadError(
            f"Could not decode '{os.path.basename(path)}' — file may be corrupted or not a valid image"
        )
    return image


def tk_file_dialog_filetypes():
    """Filetypes list formatted for tkinter.filedialog.askopenfilename."""
    return [
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("JPEG", "*.jpg *.jpeg"),
        ("PNG", "*.png"),
        ("Bitmap", "*.bmp"),
        ("All files", "*.*"),
    ]
