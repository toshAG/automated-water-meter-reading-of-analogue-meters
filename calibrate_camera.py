"""
Camera colour calibration tool for the Raspberry Pi Camera.

Captures a grid of photos at different manual white-balance gain settings
(red_gain, blue_gain), saves each one individually, and builds a single
labeled contact sheet so you can visually compare all of them at once and
pick whichever setting looks closest to correct colour.

This does NOT require guessing numbers in advance -- it sweeps a sensible
range automatically. If none of the results look acceptable, that itself is
useful evidence the problem is a NoIR module's missing IR-cut filter
(a hardware issue no gain setting can fully fix), rather than a pure
white-balance problem.

Usage (camera must not be in use by any other program, e.g. close the
Analog Meter Reader app first):

    python3 calibrate_camera.py --out_dir ~/camera_calibration

Requirements:
    sudo apt install -y python3-picamera2
    pip install opencv-python-headless pillow --break-system-packages
"""

import argparse
import itertools
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

try:
    from gpiozero import LED
except ImportError:
    LED = None


def capture_with_gains(picam2, red_gain: float, blue_gain: float) -> np.ndarray:
    """Disable auto white balance, set manual gains, capture one frame (BGR)."""
    picam2.set_controls({"AwbEnable": False, "ColourGains": (red_gain, blue_gain)})
    time.sleep(0.6)  # let the new gains actually take effect before capturing
    frame_rgb = picam2.capture_array()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def build_contact_sheet(entries, thumb_size=260, cols=4):
    """entries: list of (label, bgr_image). Returns a PIL Image contact sheet."""
    rows = (len(entries) + cols - 1) // cols
    label_height = 28
    cell_w, cell_h = thumb_size, thumb_size + label_height
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (label, bgr) in enumerate(entries):
        row, col = divmod(i, cols)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(rgb).resize((thumb_size, thumb_size))
        x, y = col * cell_w, row * cell_h
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y + thumb_size, x + cell_w, y + cell_h], fill=(20, 20, 20))
        draw.text((x + 6, y + thumb_size + 6), label, fill=(255, 255, 255), font=font)

    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.expanduser("~/camera_calibration"),
                         help="Folder to save individual photos and the contact sheet into")
    parser.add_argument("--red_gains", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0],
                         help="Red gain values to test")
    parser.add_argument("--blue_gains", type=float, nargs="+", default=[1.0, 1.8, 2.5, 3.2],
                         help="Blue gain values to test")
    parser.add_argument("--led_pin", type=int, default=None,
                         help="GPIO pin of an LED to turn on during every capture in this sweep. "
                              "Worth testing if pure gain adjustment (no LED) failed to fix a colour cast -- "
                              "a strong controlled light source can succeed where gain correction alone can't.")
    args = parser.parse_args()

    if Picamera2 is None:
        print("picamera2 is not installed. Install it with:")
        print("  sudo apt install -y python3-picamera2")
        return

    led = None
    if args.led_pin is not None:
        if LED is None:
            print("gpiozero is not installed -- install it with: pip install gpiozero --break-system-packages")
            return
        led = LED(args.led_pin)
        led.on()
        print(f"LED on GPIO{args.led_pin} turned on for this calibration run.")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Opening camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    time.sleep(2)  # let auto-exposure settle before we start overriding AWB

    combos = list(itertools.product(args.red_gains, args.blue_gains))
    print(f"Capturing {len(combos)} images (red x blue gain combinations)...")

    entries = []
    for red_gain, blue_gain in combos:
        label = f"R{red_gain:.1f} B{blue_gain:.1f}"
        print(f"  capturing {label} ...")
        frame_bgr = capture_with_gains(picam2, red_gain, blue_gain)
        filename = os.path.join(args.out_dir, f"gains_r{red_gain:.1f}_b{blue_gain:.1f}.jpg")
        cv2.imwrite(filename, frame_bgr)
        entries.append((label, frame_bgr))

    picam2.stop()
    print("Camera closed.")
    if led is not None:
        led.off()

    sheet = build_contact_sheet(entries)
    sheet_path = os.path.join(args.out_dir, "contact_sheet.jpg")
    sheet.save(sheet_path, quality=90)

    print("\nDone.")
    print(f"Individual photos saved in: {args.out_dir}")
    print(f"Comparison sheet saved as:  {sheet_path}")
    print("\nOpen the comparison sheet and find the tile with the most natural-looking colour.")
    print("Its label (e.g. 'R1.5 B2.5') gives you the red_gain and blue_gain values to use.")
    print("\nOnce you've picked one, apply it in camera.py by passing it to CameraManager, e.g.:")
    print("  CameraManager(manual_awb_gains=(1.5, 2.5))")
    print("\nIf every tile still looks tinted, this strongly suggests a NoIR camera module")
    print("(missing IR-cut filter) rather than a fixable white-balance problem --")
    print("see the deployment report's camera troubleshooting notes for hardware options.")


if __name__ == "__main__":
    main()
