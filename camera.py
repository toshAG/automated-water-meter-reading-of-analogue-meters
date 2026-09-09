"""
Camera management for the meter reader app.

Wraps picamera2 so the GUI can open/close the camera on demand without
restarting the whole application, and so desktop_app.py doesn't need to
know picamera2's API directly.

Also includes optional support for:
  - A controlled LED light source during capture (via gpiozero) -- a strong,
    consistent light source can overpower an ambient IR imbalance (e.g. on
    a NoIR camera module) that pure white-balance gain correction cannot
    fix on its own.
  - Fixed exposure/gain/colour-gain capture (matching values found through
    calibrate_camera.py), instead of relying on auto exposure/white-balance,
    for repeatable results frame to frame.
  - A standard post-capture correction pipeline (gray-world white balance +
    CLAHE contrast + gamma correction) as an alternative to the simpler
    channel-scaling correction.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # allows the module to be imported off-Pi for layout/testing

try:
    from gpiozero import LED
except ImportError:
    LED = None  # allows the module to be imported without gpiozero / without an LED wired


class CameraNotAvailableError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Post-capture image correction
# ---------------------------------------------------------------------------

class ImageCorrector:
    """Standard three-stage correction pipeline: gray-world white balance,
    then CLAHE local contrast enhancement, then gamma correction. Applied
    in that order, matching the reference calibration approach this was
    built from."""

    def __init__(self, gamma: float = 1.2, clahe_clip_limit: float = 2.0, clahe_tile_size: int = 8):
        self.gamma = gamma
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))
        self._gamma_lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        ).astype(np.uint8)

    @staticmethod
    def gray_world_white_balance(frame_bgr: np.ndarray) -> np.ndarray:
        """Assumes the average scene colour should be neutral gray, and
        scales each channel so that holds. A simple, well-established
        white-balance technique -- distinct from (and complementary to)
        the camera's own AWB, since it runs on the captured pixels rather
        than the sensor's own light metering."""
        img = frame_bgr.astype(np.float32)
        b, g, r = cv2.split(img)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        avg = (avg_b + avg_g + avg_r) / 3.0
        # Guard against a pure-black channel average (division by zero)
        b *= avg / avg_b if avg_b > 0 else 1.0
        g *= avg / avg_g if avg_g > 0 else 1.0
        r *= avg / avg_r if avg_r > 0 else 1.0
        balanced = cv2.merge((b, g, r))
        return np.clip(balanced, 0, 255).astype(np.uint8)

    def apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization on the lightness
        channel only (via LAB colour space), so contrast improves without
        distorting colour further."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.LUT(frame_bgr, self._gamma_lut)

    def correct(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Runs the full pipeline: white balance -> contrast -> gamma."""
        balanced = self.gray_world_white_balance(frame_bgr)
        balanced = self.apply_clahe(balanced)
        balanced = self.apply_gamma(balanced)
        return balanced


# ---------------------------------------------------------------------------
# Optional LED illumination
# ---------------------------------------------------------------------------

class LedIllumination:
    """Controls an LED wired to a GPIO pin, used as a controlled light
    source during capture. No-ops safely if gpiozero isn't installed or
    no pin was configured, so the rest of the app works fine without one."""

    def __init__(self, pin: Optional[int]):
        self._led = None
        if pin is not None and LED is not None:
            self._led = LED(pin)

    @property
    def available(self) -> bool:
        return self._led is not None

    def on(self):
        if self._led is not None:
            self._led.on()

    def off(self):
        if self._led is not None:
            self._led.off()

    def __enter__(self):
        self.on()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.off()


# ---------------------------------------------------------------------------
# Camera manager
# ---------------------------------------------------------------------------

class CameraManager:
    def __init__(
        self,
        size: Tuple[int, int] = (1640, 1232),
        # Fixed manual capture settings (leave any as None to keep that
        # setting on auto). Values found via calibrate_camera.py should go
        # here for repeatable results.
        exposure_time_us: Optional[int] = None,
        analogue_gain: Optional[float] = None,
        colour_gains: Optional[Tuple[float, float]] = None,   # (red_gain, blue_gain)
        # Optional LED light source, controlled during each capture.
        led_pin: Optional[int] = None,
        # Optional post-capture correction. Prefer this over the older
        # simple channel_correction below when a full white-balance +
        # contrast + gamma pass is wanted.
        corrector: Optional[ImageCorrector] = None,
        # Older, simpler correction: a flat per-channel multiply. Kept for
        # backward compatibility -- corrector (above) is the better option
        # for new setups.
        channel_correction: Optional[Tuple[float, float, float]] = None,
    ):
        self._picam2 = None
        self.is_open = False
        self.size = size
        self.exposure_time_us = exposure_time_us
        self.analogue_gain = analogue_gain
        self.colour_gains = colour_gains
        self.led = LedIllumination(led_pin)
        self.corrector = corrector
        self.channel_correction = channel_correction

    def open(self):
        if Picamera2 is None:
            raise CameraNotAvailableError(
                "picamera2 is not installed. Install it with: "
                "sudo apt install -y python3-picamera2"
            )
        if self.is_open:
            return

        try:
            self._picam2 = Picamera2()
            config = self._picam2.create_still_configuration(main={"size": self.size})
            self._picam2.configure(config)
            self._picam2.start()
        except Exception as e:
            # A previous session that didn't clean up, or another program
            # (e.g. rpicam-still) holding the camera, raises a RuntimeError
            # here -- surface it as a clear message instead of letting it
            # crash the background thread silently.
            self._picam2 = None
            self.is_open = False
            raise CameraNotAvailableError(
                f"Could not open the camera: {e}. "
                "It may already be in use by another program or a previous "
                "unclosed session. Try: pkill -9 -f desktop_app.py, then retry."
            )

        time.sleep(2)  # let auto-exposure/white-balance settle before we (optionally) override it

        manual_controls = {}
        if self.colour_gains is not None:
            manual_controls["AwbEnable"] = False
            manual_controls["ColourGains"] = self.colour_gains
        if self.exposure_time_us is not None and self.analogue_gain is not None:
            manual_controls["AeEnable"] = False
            manual_controls["ExposureTime"] = self.exposure_time_us
            manual_controls["AnalogueGain"] = self.analogue_gain
        if manual_controls:
            self._picam2.set_controls(manual_controls)
            time.sleep(1)  # let the new settings actually take effect

        self.is_open = True

    def close(self):
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            self._picam2 = None
        self.is_open = False

    def capture_frame_bgr(self) -> np.ndarray:
        """Capture a single frame and return it as a BGR numpy array (OpenCV
        convention). Turns the LED on for the capture (if configured) and
        applies whichever post-capture correction is configured."""
        if not self.is_open or self._picam2 is None:
            raise CameraNotAvailableError("Camera is not open — call open() first")

        with self.led:  # no-ops if no LED configured
            frame_rgb = self._picam2.capture_array()

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if self.corrector is not None:
            frame_bgr = self.corrector.correct(frame_bgr)
        elif self.channel_correction is not None:
            b_scale, g_scale, r_scale = self.channel_correction
            frame_bgr = frame_bgr.astype(np.float32)
            frame_bgr[:, :, 0] *= b_scale
            frame_bgr[:, :, 1] *= g_scale
            frame_bgr[:, :, 2] *= r_scale
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)

        return frame_bgr
