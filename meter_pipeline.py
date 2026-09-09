"""
Core inference pipeline for the analog meter reading system.

Stage 1: YOLOv10 Global Detection      (general_yolo.onnx)   -> odometer ("dial") box + pointer-dial boxes
Stage 2: Region crop                                          -> odometer_crop, list of pointer dial crops
Stage 3: YOLOv10 Digit Localization     (odometer_yolo.onnx)  -> individual digit boxes within odometer_crop
Stage 4: Custom Digit CNN               (digit_cnn.onnx)      -> digit class index D_k (per odometer digit)
Stage 5: Custom Pointer CNN             (pointer_cnn.onnx)    -> digit class 0-9 for EACH pointer dial

Reading Synthesis: odometer digits + "." + pointer-dial digits (ordered by
the dial's own class name, e.g. pointer_reading1/2/3/4, not by pixel position
-- this stays correct even if the source photo is rotated or mirrored).

Matches classes by NAME (read from the ONNX model's embedded metadata),
not by hardcoded numeric IDs. Auto-detects each classifier's true input
size/layout directly from its ONNX file. Preprocessing intentionally does
NOT convert BGR->RGB (matches the original training pipeline).

UI-agnostic -- imported by desktop_app.py.
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # allows import/testing without ultralytics installed

_DIGIT_SUFFIX_RE = re.compile(r"(\d+)\s*$")


def _debug(stage: str, payload):
    """Stage-by-stage console logging -- makes it obvious exactly which
    stage failed (or was uncertain) instead of a single opaque pass/fail."""
    print(f"\n========== {stage} ==========")
    print(payload)


@dataclass
class PipelineConfig:
    models_dir: str
    digit_input_size: Tuple[int, int] = (28, 28)     # fallback only -- real size auto-detected from the model
    pointer_input_size: Tuple[int, int] = (64, 64)    # fallback only -- real size auto-detected from the model
    odometer_class_name: str = "dial"                 # the class name for the digital odometer window
    pointer_class_prefix: str = "pointer_reading"      # any class starting with this is a pointer dial
    detection_conf: float = 0.15                       # lowered from 0.4 -- analog dials detect at lower confidence
    expected_odometer_digits: Optional[int] = None     # set e.g. to 5 to flag readings with the wrong digit count
    auto_orient: bool = True                           # try 0/90/180/270 and auto-pick the best-detecting orientation
    verbose: bool = True                               # print stage-by-stage debug info to the console


class YoloDetector:
    """Stage 1 & Stage 3 -- Ultralytics ONNX backend handles NMS internally.
    Resolves each detection's class NAME (not just its numeric id) from the
    model's embedded metadata, so the pipeline can match by name."""

    def __init__(self, onnx_path: str):
        if YOLO is None:
            raise RuntimeError("ultralytics not installed — pip install ultralytics")
        self.model = YOLO(onnx_path, task="detect")
        self.class_names: Dict[int, str] = self.model.names  # e.g. {0: 'dial', 1: 'pointer_reading1', ...}

    def detect(self, image: np.ndarray, conf: float = 0.15):
        """Returns list of (x1, y1, x2, y2, cls_id, cls_name, confidence)."""
        results = self.model(image, conf=conf, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            cls_name = self.class_names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0].item())
            boxes.append((int(x1), int(y1), int(x2), int(y2), cls_id, cls_name, confidence))
        return boxes


class CnnClassifier:
    """Stage 4 & Stage 5 -- raw onnxruntime classifiers, no NMS needed.

    Auto-detects the model's real expected input size (and NHWC vs NCHW
    layout) from the ONNX file itself, instead of trusting a hardcoded
    guess -- this fixed an earlier 28x28-vs-64x64 mismatch crash."""

    def __init__(self, onnx_path: str, fallback_input_size: Tuple[int, int]):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        raw_shape = self.session.get_inputs()[0].shape  # e.g. [None, 64, 64, 3] or [None, 3, 64, 64]

        self.layout = "NHWC"
        self.input_size = fallback_input_size  # (width, height), used only if shape dims are dynamic/unreadable

        if len(raw_shape) == 4:
            dim1, dim2, dim3 = raw_shape[1], raw_shape[2], raw_shape[3]
            if isinstance(dim1, int) and dim1 in (1, 3) and not isinstance(dim3, int):
                # NCHW: [batch, channels, H, W]
                self.layout = "NCHW"
                if isinstance(dim2, int) and isinstance(dim3, int):
                    self.input_size = (dim3, dim2)  # (width, height)
            elif isinstance(dim1, int) and isinstance(dim2, int):
                # NHWC: [batch, H, W, channels]
                self.layout = "NHWC"
                self.input_size = (dim2, dim1)  # (width, height)

        print(f"  [{onnx_path.split('/')[-1]}] detected input shape={raw_shape} "
              f"-> using layout={self.layout}, size={self.input_size}")

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, self.input_size)
        # NOTE: intentionally NOT converting BGR->RGB here. The reference
        # training/inference pipeline feeds raw OpenCV BGR arrays directly,
        # so converting to RGB was swapping the red/blue channels -- this
        # was silently corrupting the red-needle color cue the pointer CNN
        # relies on, and was the main cause of pointer misclassification.
        normalized = resized.astype(np.float32) / 255.0
        if self.layout == "NCHW":
            normalized = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(normalized, axis=0)  # add batch dim

    def predict(self, crop: np.ndarray) -> Tuple[int, float]:
        """Returns (predicted_class, confidence 0-1). Caller must ensure crop is non-empty."""
        input_tensor = self._preprocess(crop)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        probabilities = outputs[0][0]
        exp = np.exp(probabilities - np.max(probabilities))
        softmax = exp / exp.sum()
        class_id = int(np.argmax(softmax))
        confidence = float(softmax[class_id])
        return class_id, confidence


def _dial_sort_key(cls_name: str, x1: int) -> Tuple[int, int]:
    """Sort pointer dials by the number embedded in their class name
    (pointer_reading1, pointer_reading2, ...) so ordering stays correct
    even if the photo is rotated/mirrored. Falls back to x-position for
    any dial whose class name has no trailing number."""
    match = _DIGIT_SUFFIX_RE.search(cls_name)
    if match:
        return (0, int(match.group(1)))
    return (1, x1)  # unnumbered classes sort after numbered ones, by position


def reading_synthesis(odometer_digits: List[int], pointer_place_digit_pairs: List[Tuple[int, int]]) -> str:
    """
    Combine the odometer digit class indices (left-to-right) with the
    pointer-dial digits using each dial's OWN decimal place (from its class
    name, e.g. pointer_reading2 -> hundredths place), not simple left-to-right
    concatenation. This matters whenever a dial is missing from the photo --
    e.g. only pointer_reading2/3/4 detected (no reading1) correctly produces
    a leading 0 in the tenths place: odometer "00002" + [(2,4),(3,4),(4,3)]
    -> "00002.0443", not "00002.443".
    """
    whole_part = "".join(str(d) for d in odometer_digits) if odometer_digits else "0"
    if not pointer_place_digit_pairs:
        return whole_part
    max_place = max(place for place, _ in pointer_place_digit_pairs)
    decimal_digits = [0] * max_place
    for place, digit in pointer_place_digit_pairs:
        decimal_digits[place - 1] = digit
    decimal_part = "".join(str(d) for d in decimal_digits)
    return f"{whole_part}.{decimal_part}"


def _dial_place_number(cls_name: str, fallback_index: int) -> int:
    """Extracts the decimal place from a class name like 'pointer_reading2' -> 2.
    Falls back to sequential position (1-indexed) if the name has no trailing number."""
    match = _DIGIT_SUFFIX_RE.search(cls_name)
    return int(match.group(1)) if match else fallback_index + 1


class MeterReadingPipeline:
    def __init__(self, config: PipelineConfig, progress_callback=None):
        m = config.models_dir.rstrip("/")
        self.config = config

        def report(step, fraction):
            if progress_callback:
                progress_callback(step, fraction)

        report("Loading global detector...", 0.1)
        self.general_detector = YoloDetector(f"{m}/general_yolo.onnx")
        if config.verbose:
            _debug("MODEL CLASSES (general_yolo.onnx)", self.general_detector.class_names)

        report("Loading digit detector...", 0.35)
        self.digit_detector = YoloDetector(f"{m}/odometer_yolo.onnx")

        report("Loading digit classifier...", 0.6)
        self.digit_classifier = CnnClassifier(f"{m}/digit_cnn.onnx", config.digit_input_size)

        report("Loading pointer classifier...", 0.85)
        self.pointer_classifier = CnnClassifier(f"{m}/pointer_cnn.onnx", config.pointer_input_size)

        report("Ready", 1.0)

    def _find_best_orientation(self, frame: np.ndarray) -> Tuple[np.ndarray, int, list]:
        """Tries the image at 0/90/180/270 degrees, runs only the fast Stage 1
        detector on each, and scores by detection completeness + odometer-box
        aspect ratio (landscape preferred). Then, since a 180-degree rotation
        preserves that same landscape aspect while flipping every digit
        upside-down, disambiguates the winning orientation against its
        180-degree twin using actual digit-classifier confidence -- the
        aspect check alone cannot tell right-side-up from upside-down.
        Does NOT fix arbitrary small-angle tilts -- only the four 90-degree
        steps."""
        rotations = [
            (0, None),
            (90, cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        candidates = []  # (degrees, frame, detections, score)

        for degrees, rotate_flag in rotations:
            candidate = frame if rotate_flag is None else cv2.rotate(frame, rotate_flag)
            detections = self.general_detector.detect(candidate, conf=self.config.detection_conf)
            odometer_box = next((d for d in detections if d[5] == self.config.odometer_class_name), None)
            pointer_count = len({d[5] for d in detections if d[5].startswith(self.config.pointer_class_prefix)})
            confidence_sum = sum(d[6] for d in detections)

            aspect_bonus = 0
            if odometer_box is not None:
                ox1, oy1, ox2, oy2 = odometer_box[0], odometer_box[1], odometer_box[2], odometer_box[3]
                box_w, box_h = (ox2 - ox1), (oy2 - oy1)
                aspect_bonus = 300 if box_w >= box_h else -300

            score = (1000 if odometer_box is not None else 0) + (pointer_count * 100) + confidence_sum + aspect_bonus
            candidates.append((degrees, candidate, detections, score))

        best_degrees, best_frame, best_detections, best_score = max(candidates, key=lambda c: c[3])

        # Disambiguate against the 180-degree twin, which scores identically
        # on aspect ratio but is actually upside-down if the winner was
        # right-side-up (or vice versa).
        twin_degrees = (best_degrees + 180) % 360
        twin = next((c for c in candidates if c[0] == twin_degrees), None)
        if twin is not None:
            best_readability = self._digit_readability(best_frame, best_detections)
            twin_readability = self._digit_readability(twin[1], twin[2])
            if self.config.verbose:
                _debug("AUTO-ORIENT — 180° disambiguation (degrees: avg digit-classifier confidence)",
                       {best_degrees: round(best_readability, 3), twin_degrees: round(twin_readability, 3)})
            if twin_readability > best_readability:
                best_degrees, best_frame, best_detections = twin[0], twin[1], twin[2]

        return best_frame, best_degrees, best_detections

    def _digit_readability(self, frame: np.ndarray, detections: list) -> float:
        """Crops the odometer region (if found) and returns the average
        digit-classifier confidence across its detected digits -- a direct
        signal of whether the digits are actually right-side-up, used to
        break the 180-degree aspect-ratio tie in _find_best_orientation.
        Returns 0.0 if no odometer region or no digits were found."""
        odometer_box = next((d for d in detections if d[5] == self.config.odometer_class_name), None)
        if odometer_box is None:
            return 0.0
        ox1, oy1, ox2, oy2 = odometer_box[0], odometer_box[1], odometer_box[2], odometer_box[3]
        crop = frame[oy1:oy2, ox1:ox2]
        if crop.size == 0:
            return 0.0
        digit_boxes = self.digit_detector.detect(crop, conf=self.config.detection_conf)
        if not digit_boxes:
            return 0.0
        confidences = []
        for dx1, dy1, dx2, dy2, _, _, _ in digit_boxes:
            digit_crop = crop[dy1:dy2, dx1:dx2]
            if digit_crop.size == 0:
                continue
            _, cls_conf = self.digit_classifier.predict(digit_crop)
            confidences.append(cls_conf)
        return sum(confidences) / len(confidences) if confidences else 0.0

    def run_on_frame(self, frame: np.ndarray) -> dict:
        """Runs the full pipeline on a single BGR frame. Always returns a dict
        with a 'success' flag and the raw output of every stage, so a caller
        (or the GUI) can see exactly where a failure occurred instead of a
        bare None."""
        start_time = time.perf_counter()
        v = self.config.verbose

        # --- Stage 1: global detection (with optional auto-orientation) ---
        rotation_applied = 0
        if self.config.auto_orient:
            frame, rotation_applied, detections = self._find_best_orientation(frame)
            if v:
                _debug("AUTO-ORIENT — chosen rotation (degrees)", rotation_applied)
        else:
            detections = self.general_detector.detect(frame, conf=self.config.detection_conf)
        if v:
            _debug("STAGE 1 — raw detections (x1,y1,x2,y2,cls_id,cls_name,conf)", detections)

        odometer_candidates = [d for d in detections if d[5] == self.config.odometer_class_name]
        pointer_candidates = [d for d in detections if d[5].startswith(self.config.pointer_class_prefix)]

        # Dedupe: if the same class name appears more than once, keep only
        # the highest-confidence one.
        best_by_class: Dict[str, tuple] = {}
        for d in pointer_candidates:
            cls_name, conf = d[5], d[6]
            if cls_name not in best_by_class or conf > best_by_class[cls_name][6]:
                best_by_class[cls_name] = d
        pointer_boxes = list(best_by_class.values())
        pointer_boxes.sort(key=lambda d: _dial_sort_key(d[5], d[0]))

        odometer_box = max(odometer_candidates, key=lambda d: d[6]) if odometer_candidates else None

        if odometer_box is None or not pointer_boxes:
            missing = []
            if odometer_box is None:
                missing.append(f"'{self.config.odometer_class_name}' region")
            if not pointer_boxes:
                missing.append(f"any '{self.config.pointer_class_prefix}*' dial")
            detected_names = [(d[5], round(d[6], 2)) for d in detections]
            message = (
                f"Stage 1 did not find: {', '.join(missing)}. "
                f"Raw detections this frame (name, confidence): {detected_names or 'none at all'}."
            )
            if v:
                _debug("STAGE 1 — FAILED", message)
            return {
                "success": False,
                "message": message,
                "stage1_detections": detected_names,
                "rotation_applied": rotation_applied,
                "oriented_frame": frame,
                "inference_time_ms": (time.perf_counter() - start_time) * 1000.0,
                "timestamp": time.time(),
            }

        # --- Stage 2: crop ---
        ox1, oy1, ox2, oy2, _, _, odometer_conf = odometer_box
        odometer_crop = frame[oy1:oy2, ox1:ox2]
        pointer_crops = [frame[p[1]:p[3], p[0]:p[2]] for p in pointer_boxes]
        if v:
            _debug("STAGE 2 — crops", {
                "odometer_crop_shape": odometer_crop.shape,
                "pointer_crop_shapes": [c.shape for c in pointer_crops],
                "pointer_dial_names_in_order": [p[5] for p in pointer_boxes],
            })

        if odometer_crop.size == 0:
            return {
                "success": False,
                "message": "Odometer region detected but the crop was empty (zero-size box).",
                "rotation_applied": rotation_applied,
                "oriented_frame": frame,
                "inference_time_ms": (time.perf_counter() - start_time) * 1000.0,
                "timestamp": time.time(),
            }

        # --- Stage 3: digit localization within odometer crop ---
        digit_boxes = self.digit_detector.detect(odometer_crop, conf=self.config.detection_conf)
        # Sort along whichever axis actually matches this crop's shape --
        # a horizontal digit strip reads left-to-right (sort by x), but if a
        # crop still comes out portrait (taller than wide), the digits are
        # stacked vertically and must be read top-to-bottom (sort by y).
        crop_h, crop_w = odometer_crop.shape[:2]
        digit_boxes.sort(key=lambda d: d[0] if crop_w >= crop_h else d[1])
        if v:
            _debug("STAGE 3 — digit boxes", digit_boxes)

        if not digit_boxes:
            return {
                "success": False,
                "message": "No odometer digits detected inside the odometer crop.",
                "odometer_crop": odometer_crop,
                "rotation_applied": rotation_applied,
                "oriented_frame": frame,
                "inference_time_ms": (time.perf_counter() - start_time) * 1000.0,
                "timestamp": time.time(),
            }

        # --- Stage 4: classify each odometer digit ---
        odometer_digits = []
        digit_confidences = []       # combined (box_conf * cls_conf), kept for the blended overall score
        digit_box_confs = []         # detection-only
        digit_cls_confs = []         # classification-only
        digit_confidence_log = []    # per-digit (position, predicted_digit, classifier_confidence) for diagnosis
        digit_crops = []
        for idx, (dx1, dy1, dx2, dy2, _, _, digit_box_conf) in enumerate(digit_boxes):
            digit_crop = odometer_crop[dy1:dy2, dx1:dx2]
            if digit_crop.size == 0:
                continue
            cls_id, cls_conf = self.digit_classifier.predict(digit_crop)
            odometer_digits.append(cls_id)
            digit_confidences.append(cls_conf * digit_box_conf)
            digit_box_confs.append(digit_box_conf)
            digit_cls_confs.append(cls_conf)
            digit_confidence_log.append((idx + 1, cls_id, round(cls_conf, 3)))
            digit_crops.append(digit_crop)
        if v:
            _debug("STAGE 4 — odometer digits (position, predicted_digit, classifier_confidence)", digit_confidence_log)

        digit_count_warning = None
        if self.config.expected_odometer_digits is not None and len(odometer_digits) != self.config.expected_odometer_digits:
            digit_count_warning = (
                f"Expected {self.config.expected_odometer_digits} odometer digits, "
                f"found {len(odometer_digits)} — reading may be unreliable."
            )

        # --- Stage 5: classify each pointer dial ---
        # Small padding around each crop -- a tightly-fit box can clip the
        # needle tip right at the edge, which especially hurts on dials
        # where the needle sits between two printed numbers.
        pad = 6
        h, w = frame.shape[:2]
        pointer_digits = []
        pointer_places = []
        pointer_confidences = []      # combined
        pointer_box_confs = []        # detection-only
        pointer_cls_confs = []        # classification-only
        pointer_digit_confidences = []  # per-dial (class, place, digit, confidence) for diagnosis
        for i, (px1, py1, px2, py2, _, cls_name, pointer_box_conf) in enumerate(pointer_boxes):
            px1p, py1p = max(px1 - pad, 0), max(py1 - pad, 0)
            px2p, py2p = min(px2 + pad, w), min(py2 + pad, h)
            pointer_crop = frame[py1p:py2p, px1p:px2p]
            if pointer_crop.size == 0:
                continue
            cls_id, cls_conf = self.pointer_classifier.predict(pointer_crop)
            place = _dial_place_number(cls_name, i)
            pointer_digits.append(cls_id)
            pointer_places.append(place)
            pointer_confidences.append(cls_conf * pointer_box_conf)
            pointer_box_confs.append(pointer_box_conf)
            pointer_cls_confs.append(cls_conf)
            pointer_digit_confidences.append((cls_name, place, cls_id, round(cls_conf, 3)))
        if v:
            _debug("STAGE 5 — pointer digits (dial, decimal_place, predicted_digit, classifier_confidence)", pointer_digit_confidences)

        final_reading = reading_synthesis(odometer_digits, list(zip(pointer_places, pointer_digits)))

        all_confidences = digit_confidences + pointer_confidences + [odometer_conf]
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        detection_confs = digit_box_confs + pointer_box_confs + [odometer_conf]
        detection_confidence = sum(detection_confs) / len(detection_confs) if detection_confs else 0.0

        classification_confs = digit_cls_confs + pointer_cls_confs
        classification_confidence = (
            sum(classification_confs) / len(classification_confs) if classification_confs else 0.0
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        if v:
            _debug("READING SYNTHESIS", {
                "reading": final_reading,
                "detection_confidence": detection_confidence,
                "classification_confidence": classification_confidence,
            })

        return {
            "success": True,
            "warning": digit_count_warning,
            "odometer_digits": odometer_digits,
            "digit_confidence_log": digit_confidence_log,
            "pointer_digits": pointer_digits,
            "pointer_digit_confidences": pointer_digit_confidences,
            "reading": final_reading,
            "confidence": overall_confidence,
            "detection_confidence": detection_confidence,
            "classification_confidence": classification_confidence,
            "rotation_applied": rotation_applied,
            "inference_time_ms": inference_time_ms,
            "odometer_box": (ox1, oy1, ox2, oy2),
            "pointer_boxes": [(p[0], p[1], p[2], p[3]) for p in pointer_boxes],
            "digit_boxes": [(b[0] + ox1, b[1] + oy1, b[2] + ox1, b[3] + oy1) for b in digit_boxes],
            "odometer_crop": odometer_crop,
            "pointer_crops": pointer_crops,
            "digit_crops": digit_crops,
            "stage1_detections": [(d[5], round(d[6], 2)) for d in detections],
            "oriented_frame": frame,
            "timestamp": time.time(),
        }
