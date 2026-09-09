"""
Standalone desktop GUI for the analog meter reading system.

No web server, no Flask — a plain tkinter window running directly on
the Pi's desktop. Supports both live camera capture and uploading a
static image through the same pipeline, with an explicit step-by-step
view: annotated image -> odometer crop/digits -> pointer dial crops/digits
-> final computed reading.

Requirements on the Pi:
    sudo apt install -y python3-picamera2 python3-tk python3-pil.imagetk
    pip install ultralytics onnxruntime opencv-python-headless --break-system-packages

Usage:
    python3 desktop_app.py --models_dir /home/pi/Desktop/projectwork/onnx_models
"""

import argparse
import csv
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2

from camera import CameraManager, CameraNotAvailableError, ImageCorrector
from gui_utils import crop_to_photoimage, draw_overlays, format_history_entry, frame_to_photoimage
from image_loader import ImageLoadError, load_image_bgr, tk_file_dialog_filetypes
from meter_pipeline import MeterReadingPipeline, PipelineConfig

# --- Color palette ---
COLOR_BG = "#f4f6f8"
COLOR_PANEL = "#ffffff"
COLOR_HEADER = "#1f3a5f"
COLOR_HEADER_TEXT = "#ffffff"
COLOR_ACCENT = "#2f6fed"
COLOR_TEXT = "#1c1c1c"
COLOR_TEXT_MUTED = "#6b7280"
COLOR_SUCCESS = "#1a9850"
COLOR_WARNING = "#c98a12"
COLOR_ERROR = "#c62828"
COLOR_PREVIEW_BG = "#12161c"


def _confidence_color(fraction: float) -> str:
    """Green for confident, amber for borderline, red for weak — fraction is 0-1."""
    if fraction >= 0.6:
        return COLOR_SUCCESS
    if fraction >= 0.35:
        return COLOR_WARNING
    return COLOR_ERROR


class MeterReaderApp:
    def __init__(self, root: tk.Tk, models_dir: str, capture_interval: float = 5.0, camera_kwargs=None):
        self._camera_kwargs = camera_kwargs or {}
        self.root = root
        self.root.title("Analog Meter Reader")
        self.root.geometry("1120x680")
        self.root.minsize(760, 480)
        self.root.configure(background=COLOR_BG)
        try:
            self.root.attributes("-zoomed", True)  # start maximized on X11/Linux (e.g. Raspberry Pi OS)
        except tk.TclError:
            pass  # not supported on this platform — falls back to the geometry above

        self._setup_style()

        self.capture_interval = capture_interval
        self.camera_running = False
        self.history = []
        self.last_annotated_frame = None
        self.pending_frame = None  # the raw (possibly rotated) frame waiting to be processed
        self.result_queue = queue.Queue()
        self._stage_thumbnails = []  # keep PhotoImage references alive

        self.camera = CameraManager(**self._camera_kwargs)
        self.pipeline = None

        self._build_layout()
        self.root.after(100, self._poll_queue)

        threading.Thread(target=self._load_models, args=(models_dir,), daemon=True).start()

    def _setup_style(self):
        """Modern flat theme with a consistent color palette instead of the
        default gray Tk look. 'clam' is the base theme most reliably
        available on Raspberry Pi OS / Linux Tk installs."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass  # fall back to whatever the platform default is

        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT, font=("Helvetica", 10))
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Header.TFrame", background=COLOR_HEADER)
        style.configure("Header.TLabel", background=COLOR_HEADER, foreground=COLOR_HEADER_TEXT,
                         font=("Helvetica", 16, "bold"))
        style.configure("HeaderSub.TLabel", background=COLOR_HEADER, foreground="#cfe0ff")
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_TEXT_MUTED)
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT)
        style.configure("Reading.TLabel", background=COLOR_PANEL, foreground=COLOR_ACCENT,
                         font=("Helvetica", 30, "bold"))
        style.configure("SectionHeader.TLabel", background=COLOR_PANEL, foreground=COLOR_HEADER,
                         font=("Helvetica", 12, "bold"))
        style.configure("TButton", padding=6, font=("Helvetica", 10))
        style.map("TButton", background=[("active", "#e3ecff")])
        style.configure("Accent.TButton", padding=6, font=("Helvetica", 10, "bold"))
        style.map("Accent.TButton",
                  background=[("!disabled", COLOR_ACCENT), ("active", "#255bc7")],
                  foreground=[("!disabled", "#ffffff")])
        style.configure("TProgressbar", background=COLOR_ACCENT, troughcolor="#e0e4ea")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        header = ttk.Frame(self.root, padding=(14, 12), style="Header.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="💧 Analog Meter Reader", style="Header.TLabel").pack(side=tk.LEFT)

        self.camera_status_var = tk.StringVar(value="●  Camera: not opened")
        ttk.Label(header, textvariable=self.camera_status_var, style="HeaderSub.TLabel").pack(side=tk.RIGHT)

        main = ttk.Frame(self.root, padding=(12, 10, 12, 12))
        main.pack(fill=tk.BOTH, expand=True)

        # --- Left: controls (packed first/top so they can never be pushed
        # off-screen by a large preview image), then a fixed-size scrollable
        # preview canvas below that takes any remaining space ---
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.progress = ttk.Progressbar(left, mode="determinate", maximum=1.0)
        self.progress.pack(fill=tk.X, pady=(0, 6))

        controls_row1 = ttk.Frame(left)
        controls_row1.pack(fill=tk.X, pady=(0, 6))
        self.open_cam_btn = ttk.Button(controls_row1, text="Open Camera", command=self.open_camera, state=tk.DISABLED)
        self.open_cam_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.close_cam_btn = ttk.Button(controls_row1, text="Close Camera", command=self.close_camera, state=tk.DISABLED)
        self.close_cam_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.upload_btn = ttk.Button(controls_row1, text="Upload Image", command=self.upload_image, state=tk.DISABLED)
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.rotate_left_btn = ttk.Button(controls_row1, text="⟲ Rotate Left", command=lambda: self.rotate_pending(-90), state=tk.DISABLED)
        self.rotate_left_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.rotate_right_btn = ttk.Button(controls_row1, text="⟳ Rotate Right", command=lambda: self.rotate_pending(90), state=tk.DISABLED)
        self.rotate_right_btn.pack(side=tk.LEFT, padx=(0, 6))

        controls_row2 = ttk.Frame(left)
        controls_row2.pack(fill=tk.X, pady=(0, 6))
        self.run_btn = ttk.Button(controls_row2, text="▶  Run Inference", command=self.run_inference,
                                   state=tk.DISABLED, style="Accent.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.save_btn = ttk.Button(controls_row2, text="Save Annotated Image", command=self.save_annotated_image, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.export_btn = ttk.Button(controls_row2, text="Export CSV", command=self.export_csv)
        self.export_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.clear_btn = ttk.Button(controls_row2, text="Clear History", command=self.clear_history)
        self.clear_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Loading models...")
        self.status_label = ttk.Label(left, textvariable=self.status_var, style="Muted.TLabel", wraplength=640)
        self.status_label.pack(anchor="w", pady=(2, 6))

        # Preview: a fixed-size Canvas with scrollbars, so a large source image
        # is scaled to fit the visible area but can also be panned/scrolled
        # rather than forcing the whole window to grow to the image's size.
        preview_container = ttk.Frame(left)
        preview_container.pack(fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(preview_container, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(preview_container, orient=tk.VERTICAL)
        self.preview_canvas = tk.Canvas(
            preview_container, background=COLOR_PREVIEW_BG, highlightthickness=0,
            xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set,
        )
        h_scroll.config(command=self.preview_canvas.xview)
        v_scroll.config(command=self.preview_canvas.yview)

        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)

        self.preview_canvas.bind("<ButtonPress-1>", lambda e: self.preview_canvas.scan_mark(e.x, e.y))
        self.preview_canvas.bind("<B1-Motion>", lambda e: self.preview_canvas.scan_dragto(e.x, e.y, gain=1))

        self._preview_placeholder_id = self.preview_canvas.create_text(
            300, 150, text="Preview", fill="white", font=("Helvetica", 12)
        )
        self._preview_image_id = None
        self._preview_photo = None  # keep a reference so it isn't garbage-collected

        # --- Right: stage-by-stage results + history ---
        right_outer = ttk.Frame(main, width=372)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        right_outer.pack_propagate(False)

        right = ttk.Frame(right_outer, style="Panel.TFrame", padding=14)
        right.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="FINAL READING", style="SectionHeader.TLabel").pack(anchor="w")
        self.reading_var = tk.StringVar(value="--")
        ttk.Label(right, textvariable=self.reading_var, style="Reading.TLabel").pack(anchor="w", pady=(0, 8))

        stats = ttk.Frame(right, style="Panel.TFrame")
        stats.pack(fill=tk.X, pady=(0, 8))
        self._stat_row(stats, "Odometer digits", "odometer_var", "--")
        self._stat_row(stats, "Pointer digits", "pointer_var", "--")
        self.detection_confidence_label = self._stat_row(stats, "Detection conf.", "detection_confidence_var", "--")
        self.classification_confidence_label = self._stat_row(stats, "Classification conf.", "classification_confidence_var", "--")
        self._stat_row(stats, "Inference Time", "inference_time_var", "--")

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(right, text="STAGES", style="SectionHeader.TLabel").pack(anchor="w")

        self.stage_frame = ttk.Frame(right, style="Panel.TFrame")
        self.stage_frame.pack(fill=tk.X, pady=(6, 8))
        self.stage_placeholder = ttk.Label(self.stage_frame, text="Run inference to see odometer\nand pointer-dial crops here.",
                                            style="Panel.TLabel", foreground=COLOR_TEXT_MUTED, justify=tk.LEFT)
        self.stage_placeholder.pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(right, text="HISTORY", style="SectionHeader.TLabel").pack(anchor="w")

        history_frame = ttk.Frame(right, style="Panel.TFrame")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.history_list = tk.Listbox(history_frame, font=("Courier", 9), background=COLOR_PANEL,
                                        foreground=COLOR_TEXT, selectbackground=COLOR_ACCENT,
                                        borderwidth=0, highlightthickness=1, highlightbackground="#dde2ea")
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_list.yview)
        self.history_list.configure(yscrollcommand=scrollbar.set)
        self.history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _stat_row(self, parent, label, var_name, default):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=f"{label}:", width=16, style="Panel.TLabel").pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        setattr(self, var_name, var)
        value_label = tk.Label(row, textvariable=var, font=("Helvetica", 10, "bold"),
                                background=COLOR_PANEL, foreground=COLOR_TEXT)
        value_label.pack(side=tk.LEFT)
        return value_label

    def _set_status(self, text: str, kind: str = "info"):
        """kind: 'info' (muted gray), 'success' (green), 'warning' (amber), 'error' (red)."""
        colors = {"info": COLOR_TEXT_MUTED, "success": COLOR_SUCCESS,
                  "warning": COLOR_WARNING, "error": COLOR_ERROR}
        self.status_var.set(text)
        self.status_label.configure(foreground=colors.get(kind, COLOR_TEXT_MUTED))

    # ------------------------------------------------------------------
    # Model loading (background thread)
    # ------------------------------------------------------------------

    def _load_models(self, models_dir: str):
        def progress_callback(step, fraction):
            self.result_queue.put(("progress", (step, fraction)))

        try:
            config = PipelineConfig(models_dir=models_dir)
            self.pipeline = MeterReadingPipeline(config, progress_callback=progress_callback)
            self.result_queue.put(("models_ready", None))
        except Exception as e:
            self.result_queue.put(("error", f"Failed to load models: {e}"))

    # ------------------------------------------------------------------
    # Camera controls
    # ------------------------------------------------------------------

    def open_camera(self):
        def do_open():
            try:
                self.camera.open()
                self.result_queue.put(("camera_opened", None))
                self.camera_running = True
                threading.Thread(target=self._camera_loop, daemon=True).start()
            except CameraNotAvailableError as e:
                self.result_queue.put(("error", str(e)))

        self.open_cam_btn.config(state=tk.DISABLED)
        self._set_status("Opening camera...", "info")
        threading.Thread(target=do_open, daemon=True).start()

    def close_camera(self):
        self.camera_running = False
        self.camera.close()
        self.open_cam_btn.config(state=tk.NORMAL)
        self.close_cam_btn.config(state=tk.DISABLED)
        self.camera_status_var.set("● Camera: closed")
        self._set_status("Camera closed.", "info")

    def _camera_loop(self):
        while self.camera_running:
            try:
                frame = self.camera.capture_frame_bgr()
                self.pending_frame = frame
                self._process_and_queue(frame)
            except Exception as e:
                self.result_queue.put(("error", str(e)))
                break
            time.sleep(self.capture_interval)

    # ------------------------------------------------------------------
    # Upload image + rotation
    # ------------------------------------------------------------------

    def upload_image(self):
        path = filedialog.askopenfilename(title="Select meter image", filetypes=tk_file_dialog_filetypes())
        if not path:
            return
        self._set_status(f"Loading {os.path.basename(path)}...", "info")

        def do_load():
            try:
                frame = load_image_bgr(path)
                self.pending_frame = frame
                self.result_queue.put(("frame", frame))
                self.result_queue.put(("status", f"Loaded {os.path.basename(path)}. Rotate if needed, then click Run Inference."))
                self.result_queue.put(("enable_run", None))
            except ImageLoadError as e:
                self.result_queue.put(("error", str(e)))

        threading.Thread(target=do_load, daemon=True).start()

    def rotate_pending(self, degrees: int):
        if self.pending_frame is None:
            self._set_status("No image loaded to rotate — upload or capture one first.", "warning")
            return
        if degrees == 90:
            self.pending_frame = cv2.rotate(self.pending_frame, cv2.ROTATE_90_CLOCKWISE)
        elif degrees == -90:
            self.pending_frame = cv2.rotate(self.pending_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif abs(degrees) == 180:
            self.pending_frame = cv2.rotate(self.pending_frame, cv2.ROTATE_180)
        self._show_frame(self.pending_frame)
        self._set_status("Rotated. Click Run Inference when the meter looks upright.", "info")

    def run_inference(self):
        if self.pending_frame is None:
            self._set_status("No image loaded — click Upload Image first.", "warning")
            return
        self._set_status("Running inference...", "info")
        self.run_btn.config(state=tk.DISABLED)

        def do_run():
            self._process_and_queue(self.pending_frame)
            self.result_queue.put(("enable_run", None))

        threading.Thread(target=do_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Shared processing (camera frame or uploaded image)
    # ------------------------------------------------------------------

    def _process_and_queue(self, frame_bgr):
        if self.pipeline is None:
            self.result_queue.put(("status_info", "Models still loading — try again shortly."))
            return
        result = self.pipeline.run_on_frame(frame_bgr)
        # Box coordinates in `result` are relative to the auto-oriented frame,
        # not necessarily the original -- always draw on whichever frame the
        # pipeline actually used.
        display_frame = result.get("oriented_frame", frame_bgr)
        annotated = draw_overlays(display_frame, result)
        self.result_queue.put(("frame", annotated))
        self.result_queue.put(("reading", result))
        if not result.get("success"):
            self.result_queue.put(("status_error", result.get("message", "Detection failed.")))
        elif result.get("rotation_applied"):
            self.result_queue.put(("status_info", f"Auto-oriented {result['rotation_applied']}° before processing."))

    # ------------------------------------------------------------------
    # GUI-thread queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "progress":
                    step, fraction = payload
                    self.progress["value"] = fraction
                    self._set_status(step, "info")
                elif kind == "models_ready":
                    self._set_status("Models loaded. Open the camera or upload an image.", "success")
                    self.open_cam_btn.config(state=tk.NORMAL)
                    self.upload_btn.config(state=tk.NORMAL)
                elif kind == "enable_run":
                    self.run_btn.config(state=tk.NORMAL)
                    self.rotate_left_btn.config(state=tk.NORMAL)
                    self.rotate_right_btn.config(state=tk.NORMAL)
                elif kind == "camera_opened":
                    self.camera_status_var.set("●  Camera: live")
                    self.close_cam_btn.config(state=tk.NORMAL)
                    self._set_status("Camera running...", "success")
                elif kind == "frame":
                    self._show_frame(payload)
                elif kind == "reading":
                    self._show_reading(payload)
                elif kind == "status" or kind == "status_info":
                    self._set_status(payload, "info")
                elif kind == "status_error":
                    self._set_status(payload, "error")
                elif kind == "error":
                    self._set_status(f"Error: {payload}", "error")
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _show_frame(self, frame_bgr):
        self.last_annotated_frame = frame_bgr

        canvas_width = self.preview_canvas.winfo_width()
        target_width = canvas_width if canvas_width > 50 else 640
        photo = frame_to_photoimage(frame_bgr, target_width)

        if self._preview_image_id is None:
            self.preview_canvas.delete(self._preview_placeholder_id)
            self._preview_image_id = self.preview_canvas.create_image(0, 0, anchor="nw", image=photo)
        else:
            self.preview_canvas.itemconfig(self._preview_image_id, image=photo)

        self._preview_photo = photo  # keep a reference so tkinter doesn't garbage-collect it
        self.preview_canvas.config(scrollregion=(0, 0, photo.width(), photo.height()))
        self.save_btn.config(state=tk.NORMAL)

    def _show_reading(self, result: dict):
        if not result.get("success"):
            self.reading_var.set("--")
            self.odometer_var.set("not detected")
            self.pointer_var.set("not detected")
            self.detection_confidence_var.set("--")
            self.classification_confidence_var.set("--")
            self.detection_confidence_label.configure(foreground=COLOR_TEXT)
            self.classification_confidence_label.configure(foreground=COLOR_TEXT)
            self.inference_time_var.set(f"{result.get('inference_time_ms', 0):.0f} ms")
            self._clear_stage_thumbnails()
            self.history.append(result)
            self.history_list.insert(tk.END, format_history_entry(result))
            self.history_list.see(tk.END)
            return

        self.reading_var.set(str(result["reading"]))
        self.odometer_var.set("".join(str(d) for d in result["odometer_digits"]) or "--")
        self.pointer_var.set("".join(str(d) for d in result["pointer_digits"]) or "--")

        det_conf, cls_conf = result["detection_confidence"], result["classification_confidence"]
        self.detection_confidence_var.set(f"{det_conf*100:.1f}%")
        self.classification_confidence_var.set(f"{cls_conf*100:.1f}%")
        self.detection_confidence_label.configure(foreground=_confidence_color(det_conf))
        self.classification_confidence_label.configure(foreground=_confidence_color(cls_conf))
        self.inference_time_var.set(f"{result['inference_time_ms']:.0f} ms")

        self._show_stage_thumbnails(result)

        self.history.append(result)
        self.history_list.insert(tk.END, format_history_entry(result))
        self.history_list.see(tk.END)
        if result.get("warning"):
            self._set_status(f"⚠ {result['warning']}", "warning")
            return
        self._set_status("✓ Reading updated.", "success")

    def _clear_stage_thumbnails(self):
        for widget in self.stage_frame.winfo_children():
            widget.destroy()
        self._stage_thumbnails = []
        self.stage_placeholder = ttk.Label(self.stage_frame, text="Odometer/pointer not detected this run.",
                                            foreground="gray", justify=tk.LEFT)
        self.stage_placeholder.pack(anchor="w")

    def _show_stage_thumbnails(self, result: dict):
        for widget in self.stage_frame.winfo_children():
            widget.destroy()
        self._stage_thumbnails = []

        ttk.Label(self.stage_frame, text="Odometer crop:").pack(anchor="w")
        odometer_photo = crop_to_photoimage(result.get("odometer_crop"), target_width=260)
        if odometer_photo is not None:
            self._stage_thumbnails.append(odometer_photo)
            ttk.Label(self.stage_frame, image=odometer_photo).pack(anchor="w", pady=(0, 6))

        pointer_crops = result.get("pointer_crops", [])
        if pointer_crops:
            ttk.Label(self.stage_frame, text=f"Pointer dials ({len(pointer_crops)}), left to right:").pack(anchor="w")
            dials_row = ttk.Frame(self.stage_frame)
            dials_row.pack(anchor="w", pady=(0, 4))
            for i, crop in enumerate(pointer_crops):
                photo = crop_to_photoimage(crop, target_width=70)
                if photo is not None:
                    self._stage_thumbnails.append(photo)
                    col = ttk.Frame(dials_row)
                    col.pack(side=tk.LEFT, padx=(0, 4))
                    ttk.Label(col, image=photo).pack()
                    digit = result["pointer_digits"][i] if i < len(result["pointer_digits"]) else "?"
                    ttk.Label(col, text=str(digit), anchor="center").pack()

    # ------------------------------------------------------------------
    # Save / export / clear
    # ------------------------------------------------------------------

    def save_annotated_image(self):
        if self.last_annotated_frame is None:
            return
        os.makedirs("results", exist_ok=True)
        filename = f"results/annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, self.last_annotated_frame)
        self._set_status(f"Saved annotated image to {filename}", "success")

    def export_csv(self):
        successful = [r for r in self.history if r.get("success")]
        if not successful:
            self._set_status("No successful readings yet to export.", "warning")
            return
        os.makedirs("exports", exist_ok=True)
        filename = f"exports/readings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "reading", "odometer_digits", "pointer_digits",
                              "detection_confidence", "classification_confidence",
                              "rotation_applied_deg", "inference_time_ms"])
            for r in successful:
                writer.writerow([
                    datetime.fromtimestamp(r["timestamp"]).isoformat(),
                    r["reading"],
                    "".join(str(d) for d in r["odometer_digits"]),
                    "".join(str(d) for d in r["pointer_digits"]),
                    f"{r['detection_confidence']:.4f}",
                    f"{r['classification_confidence']:.4f}",
                    r.get("rotation_applied", 0),
                    f"{r['inference_time_ms']:.1f}",
                ])
        self._set_status(f"Exported {len(successful)} readings to {filename}", "success")

    def clear_history(self):
        self.history.clear()
        self.history_list.delete(0, tk.END)
        self._set_status("History cleared.", "info")

    def on_close(self):
        self.camera_running = False
        self.camera.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True,
                         help="Directory containing general_yolo.onnx, odometer_yolo.onnx, digit_cnn.onnx, pointer_cnn.onnx")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between camera captures")

    # Manual capture settings -- values found via calibrate_camera.py.
    parser.add_argument("--red_gain", type=float, default=None, help="Manual ColourGains red value (e.g. 1.0)")
    parser.add_argument("--blue_gain", type=float, default=None, help="Manual ColourGains blue value (e.g. 1.0)")
    parser.add_argument("--exposure_us", type=int, default=None, help="Manual exposure time in microseconds (e.g. 12000)")
    parser.add_argument("--analogue_gain", type=float, default=None, help="Manual analogue gain (e.g. 1.2)")
    parser.add_argument("--led_pin", type=int, default=None,
                         help="GPIO pin number for an LED light source turned on during each capture (requires gpiozero + wired LED)")

    # Post-capture correction.
    parser.add_argument("--correct_image", action="store_true",
                         help="Apply gray-world white balance + CLAHE contrast + gamma correction to every "
                              "captured frame. Recommended over the simpler --fix_noir_color option below.")
    parser.add_argument("--gamma", type=float, default=1.2, help="Gamma value used by --correct_image")
    parser.add_argument("--fix_noir_color", action="store_true",
                         help="Simpler alternative to --correct_image: a flat per-channel colour scale. "
                              "Off by default -- only enable if your camera photos look tinted.")
    args = parser.parse_args()

    colour_gains = (args.red_gain, args.blue_gain) if args.red_gain is not None and args.blue_gain is not None else None

    corrector = ImageCorrector(gamma=args.gamma) if args.correct_image else None
    # Starting point for a NoIR module's colour cast if using the simpler option instead.
    channel_correction = (0.75, 1.10, 1.20) if (args.fix_noir_color and not args.correct_image) else None

    camera_kwargs = dict(
        exposure_time_us=args.exposure_us,
        analogue_gain=args.analogue_gain,
        colour_gains=colour_gains,
        led_pin=args.led_pin,
        corrector=corrector,
        channel_correction=channel_correction,
    )

    root = tk.Tk()
    app = MeterReaderApp(root, models_dir=args.models_dir, capture_interval=args.interval,
                          camera_kwargs=camera_kwargs)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
