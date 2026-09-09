"""
Convert all 4 trained models to ONNX for Raspberry Pi deployment.

Run this on your DEVELOPMENT machine (not the Pi) — conversion needs
the full TensorFlow / Ultralytics training stack, which you don't
want to install on the Pi itself.

Requirements (dev machine):
    pip install tensorflow tf2onnx ultralytics onnx onnxruntime

Usage:
    python convert_models.py --models_dir ./models --out_dir ./onnx_models
"""

import argparse
import os
import subprocess
import sys


def convert_keras_to_onnx(keras_path: str, onnx_path: str, opset: int = 13):
    """Convert a .keras model to ONNX using tf2onnx (via its CLI, called as a subprocess
    to avoid keras/tf2onnx API version mismatches)."""
    print(f"\n[Keras -> ONNX] {keras_path} -> {onnx_path}")
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--keras", keras_path,
        "--output", onnx_path,
        "--opset", str(opset),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"tf2onnx conversion failed for {keras_path}")
    print(f"  done -> {onnx_path}")


def convert_yolo_to_onnx(pt_path: str, out_dir: str, imgsz: int = 640):
    """Convert a YOLOv10 .pt checkpoint to ONNX using Ultralytics' built-in exporter."""
    from ultralytics import YOLO

    print(f"\n[YOLO .pt -> ONNX] {pt_path}")
    model = YOLO(pt_path)
    exported_path = model.export(format="onnx", imgsz=imgsz, opset=13)
    # Ultralytics exports next to the .pt file by default; move it into out_dir
    final_name = os.path.splitext(os.path.basename(pt_path))[0] + ".onnx"
    final_path = os.path.join(out_dir, final_name)
    if os.path.abspath(exported_path) != os.path.abspath(final_path):
        os.replace(exported_path, final_path)
    print(f"  done -> {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="./models",
                         help="Directory containing the 4 source model files")
    parser.add_argument("--out_dir", default="./onnx_models",
                         help="Directory to write converted .onnx files")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Input image size used when the YOLO models were trained")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = {
        "digit_cnn": os.path.join(args.models_dir, "digit_cnn_model.keras"),
        "pointer_cnn": os.path.join(args.models_dir, "pointer_cnn_model.keras"),
        "general_yolo": os.path.join(args.models_dir, "General best.pt"),
        "odometer_yolo": os.path.join(args.models_dir, "Odometer YOLO best.pt"),
    }

    missing = [name for name, path in files.items() if not os.path.exists(path)]
    if missing:
        print(f"WARNING: these expected files were not found: {missing}")
        print(f"Looked in: {args.models_dir}")

    # Keras models
    if os.path.exists(files["digit_cnn"]):
        convert_keras_to_onnx(
            files["digit_cnn"],
            os.path.join(args.out_dir, "digit_cnn.onnx"),
        )
    if os.path.exists(files["pointer_cnn"]):
        convert_keras_to_onnx(
            files["pointer_cnn"],
            os.path.join(args.out_dir, "pointer_cnn.onnx"),
        )

    # YOLO models
    if os.path.exists(files["general_yolo"]):
        convert_yolo_to_onnx(files["general_yolo"], args.out_dir, imgsz=args.imgsz)
    if os.path.exists(files["odometer_yolo"]):
        convert_yolo_to_onnx(files["odometer_yolo"], args.out_dir, imgsz=args.imgsz)

    print("\nAll conversions attempted. Copy the contents of "
          f"'{args.out_dir}' to the Pi, e.g.:\n"
          f"  scp {args.out_dir}/*.onnx pi@<pi-ip>:/home/pi/models/")


if __name__ == "__main__":
    main()
