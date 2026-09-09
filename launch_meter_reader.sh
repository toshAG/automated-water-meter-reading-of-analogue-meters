#!/bin/bash

# ============================================================
# ANALOG / WATER METER READER LAUNCHER
# ============================================================

PROJECT_DIR="/home/pi/Desktop/projectwork"
MODELS_DIR="$PROJECT_DIR/onnx_models"

# ------------------------------------------------------------
# Go to project directory
# ------------------------------------------------------------

cd "$PROJECT_DIR" || {
    echo "ERROR: Project directory not found:"
    echo "$PROJECT_DIR"
    echo ""
    read -p "Press Enter to close..."
    exit 1
}

echo "=============================================="
echo "       WATER METER READER"
echo "=============================================="
echo ""
echo "Project directory:"
echo "$PROJECT_DIR"
echo ""
echo "Models directory:"
echo "$MODELS_DIR"
echo ""

# ------------------------------------------------------------
# Camera configuration
# ------------------------------------------------------------

echo "Camera settings:"
echo "  LED GPIO       : 26"
echo "  Exposure       : 12000 us"
echo "  Analogue Gain  : 1.2"
echo "  Red Gain       : 1.0"
echo "  Blue Gain      : 1.0"
echo "  Image Correction: ENABLED"
echo "  Interval       : 5 seconds"
echo ""

# ------------------------------------------------------------
# Check important files
# ------------------------------------------------------------

if [ ! -f "$PROJECT_DIR/desktop_app.py" ]; then
    echo "ERROR: desktop_app.py not found."
    read -p "Press Enter to close..."
    exit 1
fi

if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: ONNX models directory not found:"
    echo "$MODELS_DIR"
    read -p "Press Enter to close..."
    exit 1
fi

# ------------------------------------------------------------
# Check ONNX models
# ------------------------------------------------------------

MODELS=(
    "general_yolo.onnx"
    "odometer_yolo.onnx"
    "digit_cnn.onnx"
    "pointer_cnn.onnx"
)

for MODEL in "${MODELS[@]}"; do
    if [ ! -f "$MODELS_DIR/$MODEL" ]; then
        echo "ERROR: Missing model:"
        echo "$MODELS_DIR/$MODEL"
        read -p "Press Enter to close..."
        exit 1
    fi
done

echo "All required ONNX models found."
echo ""

# ------------------------------------------------------------
# Start application
# ------------------------------------------------------------

echo "Starting Water Meter Reader..."
echo ""

python3 "$PROJECT_DIR/desktop_app.py" \
    --models_dir "$MODELS_DIR" \
    --interval 5 \
    --led_pin 26 \
    --exposure_us 12000 \
    --analogue_gain 1.2 \
    --red_gain 1.0 \
    --blue_gain 1.0 \
    --correct_image

EXIT_CODE=$?

# ------------------------------------------------------------
# Application finished
# ------------------------------------------------------------

echo ""
echo "=============================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Water Meter Reader closed normally."
else
    echo "Water Meter Reader exited with an error."
    echo "Exit code: $EXIT_CODE"
fi

echo "=============================================="
echo ""

read -p "Press Enter to close this window..."
exit $EXIT_CODE