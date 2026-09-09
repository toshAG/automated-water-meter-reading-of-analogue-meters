# Raspberry Pi 4 Model B Deployment

This directory contains the Raspberry Pi 4 Model B implementation of the
automated analogue water-meter reading system.

The deployment integrates image acquisition, camera calibration, YOLOv10
meter and digit detection, CNN-based digit recognition, pointer-sector
classification, reading synthesis, and a graphical user interface (GUI).

## 1. Deployment Platform

- **Hardware:** Raspberry Pi 4 Model B
- **Operating System:** Raspberry Pi OS
- **Camera:** Raspberry Pi-compatible camera
- **Programming:** Python 3
- **Meter Detection:** YOLOv10
- **Digit Recognition:** CNN
- **Pointer Classification:** CNN
- **Deployment Models:** PyTorch / ONNX
- **Interface:** Desktop GUI

## 2. System Processing Pipeline

The deployed system processes an analogue water-meter image through the
following stages:

```text
Camera / Image Input
        ↓
Calibration & Preprocessing
        ↓
YOLOv10 Meter Detection
        ↓
Meter Region
        ↓
 ┌───────────────┬───────────────┐
 ↓                               ↓
Odometer Branch              Pointer Branch
 ↓                               ↓
YOLOv10 Digit Detection      Pointer Detection
 ↓                               ↓
Digit CNN                    Pointer CNN
 ↓                               ↓
Odometer Reading             Pointer Reading
 └───────────────┬─────────────┘
                 ↓
          Reading Synthesis
                 ↓
          Final Meter Reading
                 ↓
             GUI Display
