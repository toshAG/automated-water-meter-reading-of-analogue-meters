# Raspberry Pi 4 Model B Deployment

This directory contains the implementation files used to deploy the
automated water meter reading pipeline on a Raspberry Pi 4 Model B.

The Raspberry Pi deployment integrates the complete meter-reading
pipeline, including camera acquisition, camera calibration, meter
detection, odometer digit recognition, pointer-sector classification,
reading synthesis, and graphical user interface components.

## Deployment Platform

- Hardware: Raspberry Pi 4 Model B
- Operating System: Raspberry Pi OS
- Camera: Raspberry Pi-compatible camera
- Python-based processing pipeline
- YOLOv10 object detection
- CNN-based digit recognition
- CNN-based pointer-sector classification
- ONNX model support for deployment
- Desktop graphical user interface

## Complete Processing Pipeline

The deployed system follows the following sequence:

Captured Meter Image
        ↓
Camera Acquisition
        ↓
Camera Calibration
        ↓
YOLOv10 Global Meter Detection
        ↓
Meter Region Segmentation
        ↓
 ┌───────────────────────────────┐
 │                               │
 ↓                               ↓
Odometer Region              Pointer Region
 │                               │
 ↓                               ↓
YOLOv10 Digit Localization   Pointer CNN
 │                               │
 ↓                               ↓
Digit CNN Classification     Angular-Sector Classification
 │                               │
 └───────────────┬───────────────┘
                 ↓
        Reading Synthesis
                 ↓
        Final Meter Reading

## Main Deployment Components

### Camera Acquisition

`camera.py`

Handles image acquisition from the camera used by the Raspberry Pi
deployment.

### Camera Calibration

`calibrate_camera.py`

Provides the camera calibration functionality required to obtain
consistent image acquisition and processing.

### Main Meter Pipeline

`meter_pipeline.py`

Coordinates the automated meter-reading processing stages and connects
the detection, recognition, classification, and reading-synthesis
components.

### Desktop Application

`desktop_app.py`

Provides the graphical interface used to operate the automated meter
reading system.

### GUI Utilities

`gui_utils.py`

Contains supporting functions used by the graphical user interface.

### Image Loading

`image_loader.py`

Provides image loading and preparation functionality for the processing
pipeline.

### Model Conversion

`convert_models.py`

Provides functionality for preparing trained models for deployment,
including conversion to deployment-compatible formats.

## Trained Models

The deployment uses trained machine-learning models developed during
the research project.

The main model components include:

- YOLOv10 model for global meter detection
- YOLOv10 model for odometer digit localization
- CNN model for digit recognition
- CNN model for pointer-sector classification

Where appropriate, converted ONNX models are used to support deployment
on the Raspberry Pi.

## Startup

The deployment includes startup files that can be used to launch the
meter-reading application on the Raspberry Pi.

These include:

- `launch_meter_reader.sh`
- `Meter_Reader.desktop`

## Deployment Workflow

The Raspberry Pi receives an image of the analogue water meter through
the connected camera. The image is processed by the deployed machine
learning pipeline.

The system first detects the meter region and separates the relevant
odometer and pointer regions. The odometer branch localizes individual
digits and classifies them using the trained digit-recognition CNN.
The pointer branch determines the angular sector corresponding to the
pointer position.

The outputs from the two branches are subsequently combined to produce
the final automated meter reading.

## Repository Scope

This repository provides the software implementation, deployment
configuration, selected trained models, and representative results
required to demonstrate the developed automated meter-reading system.

The complete original training datasets and intermediate experimental
files are not included in this repository. Selected representative
data and results are provided where appropriate.

## Research Reproducibility

The repository is intended to provide sufficient information to
understand the architecture, software implementation, trained model
components, and Raspberry Pi deployment of the proposed system.
