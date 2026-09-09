# Model Development and Training

This directory contains the supporting materials used to develop and evaluate
the machine-learning models for automated reading of analogue water meters.

The model-development workflow was performed before deployment on the
Raspberry Pi 4 Model B. The training materials provided here document the
main stages of image preparation, annotation, model training, recognition,
classification, and evaluation.

## Model Development Pipeline

The complete development process consists of the following stages:

**Water Meter Images**  
↓  
**Image Preparation and Annotation**  
↓  
**YOLOv10 Meter Detection**  
↓  
**Meter Region Extraction**  
↓  
**Odometer and Pointer Region Preparation**  
↓  
**YOLOv10 Odometer Digit Detection**  
↓  
**CNN Digit Recognition**  
↓  
**Pointer Detection and Sector Classification**  
↓  
**Reading Synthesis**  
↓  
**Model Evaluation**

## Main Models Developed

### 1. Global Meter Detection

YOLOv10 was trained to identify and localize the water-meter region
within the input image.

### 2. Odometer Digit Detection

A second YOLOv10 model was developed to localize the individual digits
of the odometer display.

### 3. Digit Recognition

A CNN-based digit-recognition model was trained to classify the detected
odometer digits into the classes 0–9.

### 4. Pointer-Sector Classification

A CNN-based classification model was developed to determine the pointer
sector and support conversion of the analogue pointer position into a
numeric value.

## Supporting Training Materials

The directory contains selected source codes, training scripts, and
representative images used during model development.

```text
model_development/
│
├── README.md
│
├── codes/
│   ├── training_pipeline_1.py
│   ├── training_pipeline_2.py
│   ├── digit_training.py
│   ├── pointer_training.py
│   └── evaluation.py
│
└── images/
    ├── dataset_samples/
    ├── annotated_images/
    ├── training_outputs/
    └── prediction_results/
