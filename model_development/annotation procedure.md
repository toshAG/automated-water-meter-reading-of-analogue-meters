# Annotation Procedure

## Annotation Tool

Image annotation was performed using the **DigitalSreeni Image Annotator** tool.

The annotation process was used to prepare the images required for training the meter-detection and pointer-detection models.

## Annotation Classes

The annotated images contained the following classes:

- `adomometer part`
- `pointer_1`
- `pointer_2`
- `pointer_3`
- `pointer_4`

## Odometer Annotation

The odometer region was annotated using the **rectangular bounding-box tool**. A rectangular region was drawn around the odometer display containing the meter digits.

This annotation was used to identify and localize the odometer region for subsequent digit recognition.

## Pointer Annotation

The meter pointer locations were annotated using the **Paint Brush tool**. The brush was used to mark the pointer regions corresponding to the different pointer positions/sectors.

The pointer annotations were subsequently used for pointer detection and classification.

## Annotation Workflow

The general annotation procedure was:

1. Load the water-meter image into the DigitalSreeni Image Annotator.
2. Define the required annotation classes.
3. Use rectangular bounding boxes to annotate the odometer region.
4. Use the Paint Brush tool to annotate the pointer regions.
5. Review the annotations for consistency and completeness.
6. Save the annotated images and corresponding annotations for model development.

## Annotated Sample

Representative annotated images are provided in the [`images`](./images/) directory.

The images illustrate the annotated odometer region and the pointer locations used during model development.
