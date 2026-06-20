
Go Positions - v6 YOLOv8n-25epochs-resize640x640_aug3x
==============================

This dataset was exported via roboflow.com on November 8, 2023 at 12:31 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2400 images.
Go-pieces are annotated in YOLO v4 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 15 percent of the image
* Random rotation of between -10 and +10 degrees
* Random shear of between -2° to +2° horizontally and -2° to +2° vertically
* Random brigthness adjustment of between -20 and +20 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 1 pixels
* Salt and pepper noise was applied to 1 percent of pixels


