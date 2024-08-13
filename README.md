# YOLOv1 Implementation with Keras

This project is an implementation of the You Only Look Once (YOLO) v1 object detection algorithm using Keras and TensorFlow based on the original YOLO paper. YOLO is a state-of-the-art, real-time object detection system that can detect multiple objects in an image in a single forward pass of the neural network.

## Overview

This YOLOv1 implementation:

- Detects multiple objects in images
- Uses the PASCAL VOC dataset for training
- Implements the original YOLOv1 architecture
- Utilizes a custom loss function as described in the YOLO paper
- Employs a learning rate scheduling strategy for training

## Key Features

- **Real-time Object Detection**: Capable of processing images quickly for real-time applications.
- **Multiple Object Detection**: Can detect multiple objects of different classes in a single image.
- **PASCAL VOC Dataset**: Trained on the PASCAL Visual Object Classes dataset, which includes 20 object categories.
- **Custom Data Generator**: Efficiently prepares and augments training data.
- **Learning Rate Scheduling**: Implements a custom learning rate scheduler to optimize training.

## Project Structure

- `model.py`: Defines the YOLOv1 model architecture
- `dataset.py`: Handles data loading and preprocessing
- `loss.py`: Implements the YOLO loss function
- `train.py`: Contains the training loop and configuration
- `utils.py`: Provides utility functions including the learning rate scheduler

## Dataset

This implementation uses the PASCAL VOC 2007 dataset, which includes:

- 20 object categories
- 9,963 images containing 24,640 annotated objects
- Training, validation, and test sets

The dataset is automatically downloaded and prepared using TensorFlow Datasets.

## Training

The model is trained on the PASCAL VOC 2007 dataset. The training process includes:

- Data augmentation to increase the diversity of the training set
- Custom learning rate scheduling to optimize the training process
- Checkpointing to save the best model weights

## Usage

To train the model:

```
python train.py
```
