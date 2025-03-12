# YOLO Hotdog Classifier

A computer vision model that classifies images as either "hotdog" or "not hotdog", inspired by the Silicon Valley TV show. This project uses YOLOv8 for image classification.

## Project Description

This classifier uses a fine-tuned YOLOv8 model to determine whether an image contains a hotdog or not. The model achieves approximately 90.6% accuracy on the validation dataset and has a small footprint of only 3.0MB, making it suitable for Sony IMX500 onboard AI accelerator. 

## Features

- Binary image classification (hotdog/not hotdog)
- Fast inference (~1.5ms per image on CPU)
- Small model size (~3.0MB)
- Simple training and inference pipeline

## Dataset Structure

The dataset is organized as follows:

```
dataset/
├── train/
│   ├── hotdog/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── not_hotdog/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── hotdog/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── not_hotdog/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

Each folder contains images of the respective class for training and validation.

## Installation and Setup

### Requirements

- Python 3.8 or higher
- PyTorch 1.10 or higher
- Ultralytics YOLO package

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/yolo-hotdog-classifier.git
   cd yolo-hotdog-classifier
   ```

2. Install the required packages:
   ```
   pip install ultralytics torch torchvision
   ```

## Usage

### Training

To train the model with default parameters:

```
python train.py
```

The script automatically:
- Loads the dataset from the `dataset` directory
- Initializes a YOLOv8n classification model
- Trains for 50 epochs with batch size 16
- Saves the best model weights to `runs/classify/hotdog_classifier/weights/best.pt`

You can customize training parameters by modifying `train.py`.

### Inference

To classify new images:

```
python predict.py --source path/to/your/image.jpg
```

Or for multiple images or a directory:

```
python predict.py --source path/to/directory
```

## Model Performance

- Accuracy: 90.6% on validation set
- Inference speed: ~1.5ms per image on CPU
- Model size: ~3.0MB

## Hardware Support

The model can run on:
- CPU (default)
- NVIDIA GPU (with CUDA)
- AMD GPU (with ROCm support - requires additional setup)

## License

[Insert your license information here]

## Acknowledgements

- Ultralytics for the YOLO implementation
- [Add any other acknowledgements here]

﻿# Hot Dog Classifier

A YOLOv8-based image classifier that can distinguish between hot dog and not hot dog.

## Project Overview

This project uses YOLOv8's classification model to identify whether an image contains a hot dog or not. The model has been trained on a custom dataset of hot dog and non-hot dog images.

## Model Performance

- Training dataset: 498 images (balanced between hot dogs and non-hot dogs)
- Validation dataset: 100 images
- Accuracy: 100% on validation set
- Inference speed: 1.6ms per image

## Directory Structure

`
yolo_hotdog_classifier/
├── data.yaml         # Dataset configuration
├── train.py          # Training script
├── dataset/          # Dataset directory
│   ├── train/        # Training images
│   └── val/          # Validation images
└── runs/             # Training results and weights
`

## Usage

1. Install requirements:
   `Bash
   pip install ultralytics
   `

2. Train the model:
   `Bash
   python train.py
   `

The trained model weights will be saved in the 'runs' directory.
