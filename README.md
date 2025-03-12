# YOLO Hotdog Classifier

A computer vision model that classifies images as either "hotdog" or "not hotdog", inspired by the Silicon Valley TV show. This project uses YOLOv8 for image classification.

## Project Description

This classifier uses a fine-tuned YOLOv8 model to determine whether an image contains a hotdog or not. The model achieves approximately 90.6% accuracy on the validation dataset and has a small footprint of only 3.0MB, making it suitable for various applications including mobile.

## Features

- Binary image classification (hotdog/not hotdog)
- Fast inference (~1.5ms per image on CPU)
- Small model size (~3.0MB)
- Simple training and inference pipeline

## Dataset Information

The dataset consists of:
- Training dataset: 498 images (balanced between hotdogs and non-hotdogs)
- Validation dataset: 100 images

### Dataset Structure

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

### Dataset Preparation

The repository includes a `split_dataset.py` script that helps you prepare your dataset by:
1. Splitting your dataset into training and validation sets
2. Organizing images into the required folder structure
3. Balancing classes if needed

Usage:
```
python split_dataset.py --source your_images_folder --train_ratio 0.8
```

### Data Configuration

The dataset configuration is specified in `data.yaml`:

```yaml
# Dataset paths
path: ./dataset  # Root directory
train: train/    # Training data relative to path
val: val/        # Validation data relative to path

# Classes
names:
  0: hotdog
  1: not_hotdog
```

## Installation and Setup

### Requirements

- Python 3.8 or higher
- PyTorch 1.10 or higher
- Ultralytics YOLO package

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/Ruxpins/yolo-hotdog-classifier.git
   cd yolo-hotdog-classifier
   ```

2. Install the required packages using requirements.txt:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
yolo_hotdog_classifier/
├── data.yaml         # Dataset configuration
├── train.py          # Training script
├── predict.py        # Inference script
├── split_dataset.py  # Dataset preparation tool
├── requirements.txt  # Dependencies
├── dataset/          # Dataset directory
│   ├── train/        # Training images
│   └── val/          # Validation images
└── runs/             # Training results and weights
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
