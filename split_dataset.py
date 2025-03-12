import os
import shutil
import random
from pathlib import Path

# Set paths
base_path = 'C:/Users/Mandi/yolo_hotdog_classifier'
dataset_path = os.path.join(base_path, 'dataset')
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')

# Create directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Function to split and copy files
def split_files(src_dir, train_dir, val_dir, split=0.8):
    files = os.listdir(src_dir)
    n_files = len(files)
    n_train = int(n_files * split)
    
    # Randomly shuffle files
    random.shuffle(files)
    
    # Split into train and val
    train_files = files[:n_train]
    val_files = files[n_train:]
    
    # Copy files to respective directories
    for f in train_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(val_dir, f))

# Process hot dog images
os.makedirs(os.path.join(train_path, 'hot_dog'), exist_ok=True)
os.makedirs(os.path.join(val_path, 'hot_dog'), exist_ok=True)
split_files(
    'C:/Users/Mandi/Documents/archive-1/train/hot_dog',
    os.path.join(train_path, 'hot_dog'),
    os.path.join(val_path, 'hot_dog')
)

# Process not hot dog images
os.makedirs(os.path.join(train_path, 'not_hot_dog'), exist_ok=True)
os.makedirs(os.path.join(val_path, 'not_hot_dog'), exist_ok=True)
split_files(
    'C:/Users/Mandi/Documents/archive-1/train/not_hot_dog',
    os.path.join(train_path, 'not_hot_dog'),
    os.path.join(val_path, 'not_hot_dog')
)

# Print dataset statistics
print('Dataset statistics:')
print(f'Training hot dogs: {len(os.listdir(os.path.join(train_path, "hot_dog")))}')
print(f'Training not hot dogs: {len(os.listdir(os.path.join(train_path, "not_hot_dog")))}')
print(f'Validation hot dogs: {len(os.listdir(os.path.join(val_path, "hot_dog")))}')
print(f'Validation not hot dogs: {len(os.listdir(os.path.join(val_path, "not_hot_dog")))}')
