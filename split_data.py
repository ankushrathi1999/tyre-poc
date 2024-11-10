import os
import shutil
import math

# Set the split ratio (80% for training, 20% for validation)
split_ratio = 0.8

# Set the directory paths
root_dir = './tyre_annotations'
img_dir = os.path.join(root_dir, 'images')
label_dir = os.path.join(root_dir, 'labels')
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')

# Create the training and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get the list of image files
img_files = os.listdir(img_dir)

# Calculate the number of files for training and validation
num_files = len(img_files)
train_count = math.ceil(num_files * split_ratio)

# Split the data into training and validation folders
for i, img_file in enumerate(img_files):
    label_file = img_file.replace('.jpg', '.txt')
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, label_file)

    if i < train_count:
        target_dir = train_dir
    else:
        target_dir = val_dir

    # Create the image and label directories in the target folder
    img_target_dir = os.path.join(target_dir, 'images')
    label_target_dir = os.path.join(target_dir, 'labels')
    os.makedirs(img_target_dir, exist_ok=True)
    os.makedirs(label_target_dir, exist_ok=True)

    # Copy the image and label files to the target folder
    shutil.copy(img_path, os.path.join(img_target_dir, img_file))
    shutil.copy(label_path, os.path.join(label_target_dir, label_file))

    print(f"Copied {img_file} to {'train' if i < train_count else 'val'} folder.")
