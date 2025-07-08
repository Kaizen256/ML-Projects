import os
import shutil

val_dir = 'tiny-imagenet-200/val'
val_images_dir = os.path.join(val_dir, 'images')
annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# Read the annotations file
val_annotations = {}
with open(annotations_file, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split('\t')
        img_file, class_name = parts[0], parts[1]
        val_annotations[img_file] = class_name

# Create subfolders and move images
for img_file, class_name in val_annotations.items():
    class_dir = os.path.join(val_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    src = os.path.join(val_images_dir, img_file)
    dst = os.path.join(class_dir, img_file)
    shutil.move(src, dst)
