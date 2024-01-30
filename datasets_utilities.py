# Some imports
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


# Define labels
class_mapping = {
        'Bedroom':0,
        'Coast': 1,
        'Industrial': 2,
        'Highway': 3,
        'Forest': 4,
        'InsideCity': 5,
        'Kitchen': 6,
        'Office': 7,
        'Mountain': 8,
        'LivingRoom': 9,
        'Store': 10,
        'OpenCountry': 11,
        'Street': 12,
        'TallBuilding': 13,
        'Suburb': 14
    }


# Function to show some images from a list (useful for testing)
def show_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        ax = axes[i]
        image = images[i].numpy().transpose((1, 2, 0))  # Change tensor shape to (H, W, C) for display
        label = labels[i]
        class_name = [key for key, value in class_mapping.items() if value == label][0]

        ax.imshow(image.astype(np.uint8))
        ax.set_title(f"Class: {class_name}")
        ax.axis("off")

    plt.show()


# Function to load images from a folder
def load_images_from_folder(folder, transform):
    images = []
    labels = []

    class_folders = [class_folder for class_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, class_folder))][0:]
    for class_folder in class_folders:
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            class_label = class_mapping[class_folder]

            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                image = transform(Image.open(image_path))
                images.append(image)
                labels.append(class_label)

    return images, labels
    

# Function to generate new images by horizontally flipping some given
def flip_images(set):
    new_images=[]
    images_to_flip=set.copy()
    for image in images_to_flip:
      new_images.append(torch.flip(image, dims=[2]) ) # 2 corresponds to the width axis

    return new_images