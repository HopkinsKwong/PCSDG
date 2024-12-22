import os
import numpy as np
from PIL import Image

def find_center_of_mass(label_array):
    labeled_pixels = np.argwhere(label_array > 0)
    center_y, center_x = np.mean(labeled_pixels, axis=0)
    return int(center_x), int(center_y)

def pad_image(image, pad_size):
    return Image.fromarray(np.pad(np.array(image), ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant'))

def pad_label(label_array, pad_size):
    return np.pad(label_array, pad_size, mode='constant')

base_folder = "REFUGE"
subfolders = ["REFUGE_Test", "REFUGE_Train", "REFUGE_Validation"]

output_size = 800
padding_size = 400

for subfolder in subfolders:
    image_folder = os.path.join(base_folder, subfolder)
    label_folder = os.path.join(base_folder, subfolder)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_file = image_file.replace('.jpg', '.bmp')
        label_path = os.path.join(label_folder, label_file)

        image = Image.open(image_path)
        label = Image.open(label_path)

        padded_image = pad_image(image, padding_size)
        padded_label = pad_label(np.array(label), padding_size)

        label_array = np.array(label)
        center_x, center_y = find_center_of_mass(label_array)

        left = center_x + padding_size - output_size // 2
        top = center_y + padding_size - output_size // 2

        new_image = padded_image.crop((left, top, left + output_size, top + output_size))
        new_label = Image.fromarray(padded_label[top:top+output_size, left:left+output_size])

        new_image_path = image_path
        new_label_path = label_path

        new_image.save(new_image_path)
        new_label.save(new_label_path)

        print(f"Processed: {image_file}")

print("Processing complete.")
