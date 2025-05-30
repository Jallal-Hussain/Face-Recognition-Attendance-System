"""
Handles data loading and preprocessing for the face recognition dataset.
"""

import os
from PIL import Image

def load_dataset(dataset_path):
    """
    Loads the dataset from the specified path.

    Args:
        dataset_path (str): The path to the dataset directory.

    Returns:
        dict: A dictionary mapping student IDs to lists of image paths.
    """
    dataset = {}
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return dataset

    for student_id in os.listdir(dataset_path):
        student_dir = os.path.join(dataset_path, student_id)
        if os.path.isdir(student_dir):
            image_files = [os.path.join(student_dir, f) for f in os.listdir(student_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            dataset[student_id] = image_files
    return dataset

def preprocess_image(image_path):
    """
    Preprocesses a single image for model inference.

    Args:
        image_path (str): The path to the image file.

    Returns:
        PIL.Image.Image: The preprocessed image, or None if an error occurs.
    """
    try:
        img = Image.open(image_path).convert("RGB") # Ensure image is in RGB format
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None 