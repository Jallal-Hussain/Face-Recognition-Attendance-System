"""
Contains model loading and inference functions.
"""

from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import os
import pickle
import torch.nn.functional as F

from data_loader import load_dataset, preprocess_image

# Implement a function to generate and store embeddings for the dataset.
# This will be needed for the recognize_face function to compare embeddings.
# A simple approach could be to store a dictionary mapping student_id to their average embedding.

EMBEDDINGS_FILE = "student_embeddings.pkl"

def generate_and_save_embeddings(dataset_path, model, processor):
    """
    Generates embeddings for all images in the dataset and saves them.

    Args:
        dataset_path (str): The path to the dataset directory.
        model: The loaded model.
        processor: The loaded processor.
    """
    print("Generating embeddings...")
    dataset = load_dataset(dataset_path)
    student_embeddings = {}

    for student_id, image_paths in dataset.items():
        embeddings = []
        for image_path in image_paths:
            img = preprocess_image(image_path)
            if img:
                try:
                    # Process image and get embeddings
                    inputs = processor(images=img, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        # Extract features before the classification head. For ViT, this is typically the pooled output
                        # or the representation of the [CLS] token from the last hidden state.
                        # Let's use the pooled output if available, otherwise the [CLS] token.
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                             embedding = outputs.pooler_output.squeeze().numpy()
                        else:
                             # Assuming [CLS] token is the first token in the last hidden state
                             embedding = outputs.hidden_states[-1][:, 0, :].squeeze().numpy()

                    embeddings.append(embedding)
                    print("reg no ", image_paths)
                except Exception as e:
                    print(f"Error generating embedding for {image_path}: {e}")

        if embeddings:
            # Average the embeddings for each student
            avg_embedding = torch.mean(torch.tensor(embeddings), dim=0).numpy()
            student_embeddings[student_id] = avg_embedding

    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(student_embeddings, f)
    print(f"Embeddings generated and saved to {EMBEDDINGS_FILE}")

def load_embeddings():
    """
    Loads student embeddings from the pickle file.

    Returns:
        dict: A dictionary mapping student IDs to their embeddings, or None if the file is not found.
    """
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def load_recognition_model(model_name="jayanta/vit-base-patch16-224-in21k-face-recognition"):
    """
    Loads the face recognition model.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        tuple: The loaded model and processor.
    """
    try:
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        return model, processor
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def recognize_face(image, model, processor, known_embeddings, student_ids, threshold=0.6):
    """
    Recognizes a face in the given image by comparing its embedding with known embeddings.

    Args:
        image (PIL.Image.Image): The input image.
        model: The loaded model.
        processor: The loaded processor.
        known_embeddings (dict): Dictionary mapping student IDs to their embeddings.
        student_ids (list): List of known student IDs.
        threshold (float): Similarity threshold for recognition.

    Returns:
        str: The recognized student ID, or None if no match is found.
    """
    if not known_embeddings:
        print("Error: No known embeddings available for recognition.")
        return None, None

    try:
        # Process image and get embedding
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Extract features before the classification head (embedding)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                 image_embedding = outputs.pooler_output.squeeze()
            else:
                 # Assuming [CLS] token is the first token in the last hidden state
                 image_embedding = outputs.hidden_states[-1][:, 0, :].squeeze()

        # Ensure the image embedding is a torch tensor for calculations
        image_embedding = torch.tensor(image_embedding)

        # Compare the image embedding with known embeddings using cosine similarity
        max_similarity = -1
        recognized_id = None

        for student_id in student_ids:
            known_embedding = known_embeddings[student_id]
            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                torch.tensor(image_embedding).unsqueeze(0),
                torch.tensor(known_embedding).unsqueeze(0)
            ).item()

            if similarity > max_similarity:
                max_similarity = similarity
                recognized_id = student_id

        # Return recognized ID and similarity score if similarity exceeds threshold
        if max_similarity >= threshold and recognized_id is not None:
            print(f"Recognized {recognized_id} with similarity {max_similarity:.4f}")
            return recognized_id, max_similarity
        else:
            print(f"No match found (highest similarity {max_similarity:.4f})")
            return None, max_similarity

    except Exception as e:
        print(f"Error in face recognition: {e}")
        return None, None 