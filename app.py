"""
Integrates all modules and runs the Gradio interface.
"""

import gradio as gr
from data_loader import preprocess_image
from model import (
    load_recognition_model,
    recognize_face,
    generate_and_save_embeddings,
    load_embeddings,
)
import attendance  # Import the entire attendance module
import os

# --- Configuration ----
dataset_path = "dataset/"
# ----------------------

# Load model and processor
print("Loading model...")
model, processor = load_recognition_model()

# Load or generate embeddings
known_embeddings = load_embeddings()
if known_embeddings is None:
    print("Embeddings not found. Generating from dataset...")
    # Ensure the dataset directory exists before generating embeddings
    if not os.path.exists(dataset_path):
        print(
            f"Error: Dataset directory '{dataset_path}' not found. Cannot generate embeddings."
        )
        # Exit or handle error appropriately if dataset is missing
    else:
        generate_and_save_embeddings(dataset_path, model, processor)
        known_embeddings = load_embeddings()  # Load again after generation

# Extract student IDs from known embeddings for recognition function
student_ids = list(known_embeddings.keys()) if known_embeddings else []


def load_and_display_attendance():
    """
    Loads attendance records and returns them as a pandas DataFrame.
    """
    print("Loading attendance records for display...")
    df = attendance.load_attendance_records()
    # Optionally, format the date/time columns for better display
    return df


def attendance_system(image):
    """
    Processes the input image, performs face recognition, and records attendance.

    Args:
        image (PIL.Image.Image): The input image from Gradio.

    Returns:
        str: The attendance status message.
    """
    if (
        model is None
        or processor is None
        or known_embeddings is None
        or not student_ids
    ):
        return (
            "System not fully initialized. Model, embeddings, or student data missing."
        )

    # 1. Preprocess the image.
    # The input `image` from Gradio with type="pil" is already a PIL Image.
    # We might need additional preprocessing based on the model's requirements,
    # but preprocess_image in data_loader is designed for file paths.
    # Let's assume the PIL image is sufficient for the processor for now.
    # If the model requires a specific format, conversion would happen in recognize_face or here.

    # 2. Perform face recognition to get student_id and similarity score.
    predicted_student_id, similarity_score = recognize_face(
        image, model, processor, known_embeddings, student_ids
    )

    if predicted_student_id:
        # 3. Call record_attendance with the student_id.
        message = attendance.record_attendance(predicted_student_id)
        # Append similarity score to the message
        message += f" (Confidence: {similarity_score:.4f})"
    elif similarity_score is not None:
        # Case where no match is found but a similarity score was calculated
        message = f"Face not recognized. (Highest Confidence: {similarity_score:.4f})"
    else:
        message = "Face not recognized."

    # 4. Return the message from record_attendance.
    return message


# Set up the Gradio interface.
if __name__ == "__main__":
    if (
        model is not None
        and processor is not None
        and known_embeddings is not None
        and student_ids
    ):
        # Set up the Gradio interface with theme and title
        with gr.Blocks(theme=gr.themes.Default(), title="Face Recognition System") as demo:
            gr.Label("Face Recognition Attendence System") # Main label for the app
            gr.Markdown("Upload an image to record your attendance (Check-in/Check-out)."
            )

            # Main layout row with two columns
            with gr.Row():
                # Left Column: Status and Attendance Records
                with gr.Column(scale=1):
                    output_message = gr.Textbox(label="Status")

                    gr.Markdown("## Attendance Records")
                    with gr.Row(): # Row for the refresh button
                        load_button = gr.Button("Refresh Attendance Records")

                    attendance_table = gr.DataFrame(label="Attendance Records")

                # Right Column: Image Upload and Buttons
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Image")

                    # Buttons below the image input
                    with gr.Row():
                        clear_button = gr.Button("Clear", scale=0)
                        submit_button = gr.Button("Submit", scale=0, variant="primary")

            # Event listeners
            submit_button.click(
                fn=attendance_system,
                inputs=image_input,
                outputs=output_message
            )

            clear_button.click(
                fn=lambda: [None, ""],
                inputs=None,
                outputs=[image_input, output_message]
            )

            load_button.click(
                fn=load_and_display_attendance,
                inputs=None,
                outputs=attendance_table
            )

        print("Launching Gradio interface...")
        demo.launch()
    else:
        print("Cannot launch Gradio interface due to initialization errors.")
