# Import dependencies
from ultralytics import YOLO
import gradio as gr
from PIL import Image
import remedies  # make sure remedies.py is in the same folder

# Load trained YOLOv8 classification model
model = YOLO("/content/drive/MyDrive/best.pt")  # path to your best.pt

# Prediction function
def predict_disease(img):
    """
    Takes a PIL image input, runs YOLOv8 classification,
    and returns: uploaded image, predicted disease, confidence, remedy.
    """
    # Run YOLO prediction
    results = model(img)

    # Get top prediction
    pred_class = results[0].names[results[0].probs.top1]
    pred_conf = float(results[0].probs.top1conf)  # convert to float

    # Get remedy from dictionary
    remedy = remedies.remedies.get(pred_class, "No remedy found for this class")

    return img, pred_class, f"{pred_conf:.2f}", remedy

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🌿 Plant Disease Detection & Remedies")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Leaf Image", type="pil")
            predict_btn = gr.Button("Detect Disease & Get Remedy")

        with gr.Column():
            image_output = gr.Image(label="Uploaded Image")
            disease_output = gr.Textbox(label="Predicted Disease")
            confidence_output = gr.Textbox(label="Confidence")
            remedy_output = gr.Textbox(label="Recommended Remedy", lines=10, interactive=False)

    # Connect button click to prediction function
    predict_btn.click(
        fn=predict_disease,
        inputs=image_input,
        outputs=[image_output, disease_output, confidence_output, remedy_output]
    )

# Launch app
demo.launch()
