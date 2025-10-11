# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import remedies  # make sure remedies.py is in the same directory

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(page_title="🌿 Plant Disease Detection", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f9f3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------
# LOAD YOLOv8 MODEL
# --------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # update path if needed
    return model

model = load_model()

# --------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------
def predict_disease(img):
    results = model(img)
    pred_class = results[0].names[results[0].probs.top1]
    pred_conf = float(results[0].probs.top1conf)

    remedy = remedies.remedies.get(pred_class, "No remedy found for this class")

    return pred_class, pred_conf, remedy

# --------------------------------------------
# UI LAYOUT
# --------------------------------------------
st.title("🌿 Plant Disease Detection & Remedies")
st.write("Upload a leaf image to detect the disease and get a suggested remedy.")

col1, col2 = st.columns(2)

with col1:
    uploaded_img = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Disease & Get Remedy"):
            with st.spinner("Analyzing... Please wait..."):
                pred_class, pred_conf, remedy = predict_disease(img)

            st.success("✅ Detection Complete!")
            st.subheader("Prediction Results:")
            st.write(f"**Predicted Disease:** {pred_class}")
            st.write(f"**Confidence:** {pred_conf:.2f}")
            st.write(f"**Recommended Remedy:**")
            st.info(remedy)
    else:
        st.warning("Please upload a leaf image to start detection.")

with col2:
    st.markdown(
        """
        ### 🧾 About this App  
        - Uses **YOLOv8 Classification** for plant disease detection  
        - Displays **confidence score** and **remedy suggestions**  
        - Built with **Streamlit**  
        """
    )

# --------------------------------------------
# RUN COMMAND
# --------------------------------------------
# Run locally:   streamlit run app.py
# Deploy on Streamlit Cloud: push this file + remedies.py + model to GitHub
