import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

from config import CLASS_NAMES, TRUCK_CLASSES, BASE_TOLL_RATES, AXLE_RATE

#PAGE CONFIG
st.set_page_config(page_title="Toll Calculator", layout="wide")

#HELPER CLASSES
class TrueDivide(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.math.truediv(inputs[0], inputs[1])
    def get_config(self):
        return super().get_config()

#MODEL BUILDING (MobileNetV2)
def build_mobilenet_model():
    """
    Structure: Input -> Preprocess -> MobileNetV2 -> GlobalAvgPool -> Dropout -> Dense
    """
    IMAGE_SHAPE = (224, 224, 3) 
    NUM_CLASSES = 11
    
    # Base model with ImageNet weights
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        weights=None # Loading custom weights later
    )
    
    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) 
    x = base_model(x, training=False) 
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

#MODEL LOADING
@st.cache_resource
def load_classification_model(model_choice):
    custom_objects = {'TrueDivide': TrueDivide}
    
    if model_choice == "CNN (Custom)":
        return tf.keras.models.load_model('Vehicle Classification/CNN/basic1_dataaugmented.keras', custom_objects=custom_objects)
    
    elif model_choice == "MobileNetV2":
        model = build_mobilenet_model()
        weights_path = 'Vehicle Classification/MobileNetV2/mio_tcd_classifier_final.h5'
        try:
            model.load_weights(weights_path)
        except Exception as e:
            st.error(f"Error loading MobileNet weights: {e}")
            return None
        return model
        
    return None

@st.cache_resource
def load_axle_model(model_choice):
    if model_choice == "YOLOv8n":
        return YOLO('Axle Detection/YOLOv8n/best.pt')
    elif model_choice == "YOLOv10s":
        return YOLO('Axle Detection/YOLOv10s/best.pt')
    elif model_choice == "RT-DETR-Large":
        return YOLO('Axle Detection/RT-DETR-Large/best.pt') 
    return None

#PREPROCESSING
def preprocess_image(image, model_name):
    # Convert to NumPy array
    if model_name == "CNN (Custom)":
        target_size = (128, 128)
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32)
        
    elif model_name == "MobileNetV2":
        target_size = (224, 224)
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32)
        
    else:
        return None

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

#MAIN APP
st.title("Vehicle Classifier, Axle Counter & Toll Calculator")

# Sidebar
with st.sidebar:
    st.header("Model Settings")

    st.markdown("""
    **Recommended Models:**
    * Classification: **CNN (Custom)**
    * Axle Detection: **RT-DETR-Large**
    """)
    st.divider()

    clf_model_name = st.selectbox("Classification Model", ["CNN (Custom)", "MobileNetV2"])
    
    axle_model_name = st.selectbox("Axle Detection Model", ["RT-DETR-Large", "YOLOv10s", "YOLOv8n"])
    
    conf_threshold = st.slider("Axle Detection Confidence", 0.1, 1.0, 0.30)
    
    if axle_model_name == "RT-DETR-Large":
        st.info("RT-DETR Selected: slower but higher accuracy.")

    st.divider()
    st.info("Please upload a clear picture with visible wheels.")
    st.warning("Prediction results may not be accurate.")

# File Uploader
uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Input Image")
        st.image(original_image, use_container_width=True)

    if st.button("Calculate Toll", type="primary"):
        with st.spinner("Processing image..."):
            
            #classify
            clf_model = load_classification_model(clf_model_name)
            
            if clf_model:
                processed_img = preprocess_image(original_image, clf_model_name)
                
                predictions = clf_model.predict(processed_img)
                score = tf.nn.softmax(predictions[0])
                class_idx = np.argmax(predictions[0])
                
                raw_class = CLASS_NAMES[class_idx]
                predicted_class = raw_class.lower().replace(" ", "_")
                
                confidence = 100 * np.max(score)
                base_toll = BASE_TOLL_RATES.get(predicted_class, 0)
                
                with col2:
                    st.subheader("Analysis Results")
                    st.success(f"**Vehicle Type:** {predicted_class.replace('_', ' ').title()}")
                    st.caption(f"Confidence: {confidence:.2f}%")
                    st.metric("Base Toll", f"₹{base_toll}")

                #detect axle
                total_toll = base_toll
                
                if predicted_class in TRUCK_CLASSES:
                    st.divider()
                    st.write(f"**Heavy Vehicle Detected** - Counting Axles with {axle_model_name}...")
                    
                    axle_model = load_axle_model(axle_model_name)
                    
                    if axle_model:
                        #predict
                        results = axle_model.predict(original_image, conf=conf_threshold, verbose=False)
                        result = results[0]
                        
                        # Count only axle (class 0)
                        axle_count = 0
                        for box in result.boxes:
                            if int(box.cls[0]) == 0:
                                axle_count += 1
                        
                        # Generate Plot
                        annotated_array = result.plot(line_width=2, font_size=10)
                        annotated_pil = Image.fromarray(annotated_array[..., ::-1]) # BGR to RGB

                        # Upscaling for RT-DETR visibility
                        if axle_model_name == "RT-DETR-Large":
                            w, h = annotated_pil.size
                            annotated_pil = annotated_pil.resize((w*2, h*2), Image.Resampling.LANCZOS)
                            st.caption("Output upscaled 2x for visibility")

                        st.image(annotated_pil, caption=f"Axle Detection ({axle_model_name})", use_container_width=True)
                        
                        # Toll Calculation
                        axle_cost = axle_count * AXLE_RATE
                        total_toll += axle_cost
                        
                        st.info(f"Axles Detected: {axle_count} | Extra Charge: ₹{axle_cost}")
                    else:
                        st.error("Failed to load axle model.")
                
                else:
                    st.info(f"Vehicle type '{predicted_class}' does not require axle counting.")

                st.divider()
                st.metric(label="TOTAL TOLL AMOUNT", value=f"₹{total_toll}", delta_color="inverse")
            else:
                st.error("Failed to load classification model.")