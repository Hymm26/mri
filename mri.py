import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Alzheimer MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

# Debug - show available files
st.sidebar.header("🔍 Debug Info")
available_files = [f for f in os.listdir(".") if "Alzheimer" in f or f.endswith(('.h5', '.keras'))]
st.sidebar.write("Available model files:", available_files)

# Check if model file exists
if not os.path.exists("Alzheimer_MRI_CNN_augmented.keras"):
    st.sidebar.error("❌ Model file not found!")
else:
    st.sidebar.success("✅ Model file found!")

# Title and description
st.title("🧠 Alzheimer MRI Classification")
st.markdown("""
This AI model classifies brain MRI scans into 4 categories:
- **Non Demented** 🟢
- **Very Mild Demented** 🟡  
- **Mild Demented** 🟠
- **Moderate Demented** 🔴
""")

# Load model with caching - UPDATED VERSION
@st.cache_resource
def load_model():
    try:
        # Try loading the .keras model first
        model = tf.keras.models.load_model('Alzheimer_MRI_CNN_augmented.keras')
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        
        # Try alternative model names
        alternative_models = [
            "Alzheimer_MRI_CNN.h5",
            "Alzheimer_Final_Model.h5", 
            "model.h5"
        ]
        
        for alt_model in alternative_models:
            try:
                if os.path.exists(alt_model):
                    model = tf.keras.models.load_model(alt_model)
                    st.sidebar.success(f"✅ Loaded alternative model: {alt_model}")
                    return model
            except:
                continue
        
        return None

# Class names
class_names = {
    0: "Mild Demented",
    1: "Moderate Demented", 
    2: "Non Demented",
    3: "Very Mild Demented"
}

# Class colors for visualization
class_colors = {
    "Non Demented": "#2ecc71",  # Green
    "Very Mild Demented": "#f1c40f",  # Yellow
    "Mild Demented": "#e67e22",  # Orange
    "Moderate Demented": "#e74c3c"  # Red
}

# Preprocess image
def preprocess_image(image):
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 128x128
    image = image.resize((128, 128))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Reshape for model
    img_array = img_array.reshape(1, 128, 128, 1)
    
    return img_array

# Main function
def main():
    # Sidebar for upload
    st.sidebar.header("📤 Upload MRI Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a brain MRI image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan for Alzheimer classification"
    )
    
    # Load model
    model = load_model()
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            
            # Image info
            st.write(f"**Image Details:**")
            st.write(f"- Size: {image.size}")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- Format: {uploaded_file.type}")
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            if model is not None:
                # Preprocess and predict
                with st.spinner("Analyzing MRI scan..."):
                    try:
                        processed_image = preprocess_image(image)
                        predictions = model.predict(processed_image, verbose=0)
                        predicted_class = np.argmax(predictions)
                        confidence = np.max(predictions)
                        
                        # Get class name and color
                        result_class = class_names[predicted_class]
                        result_color = class_colors[result_class]
                        
                        # Display results
                        st.markdown(f"### 🎯 **Prediction: {result_class}**")
                        
                        # Confidence meter
                        st.markdown(f"### 📊 **Confidence: {confidence*100:.2f}%**")
                        st.progress(float(confidence))
                        
                        # Color-coded result
                        st.markdown(
                            f"<div style='background-color: {result_color}; padding: 20px; border-radius: 10px;'>"
                            f"<h3 style='color: white; text-align: center;'>Diagnosis: {result_class}</h3>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                        
                        # Detailed probabilities
                        st.subheader("📈 Detailed Probabilities")
                        for i, prob in enumerate(predictions[0]):
                            class_name = class_names[i]
                            color = class_colors[class_name]
                            
                            col_prob, col_bar = st.columns([2, 5])
                            with col_prob:
                                st.write(f"{class_name}:")
                            with col_bar:
                                st.progress(float(prob))
                            
                            st.write(f"`{prob*100:.2f}%`")
                        
                        # Interpretation
                        st.subheader("💡 Interpretation")
                        if result_class == "Non Demented":
                            st.success("✅ The MRI shows no significant signs of Alzheimer's disease.")
                        elif result_class == "Very Mild Demented":
                            st.warning("⚠️ Early signs of Alzheimer's detected. Regular monitoring recommended.")
                        elif result_class == "Mild Demented":
                            st.warning("🚨 Mild Alzheimer's symptoms detected. Medical consultation advised.")
                        else:
                            st.error("🚨 Moderate Alzheimer's detected. Immediate medical attention recommended.")
                            
                    except Exception as e:
                        st.error(f"❌ Prediction error: {e}")
            
            else:
                st.error("""
                ❌ Model not loaded. 
                
                **Possible solutions:**
                1. Make sure 'Alzheimer_MRI_CNN_augmented.keras' is in the same folder
                2. Check the debug info in sidebar
                3. Try uploading a different model file
                """)
                
                # Show mock results for demonstration
                st.warning("🔄 Showing mock results for demonstration:")
                st.success("**Mock Prediction:** Non Demented")
                st.info("**Confidence:** 92.5%")
                st.progress(0.925)
    
    else:
        # Demo section when no file is uploaded
        st.info("👆 Please upload a brain MRI image using the sidebar to get started.")
        
        # Sample layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Instructions")
            st.markdown("""
            1. **Upload** a brain MRI image using the sidebar
            2. **Wait** for the AI analysis (takes a few seconds)
            3. **View** the classification results and confidence scores
            4. **Read** the interpretation and recommendations
            
            **Supported formats:** JPG, JPEG, PNG
            **Recommended:** Clear MRI scans with visible brain structure
            """)
        
        with col2:
            st.subheader("🔧 System Status")
            if model is not None:
                st.success("✅ Model: LOADED")
                st.success("✅ System: READY")
            else:
                st.error("❌ Model: NOT LOADED")
                st.warning("⚠️ System: LIMITED FUNCTIONALITY")
            
            st.info("""
            **Model Details:**
            - Type: Convolutional Neural Network
            - Input: 128x128 grayscale images  
            - Classes: 4 Alzheimer stages
            - Accuracy: 97% on test data
            """)

# Footer
st.markdown("---")
st.markdown(
    "**Alzheimer MRI Classifier** | Final Graduation Project | "
    "For educational and research purposes only"
)

if __name__ == "__main__":
    main()