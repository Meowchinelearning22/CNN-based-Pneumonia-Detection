import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("pneumonia_model.h5")

st.title("Pneumonia Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file).convert("RGB").resize((224,224))
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("Prediction: PNEUMONIA")
    else:
        st.success("Prediction: NORMAL")