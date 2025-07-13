import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 🌼 Page Configuration
st.set_page_config(page_title="🌸 Flower Classifier", page_icon="🌼", layout="centered")

# 🌒🌞 Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

# 🎨 Custom CSS
if dark_mode:
    bg_color = "#121212"
    text_color = "#ffffff"
    card_bg = "#1e1e1e"
else:
    bg_color = "#fbeef3"
    text_color = "#333333"
    card_bg = "#fff3cd"

st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .main {{
        background-color: {bg_color};
        padding: 2rem;
        border-radius: 12px;
    }}
    .title {{
        color: #ff4b91;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
    }}
    .result {{
        background-color: {card_bg};
        color: {text_color};
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
    }}
    </style>
""", unsafe_allow_html=True)

# 🌼 Sidebar content
st.sidebar.header("🌺 About the App")
st.sidebar.info("This app uses a Convolutional Neural Network (CNN) to classify flowers into five types:\n\n"
                "- Daisy 🌼\n"
                "- Dandelion 🌾\n"
                "- Rose 🌹\n"
                "- Sunflower 🌻\n"
                "- Tulip 🌷")

st.sidebar.markdown("---")
st.sidebar.header("🌸 Fun Flower Fact")
st.sidebar.success("🌹 Roses are one of the oldest flowers and have existed for about 35 million years!")

# 🌸 App Title
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🌸 Flower Classification CNN Model 🌸</div>', unsafe_allow_html=True)

# 🌼 Load Model
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model = load_model('Flower_Recog_Model.h5')

# 📊 Classification Function
def classify_images(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_expanded = tf.expand_dims(img_array, axis=0)

    prediction = model.predict(img_expanded)
    scores = tf.nn.softmax(prediction[0]).numpy()
    predicted_label = flower_names[np.argmax(scores)]
    confidence = np.max(scores) * 100
    return predicted_label, confidence, scores

# 📁 Upload Image
uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    os.makedirs("upload", exist_ok=True)
    path = os.path.join("upload", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="🌼 Uploaded Image", width=250)

    label, confidence, all_scores = classify_images(path)
    st.markdown(
        f'<div class="result">🌺 The image is classified as <b>{label}</b> '
        f'with a confidence of <b>{confidence:.2f}%</b>.</div>',
        unsafe_allow_html=True
    )

    # 📊 Bar Chart of Prediction Scores
    st.subheader("📊 Prediction Confidence Scores")
    fig, ax = plt.subplots()
    ax.barh(flower_names, all_scores * 100, color="#ff66b2" if not dark_mode else "#ffccff")
    ax.set_xlabel("Confidence (%)", color=text_color)
    ax.set_xlim(0, 100)
    ax.set_facecolor(bg_color)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)
