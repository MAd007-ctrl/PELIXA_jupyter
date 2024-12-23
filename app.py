import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from transformers import pipeline

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the intent classification model with a specific model
nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible intents
intents = ["What is melanoma?", "What is skin cancer?", "How can I upload an image?", "What are the symptoms of skin lesions?", "Help"]

# Define the correct class labels mapping
class_labels = {
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    1: ('bcc', 'Basal cell carcinoma'),
    2: ('bkl', 'Benign keratosis-like lesions'),
    3: ('df', 'Dermatofibroma'),
    4: ('nv', 'Melanocytic nevi'),
    5: ('vasc', 'Pyogenic granulomas and hemorrhage'),
    6: ('mel', 'Melanoma')
}

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('final_model.h5')

model = load_model()

# Load mean and std from training
@st.cache_resource
def load_mean_std():
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    return mean, std

mean, std = load_mean_std()

# Define the image preprocessing function
def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to match model input
    image = np.asarray(image)
    image = (image - mean) / std  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title('PELIXA AI - Your Personal Assistant')

# Text input for user queries
user_input = st.text_input("How can I help you? (e.g., 'What is melanoma?')")

# Advanced NLP response logic using zero-shot classification
if user_input:
    result = nlp_model(user_input, candidate_labels=intents)
    intent = result['labels'][0]  # Get the top intent

    if intent == "What is melanoma?":
        st.write("Melanoma is a type of skin cancer that develops from the pigment-producing cells known as melanocytes.")
    elif intent == "What is skin cancer?":
        st.write("Skin cancer is the abnormal growth of skin cells, often caused by exposure to UV radiation.")
    elif intent == "How can I upload an image?":
        st.write("You can upload an image of a skin lesion using the file uploader below.")
    elif intent == "What are the symptoms of skin lesions?":
        st.write("Common symptoms include changes in the size, shape, or color of a mole or skin growth.")
    elif intent == "Help":
        st.write("You can ask me about skin conditions or upload an image for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)
    
    # Get predicted class index and name
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_idx][1]  # Full name
    confidence = predictions[0][predicted_class_idx]
    
    # Display results
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
    
    # Display bar chart of all predictions
    st.bar_chart(dict(zip([label[1] for label in class_labels.values()], predictions[0])))

st.write("Note: This is a prototype and should not be used for medical diagnosis.")
