import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import medicinal_details

plant_details = medicinal_details.Plant_Details

# Load the model and class dictionary
try:
    model = load_model(r'D:\Identification_plant\models\model_version_1.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    class_dict = np.load(r'D:\Identification_plant\models\label_map.npy', allow_pickle=True).item()
except Exception as e:
    st.error(f"Error loading label map: {e}")
    st.stop()

def predict(model: tf.keras.Model, image: np.ndarray) -> tuple:
    """
    Predict the class of an image using the given model.

    Args:
        model: The Keras model to use for prediction.
        image: The input image to predict.

    Returns:
        A tuple containing the predicted class name and confidence.
    """
    try:
        # Resize the image to the size that the model expects
        img_size = (256, 256)
        image = image.resize(img_size)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize if needed

        prediction = model.predict(image)
        st.write(f"Prediction array: {prediction}")

        predicted_class_index = np.argmax(prediction[0])
        st.write(f"Predicted class index: {predicted_class_index}")
        predicted_class_name = class_dict.get(predicted_class_index, "Unknown")
        st.write(f"Predicted class name: {predicted_class_name}")

        confidence = prediction[0][predicted_class_index] * 100  
        return predicted_class_name, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Unknown", 0.0

def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        st.markdown(
            f"""
            <style>
            .stApp {{
                position: relative;
                overflow: hidden;
            }}
            .stApp::before {{
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: url(data:image/png;base64,{encoded_string});
                background-size: cover;
                background-position: center;
                filter: blur(8px); /* Adjust the blur intensity */
                z-index: -1;
            }}
            .content {{
                position: relative;
                z-index: 1;
                padding: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error loading background image: {e}")

content = """
<p>Exploring the hidden world of medicinal plants through advanced image processing can reveal incredible insights. With the decline of many plant species, it is imperative to harness technology to preserve and protect these invaluable resources.</p> 
<p>By employing computer vision techniques, we aim to streamline the identification process, making it more accessible and efficient. This approach not only supports biodiversity but also contributes to the conservation of herbal knowledge for future generations.</p>
"""

def main():
    add_bg_from_local(r"D:\Identification_plant\static\flora_image.jpg")
    new_title =  '<p style="font-family:sans-serif; color: #000000; font-size: 42px;">Welcome to the VedantVerifier!</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(content, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            img = img.resize((256, 256))  # Adjusted to match the model's input size
            st.image(img)

            if st.button("Predict"):
                pred, confidence = predict(model, img)
                st.write(f"Prediction Raw Output: {pred}")
                st.write(f"Confidence Raw Output: {confidence:.2f}%")

                # Retrieve plant details safely
                plant_name = plant_details.get(pred, "Unknown")

                st.write(f"The given image is {pred} with a confidence of {confidence:.2f}%.")
                if plant_name != "Unknown":
                    st.write("Scientific Name: ", plant_name.get("scientificName", "N/A"))
                    st.write("Medicinal Property: ", plant_name.get("medicinalProperty", "N/A"))
                    st.write("Medicinal Details: ")
                    for detail in plant_name.get("medicinalDetails", []):
                        st.write(detail)
                    st.write("Common Growth Location: ", plant_name.get("commonGrowthLocation", "N/A"))
                    st.write("Disclaimer: ", plant_name.get("disclaimer", "N/A"))
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
