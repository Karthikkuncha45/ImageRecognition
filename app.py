from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load a pre-trained model (e.g., MobileNet)
try:
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")

def preprocess_image(image):
    try:
        # Convert image to RGB if it has an alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = image.resize((224, 224))  # Resize to match model input size
        image = np.array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

def predict_image(image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        logging.error(f"Error predicting image: {e}")
        raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))
        predictions = predict_image(image)
        results = [{"label": label, "probability": float(prob)} for (_, label, prob) in predictions]
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in /predict route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
