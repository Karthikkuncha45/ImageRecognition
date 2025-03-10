from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load a pre-trained model (e.g., MobileNet)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def preprocess_image(image):
    # Convert image to RGB if it has an alpha channel
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    return decoded_predictions

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    predictions = predict_image(image)
    results = [{"label": label, "probability": float(prob)} for (_, label, prob) in predictions]
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)