from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io
import json

# --- SETUP ---
app = Flask(__name__)

# --- Load the trained model and class labels ---
# This part runs only once when the server starts.
try:
    # Load the entire model from the .h5 file.
    model = tf.keras.models.load_model('crop_vision_model.h5')
    
    # Load the class names that the model was trained on.
    with open('class_labels.json', 'r') as f:
        class_labels = json.load(f)
    print("✅ Model and class labels loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or labels: {e}")
    print("Please ensure you have run 'train_image_model.py' successfully and the .h5 and .json files exist.")
    model = None
    class_labels = None

def preprocess_image(base64_string, target_size=(224, 224)):
    """
    Takes a base64 encoded image string, decodes it, 
    and prepares it for the model.
    """
    # Decode the base64 string into bytes.
    img_bytes = base64.b64decode(base64_string)
    
    # Open the image from the bytes.
    img = Image.open(io.BytesIO(img_bytes))
    
    # Resize the image to the size the model expects.
    if img.size != target_size:
        img = img.resize(target_size)
    
    # Convert the image to a NumPy array.
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Add a batch dimension, as the model expects a batch of images.
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values from 0-255 to 0-1, as the model was trained this way.
    img_array /= 255.0
    
    return img_array

# --- API ENDPOINT ---
@app.route('/diagnose-image', methods=['POST'])
def diagnose_image():
    # First, check if the model was loaded correctly.
    if not model or not class_labels:
        return jsonify({"error": "Model is not available. Please check server logs."}), 500

    # Get the JSON data from the request.
    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({"error": "Invalid input: 'image_base64' key is required."}), 400
    
    try:
        # 1. Preprocess the uploaded image from the base64 string.
        processed_image = preprocess_image(data['image_base64'])
        
        # 2. Use the model to make a prediction.
        predictions = model.predict(processed_image)
        
        # 3. Get the top prediction and its confidence score.
        confidence_score = float(np.max(predictions[0]))
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_disease = class_labels[predicted_class_index]
        
        # 4. Format the disease name for better readability.
        # (e.g., "Tomato___Early_blight" becomes "Tomato Early blight")
        formatted_disease_name = predicted_disease.replace('___', ' ').replace('_', ' ')
        
        # 5. Create a clean JSON response and send it back to the app.
        result = {
            "disease_name": formatted_disease_name,
            "confidence": f"{confidence_score:.2%}" # Format as a percentage string
        }
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Failed to process the image. It might be in an invalid format."}), 500


if __name__ == '__main__':
    # host='0.0.0.0' makes the API accessible from other devices on your network (like your phone).
    app.run(debug=True, host='0.0.0.0')
