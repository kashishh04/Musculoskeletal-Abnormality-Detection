from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import os

app = Flask(__name__)

# Secret key for Flask sessions and flash messages
app.secret_key = "your_secret_key"

# Path to save uploaded files temporarily
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model paths for different types of X-ray images
MODEL_PATHS = {
    "XR_WRIST": r"D:\Project Mura\WRIST.h5",
    "XR_FINGER": r"D:\Project Mura\FINGER.h5",
    "XR_HUMERUS": r"D:\Project Mura\HUMERUS.h5",
    "XR_FOREARM": r"D:\Project Mura\FOREARM.h5",
    "XR_HAND": r"D:\Project Mura\HAND.h5",
}

# ---- Image Preprocessing Function ----
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = imread(img_path)
        img_resized = resize(img, (300, 300, 3))  # Resize to (300, 300, 3) first
        h, w, _ = img_resized.shape

        startx = w // 2 - (target_size[0] // 2)
        starty = h // 2 - (target_size[1] // 2)
        img_cropped = img_resized[starty:starty + target_size[1], startx:startx + target_size[0]]

        img_normalized = img_cropped / 255.0
        return img_normalized
    except Exception as e:
        return None, str(e)

# ---- Class Labels ----
CLASS_NAMES = {
    "XR_HAND": ["Positive Abnormality", "Negative Abnormality"],
    "XR_WRIST": ["Positive Abnormality", "Negative Abnormality"],
    "XR_FINGER": ["Positive Abnormality", "Negative Abnormality"],
    "XR_HUMERUS": ["Positive Abnormality", "Negative Abnormality"],
    "XR_FOREARM": ["Positive Abnormality", "Negative Abnormality"],
}

# Models cache
models = {}

# Load model on demand (lazy loading)
def load_model_on_demand(model_name):
    if model_name not in models:
        try:
            print(f"Loading model: {model_name}")
            models[model_name] = load_model(MODEL_PATHS[model_name])
            print(f"Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise ValueError(f"Failed to load model {model_name}: {e}")
    return models[model_name]

# ---- Routes ----

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        file = request.files.get('xray_image')

        if not file or not model_name:
            flash('Please select a model and upload an X-ray image.', 'error')
            return redirect(url_for('predict'))

        # Validate the model_name
        if model_name not in MODEL_PATHS:
            flash('Invalid model selected.', 'error')
            return redirect(url_for('predict'))

        # Save the uploaded file temporarily
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(temp_file_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(temp_file_path)
        if preprocessed_image is None:
            flash('Error in preprocessing the image.', 'error')
            os.remove(temp_file_path)
            return redirect(url_for('predict'))

        # Load the selected model
        try:
            model = load_model_on_demand(model_name)
        except ValueError as e:
            flash(str(e), 'error')
            os.remove(temp_file_path)
            return redirect(url_for('predict'))

        # Prepare the image for prediction
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        predictions = model.predict(img_batch)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        # Clean up the uploaded file
        os.remove(temp_file_path)

        result = {
            "detected_issue": CLASS_NAMES[model_name][predicted_class],  # Matching the template variable
            "confidence_level": f"{confidence * 100:.2f}"  # Showing confidence as percentage
        }

        return render_template('predict.html', result=result)  # Pass the result dict here

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
