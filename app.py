import os
from flask import Flask, render_template, request, jsonify
import cv2
from keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = load_model('intel_scene_classifier.h5')

# Define class labels (update these based on your model's classes)
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)
    return class_labels[class_idx], confidence, pred[0].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            predicted_class, confidence, all_predictions = predict_image(filepath)
            
            # Create prediction details for all classes
            predictions_detail = [
                {'class': class_labels[i], 'confidence': float(all_predictions[i] * 100)}
                for i in range(len(class_labels))
            ]
            predictions_detail.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': predictions_detail
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
