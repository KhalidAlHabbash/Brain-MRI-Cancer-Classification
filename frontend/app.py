import webbrowser
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image
from backend.src.model import BrainTumorClassifier
import torch
import torchvision.transforms as transforms
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model
model = BrainTumorClassifier(4)
model.load_state_dict(torch.load('../backend/models/best_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('brain_tumor_classifier.html')  # Render the HTML file


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Prepare the image and make a prediction
        img = Image.open(file_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_class_index = predicted.item()  # Get  predicted index

        # Map indices to class names
        class_names = {
            0: 'Glioma',
            1: 'Meningioma',
            2: 'No Tumor',
            3: 'Pituitary Tumor'
        }

        # Get the predicted class name
        predicted_class = class_names.get(predicted_class_index, "Unknown Class")
        return jsonify({'prediction': predicted_class})

    return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:5000/')  # Open the browser automatically
    app.run(host='0.0.0.0', port=5000, debug=True)  # Run the Flask app
