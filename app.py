import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload folder exists

# Load face detection model and age/gender model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('models/age_gender_model.keras')
print(" Model loaded successfully.")

# Detect and crop face from uploaded image
def detect_and_crop_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    return face_img

# Preprocess the cropped face for the model
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            face = detect_and_crop_face(image_path)
            if face is None:
                return render_template('index.html', error="No face detected. Please upload a clearer image.")

            processed_face = preprocess_face(face)
            age_pred, gender_pred = model.predict(processed_face)

            age = round(float(age_pred[0]), 1)
            gender = "Male" if np.argmax(gender_pred[0]) == 0 else "Female"

            return render_template('index.html', age=age, gender=gender, filename=image.filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
