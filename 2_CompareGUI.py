import os
import cv2

import numpy as np
import tkinter as tk

from tkinter import filedialog
from PIL import ImageTk, Image
from keras.utils import load_img, img_to_array
from keras.models import load_model


def extract_face(image_path):
    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))

    # Extract the first face (assuming there is only one face in the image)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face = face / 255.0
        return face
    else:
        return None


def browse_image(entry):
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image",
                                          filetypes=(("JPEG Files", "*.jpg"), ("PNG Files", "*.png")))
    entry.delete(0, tk.END)
    entry.insert(0, filename)


def predict():
    test_image1_path = entry_image1.get()
    test_image2_path = entry_image2.get()

    # Extract faces from the selected images
    test_image1 = extract_face(test_image1_path)
    test_image2 = extract_face(test_image2_path)

    if test_image1 is None or test_image2 is None:
        prediction_result.set("No faces found in the selected images.")
    else:
        # Expand the dimensions to match the model's input shape
        test_image1 = np.expand_dims(test_image1, axis=0)
        test_image2 = np.expand_dims(test_image2, axis=0)

        model = load_model("weights.best.hdf5")

        prediction = model.predict([test_image1, test_image2])
        predicted_label = 1 if prediction >= 0.5 else 0
        if predicted_label == 1:
            prediction = "Same face."
        prediction_result.set(f"Prediction: {prediction}")
        print(prediction)
        print(predicted_label)


# Create the main window
window = tk.Tk()
window.title("Face Comparison Prediction")
window.geometry("500x400")

# Image 1 Entry and Browse Button
entry_image1 = tk.Entry(window, width=50)
entry_image1.pack()

button_browse1 = tk.Button(window, text="Browse Image 1", command=lambda: browse_image(entry_image1))
button_browse1.pack()

# Image 2 Entry and Browse Button
entry_image2 = tk.Entry(window, width=50)
entry_image2.pack()

button_browse2 = tk.Button(window, text="Browse Image 2", command=lambda: browse_image(entry_image2))
button_browse2.pack()

# Prediction Button
button_predict = tk.Button(window, text="Predict", command=predict)
button_predict.pack()

# Prediction Result Label
prediction_result = tk.StringVar()
label_result = tk.Label(window, textvariable=prediction_result)
label_result.pack()

# Start the main loop
window.mainloop()
