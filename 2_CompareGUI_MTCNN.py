import os
import numpy as np
import tkinter as tk

from mtcnn import MTCNN
from keras.models import Model
from keras.layers import Input, Dense
from tkinter import filedialog
from keras.models import load_model
from tensorflow.keras import backend as K
from PIL import ImageTk, Image, ImageDraw
from tensorflow.keras.utils import custom_object_scope

# Define the window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800


# Define the custom loss function
def contrastive_loss(y_true, y_pred):
    margin = 1
    return (1 - y_true) * 0.5 * K.square(y_pred) + y_true * 0.5 * K.square(K.maximum(margin - y_pred, 0))


def extract_face(image_path):
    # Load the image using PIL
    image = Image.open(image_path)

    # Resize the image to fit the window size
    image.thumbnail((WINDOW_WIDTH // 2, WINDOW_HEIGHT))

    # Convert the image to RGB
    image = image.convert('RGB')

    # Convert the image to numpy array
    pixels = np.array(image)

    # Initialize the MTCNN detector
    detector = MTCNN()

    # Detect faces in the image
    results = detector.detect_faces(pixels)

    # Extract the first face (assuming there is only one face in the image)
    if len(results) > 0:
        x, y, width, height = results[0]['box']
        x1, y1 = x, y
        x2, y2 = x + width, y + height

        # Extract the face from the image
        face = pixels[y1:y2, x1:x2]

        # Draw bounding box on the image
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)

        # Resize the face to match the model's input shape
        face = Image.fromarray(face)
        face = face.resize((32, 32))

        # Convert the face to numpy array
        face = np.array(face)

        # Normalize the face pixels
        face = face / 255.0

        return face, image
    else:
        return None, image


def browse_image(entry, image_label):
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image",
                                          filetypes=(("JPEG Files", "*.jpg"), ("PNG Files", "*.png")))
    entry.delete(0, tk.END)
    entry.insert(0, filename)
    show_image(filename, image_label)


def show_image(image_path, image_label):
    # Load the image using PIL
    image = Image.open(image_path)

    # Resize the image to fit in the label
    image.thumbnail((WINDOW_WIDTH // 2, WINDOW_HEIGHT))

    # Convert the image to PhotoImage
    photo = ImageTk.PhotoImage(image)

    # Update the image label
    image_label.configure(image=photo)
    image_label.image = photo


def predict():
    test_image1_path = entry_image1.get()
    test_image2_path = entry_image2.get()

    # Extract faces from the selected images
    test_image1, image1 = extract_face(test_image1_path)
    test_image2, image2 = extract_face(test_image2_path)

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

    # Show the images with bounding boxes
    show_image1 = ImageTk.PhotoImage(image1)
    show_image2 = ImageTk.PhotoImage(image2)
    image_label1.configure(image=show_image1)
    image_label1.image = show_image1
    image_label2.configure(image=show_image2)
    image_label2.image = show_image2


# Create the main window
window = tk.Tk()
window.title("Face Comparison Prediction")
window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

# Image 1 Entry, Browse Button, and Label
entry_image1 = tk.Entry(window, width=50)
entry_image1.pack()

button_browse1 = tk.Button(window, text="Browse Image 1", command=lambda: browse_image(entry_image1, image_label1))
button_browse1.pack()

image_frame1 = tk.Frame(window)
image_frame1.pack(side=tk.LEFT)
image_label1 = tk.Label(image_frame1)
image_label1.pack()

# Image 2 Entry, Browse Button, and Label
entry_image2 = tk.Entry(window, width=50)
entry_image2.pack()

button_browse2 = tk.Button(window, text="Browse Image 2", command=lambda: browse_image(entry_image2, image_label2))
button_browse2.pack()

image_frame2 = tk.Frame(window)
image_frame2.pack(side=tk.LEFT)
image_label2 = tk.Label(image_frame2)
image_label2.pack()

# Prediction Button
button_predict = tk.Button(window, text="Predict", command=predict)
button_predict.pack()

# Prediction Result Label
prediction_result = tk.StringVar()
label_result = tk.Label(window, textvariable=prediction_result)
label_result.pack()

# Start the main loop
window.mainloop()
