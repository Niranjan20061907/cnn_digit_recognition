import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import PIL.ImageOps

# ---------------------------------
# Load CNN Model (Keras)
# ---------------------------------
model = load_model("mnist_cnn.keras")

# Canvas size
WIDTH, HEIGHT = 250, 250

# Tkinter window
root = tk.Tk()
root.title("Handwritten Digit Recognition (CNN)")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

# PIL image to draw on
image = Image.new("L", (WIDTH, HEIGHT), 255)
draw = ImageDraw.Draw(image)


# ---------------------------------
# Draw on canvas
# ---------------------------------
def paint(event):
    x1, y1 = event.x - 6, event.y - 6
    x2, y2 = event.x + 6, event.y + 6
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill=0)


def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=255)
    result_label.config(text="Draw a digit")


# ---------------------------------
# Predict using CNN
# ---------------------------------


def predict():
    # 1. Get the bounding box of the drawn digit (ignore whitespace)
    # The image is white background, black drawing.
    # Invert first so the drawing is the "object" (white on black) for getbbox
    inverted_image = PIL.ImageOps.invert(image)
    bbox = inverted_image.getbbox()

    if bbox:
        # 2. Crop the image to the contents (the digit)
        img_cropped = inverted_image.crop(bbox)

        # 3. Create a new 28x28 blank (black) image
        # MNIST requires the digit to be centered in a 28x28 box
        new_img = Image.new("L", (28, 28), 0)  # 0 = Black background

        # 4. Resize the cropped digit to fit in a 20x20 box (keeping aspect ratio)
        # This leaves 4px padding on all sides, matching MNIST dataset style
        img_cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

        # 5. Paste the resized digit into the center of the 28x28 canvas
        w, h = img_cropped.size
        x_offset = (28 - w) // 2
        y_offset = (28 - h) // 2
        new_img.paste(img_cropped, (x_offset, y_offset))

        # 6. Prepare for Model
        img_arr = np.array(new_img)
        img_arr = img_arr / 255.0  # Normalize
        img_arr = img_arr.reshape(1, 28, 28, 1)

        # 7. Predict
        prediction = model.predict(img_arr)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        result_label.config(text=f"Predicted Digit: {digit} ({confidence:.2f})")
    else:
        result_label.config(text="Canvas is empty!")


# ---------------------------------
# Bind mouse & buttons
# ---------------------------------
canvas.bind("<B1-Motion>", paint)

btn_predict = tk.Button(root, text="Predict", command=predict)
btn_predict.pack(pady=5)

btn_clear = tk.Button(root, text="Clear", command=clear)
btn_clear.pack(pady=5)

result_label = tk.Label(root, text="Draw a digit", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
