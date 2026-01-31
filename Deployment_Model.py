import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

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
    img = image.resize((28, 28))
    img_arr = np.array(img)

    img_arr = 255 - img_arr
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)

    prediction = model.predict(img_arr)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")


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
