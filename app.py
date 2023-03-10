from CapsNet import CapsNet

import tkinter as tk

from tkinter import *
from PIL import ImageTk, ImageDraw, Image

import torch
from torchvision import transforms


# Prepare model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
infer_model = CapsNet(n_routing_iterations=3, device=device)
infer_model.load_state_dict(torch.load('./model/best_v3.pt', map_location=torch.device('cpu')))
infer_model.to(device)

transform = transforms.Compose([
    transforms.Resize(size=(28, 28)),
    transforms.ToTensor(),
])

# =======================================================================
# TKINTER GUI
# =======================================================================

master = tk.Tk()
master.title("Capsule Network for Handwritten Digits Classification")
master.geometry("800x300")

frame_1 = tk.Frame(master)
frame_1.place(x=60, y=50)

# Create a label widget
string_var = tk.StringVar()
string_var.set("Write a digit {0...9}")
label = tk.Label(frame_1, textvariable=string_var, font=14)
label.grid(row=0, column=0, padx=10)

# Create a canvas
canvas_width = 140
canvas_height = 140
canvas = tk.Canvas(frame_1, width=canvas_width, height=canvas_height, background='black')
canvas.grid(row=1, column=0, padx=10)

# Create a PIL image and draw object
pil_image = Image.new("L", (canvas_width, canvas_height), color="black")
draw = ImageDraw.Draw(pil_image)

# Define a function to draw on the canvas
def draw_on_canvas(event):
    x, y = event.x, event.y
    r = 6
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], outline="white", fill="white")

# Bind the function to the canvas
canvas.bind("<B1-Motion>", draw_on_canvas)

def predict():
    image_tensor = transform(pil_image)
    image_tensor = image_tensor.to(device)
    infer_caps_2_outputs, infer_y_pred, infer_reconstructions = infer_model(image_tensor)

    # Show predicted value
    y_pred = str(infer_y_pred.item())
    string_var = tk.StringVar()
    string_var.set(f"Predicted Class: {y_pred}")
    label = tk.Label(frame_1, textvariable=string_var, font=14)
    label.grid(row=1, column=3, padx=10)

    string_var = tk.StringVar()
    string_var.set(f"Reconstructed image:")
    label = tk.Label(frame_1, textvariable=string_var, font=14)
    label.grid(row=0, column=4, padx=10)

    # Show reconstructed image
    infer_image = torch.reshape(infer_reconstructions, [-1, 28, 28])[0]
    infer_image_numpy = infer_image.cpu().detach().numpy()
    infer_image_numpy *= 255
    infer_image = Image.fromarray(infer_image_numpy)
    infer_image = infer_image.convert('RGB')
    infer_image = infer_image.resize((140, 140))
    
    tk_image = ImageTk.PhotoImage(infer_image)

    # Create a label widget and set its image
    label = tk.Label(frame_1, image=tk_image)
    label.image = tk_image
    label.grid(row=1, column=4, padx=10)

predict_button = tk.Button(frame_1, text="Predict", command=predict)
predict_button.grid(row=1, column=1, padx=10)

master.mainloop()