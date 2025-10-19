# flood_gui.py

import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load the trained model
try:
    model = joblib.load("flood_model.pkl")
except:
    messagebox.showerror("Error", "Model not found! Please train the model first by running main.py.")
    exit()

# GUI window setup
root = tk.Tk()
root.title("🌊 Flood Risk Prediction System")
root.geometry("450x400")
root.config(bg="#e0f7fa")

title = tk.Label(root, text="Flood Risk Prediction", font=("Arial", 18, "bold"), bg="#e0f7fa", fg="#00796b")
title.pack(pady=10)

# --------------------------
# Input fields (change labels as per your dataset)
# --------------------------
labels = ["Rainfall (mm)", "Temperature (°C)", "Humidity (%)", "River Level (m)"]
entries = []

for lbl in labels:
    frame = tk.Frame(root, bg="#e0f7fa")
    frame.pack(pady=5)
    tk.Label(frame, text=lbl + ":", font=("Arial", 12), bg="#e0f7fa").pack(side=tk.LEFT, padx=5)
    ent = tk.Entry(frame, font=("Arial", 12))
    ent.pack(side=tk.RIGHT, padx=5)
    entries.append(ent)

# --------------------------
# Prediction Function
# --------------------------
def predict_flood():
    try:
        # Get user input and convert to floats
        data = [float(e.get()) for e in entries]
        data = np.array(data).reshape(1, -1)

        # Predict flood risk
        prediction = model.predict(data)[0]

        # Show result
        if prediction == 1:
            messagebox.showwarning("Result", "🚨 High Flood Risk Detected!")
        else:
            messagebox.showinfo("Result", "✅ Low Flood Risk (Safe Zone)")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# --------------------------
# Buttons
# --------------------------
btn_predict = tk.Button(root, text="Predict Flood Risk", command=predict_flood, bg="#00796b", fg="white", font=("Arial", 12, "bold"), padx=20, pady=5)
btn_predict.pack(pady=20)

btn_exit = tk.Button(root, text="Exit", command=root.destroy, bg="#d32f2f", fg="white", font=("Arial", 12, "bold"), padx=20, pady=5)
btn_exit.pack()

root.mainloop()
