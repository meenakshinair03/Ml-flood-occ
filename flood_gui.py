# flood_gui_with_risk_levels.py
# Enhanced GUI with High/Medium/Low risk levels

import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import joblib
import os

# Load the trained model and encoders
try:
    model = joblib.load("models/flood_model.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    print("✅ Model and encoders loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    messagebox.showerror("Error", "Model not found! Please train the model first by running main.py.")
    exit()

# GUI window setup
root = tk.Tk()
root.title("🌊 Flood Risk Prediction System - India")
root.geometry("600x800")
root.config(bg="#e0f7fa")

# Title
title = tk.Label(root, text="Flood Risk Prediction System", 
                font=("Arial", 20, "bold"), bg="#e0f7fa", fg="#00796b")
title.pack(pady=15)

subtitle = tk.Label(root, text="Enter environmental parameters to assess flood risk", 
                   font=("Arial", 10), bg="#e0f7fa", fg="#004d40")
subtitle.pack()

# Create a frame for inputs
input_frame = tk.Frame(root, bg="#e0f7fa")
input_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

# Input fields based on actual dataset
entries = {}

# Continuous features
continuous_features = [
    ("Latitude", -90, 90),
    ("Longitude", -180, 180),
    ("Rainfall (mm)", 0, 500),
    ("Temperature (°C)", -10, 50),
    ("Humidity (%)", 0, 100),
    ("River Discharge (m³/s)", 0, 10000),
    ("Water Level (m)", 0, 50),
    ("Elevation (m)", 0, 9000),
    ("Population Density", 0, 10000),
]

row = 0
for feature, min_val, max_val in continuous_features:
    tk.Label(input_frame, text=f"{feature}:", font=("Arial", 11), 
            bg="#e0f7fa", anchor='w').grid(row=row, column=0, sticky='w', pady=5, padx=5)
    
    entry = tk.Entry(input_frame, font=("Arial", 11), width=20)
    entry.grid(row=row, column=1, pady=5, padx=5)
    entries[feature] = entry
    
    # Add range hint
    tk.Label(input_frame, text=f"({min_val}-{max_val})", font=("Arial", 9), 
            bg="#e0f7fa", fg="gray").grid(row=row, column=2, sticky='w', padx=5)
    row += 1

# Binary features (0 or 1)
binary_features = [
    "Infrastructure",
    "Historical Floods"
]

for feature in binary_features:
    tk.Label(input_frame, text=f"{feature}:", font=("Arial", 11), 
            bg="#e0f7fa", anchor='w').grid(row=row, column=0, sticky='w', pady=5, padx=5)
    
    combo = ttk.Combobox(input_frame, values=["0 (No)", "1 (Yes)"], 
                         font=("Arial", 11), width=18, state='readonly')
    combo.set("0 (No)")
    combo.grid(row=row, column=1, pady=5, padx=5)
    entries[feature] = combo
    row += 1

# Categorical features
tk.Label(input_frame, text="Land Cover:", font=("Arial", 11), 
        bg="#e0f7fa", anchor='w').grid(row=row, column=0, sticky='w', pady=5, padx=5)

land_cover_options = ['Agricultural', 'Forest', 'Urban', 'Water Bodies', 'Desert']
land_cover_combo = ttk.Combobox(input_frame, values=land_cover_options, 
                                font=("Arial", 11), width=18, state='readonly')
land_cover_combo.set(land_cover_options[0])
land_cover_combo.grid(row=row, column=1, pady=5, padx=5)
entries['Land Cover'] = land_cover_combo
row += 1

tk.Label(input_frame, text="Soil Type:", font=("Arial", 11), 
        bg="#e0f7fa", anchor='w').grid(row=row, column=0, sticky='w', pady=5, padx=5)

soil_type_options = ['Clay', 'Loam', 'Sandy', 'Silt', 'Peat']
soil_type_combo = ttk.Combobox(input_frame, values=soil_type_options, 
                               font=("Arial", 11), width=18, state='readonly')
soil_type_combo.set(soil_type_options[0])
soil_type_combo.grid(row=row, column=1, pady=5, padx=5)
entries['Soil Type'] = soil_type_combo

# Function to determine risk level based on probability
def get_risk_level(flood_probability):
    """
    Classify risk into three levels based on flood probability
    
    Risk Levels:
    - LOW:    0% - 33%
    - MEDIUM: 34% - 66%
    - HIGH:   67% - 100%
    """
    if flood_probability < 0.34:
        return "LOW", "🟢", "#4caf50"
    elif flood_probability < 0.67:
        return "MEDIUM", "🟡", "#ff9800"
    else:
        return "HIGH", "🔴", "#f44336"

# Prediction Function with Risk Levels
def predict_flood():
    try:
        # Collect all input values
        input_data = []
        
        # Get continuous features
        for feature in ['Latitude', 'Longitude', 'Rainfall (mm)', 'Temperature (°C)', 
                       'Humidity (%)', 'River Discharge (m³/s)', 'Water Level (m)', 
                       'Elevation (m)']:
            value = float(entries[feature].get())
            input_data.append(value)
        
        # Encode categorical features
        land_cover = entries['Land Cover'].get()
        land_cover_encoded = encoders['Land Cover'].transform([land_cover])[0]
        input_data.append(land_cover_encoded)
        
        soil_type = entries['Soil Type'].get()
        soil_type_encoded = encoders['Soil Type'].transform([soil_type])[0]
        input_data.append(soil_type_encoded)
        
        # Get remaining features
        input_data.append(float(entries['Population Density'].get()))
        
        # Binary features
        for feature in binary_features:
            value = int(entries[feature].get().split()[0])
            input_data.append(value)
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]
        
        flood_prob = probability[1]  # Probability of flood
        safe_prob = probability[0]   # Probability of no flood
        
        # Get risk level
        risk_level, emoji, color = get_risk_level(flood_prob)
        
        # Create detailed message
        message = f"""
╔═══════════════════════════════════════╗
║        FLOOD RISK ASSESSMENT          ║
╚═══════════════════════════════════════╝

{emoji} RISK LEVEL: {risk_level} {emoji}

Flood Probability: {flood_prob*100:.1f}%
Safe Probability:  {safe_prob*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        if risk_level == "LOW":
            message += """✅ LOW FLOOD RISK

The area appears relatively safe from flooding.

Recommendations:
• Continue normal activities
• Monitor weather forecasts
• Be prepared with basic emergency supplies
• Stay informed of local alerts
"""
            msg_type = "info"
            
        elif risk_level == "MEDIUM":
            message += """⚠️ MEDIUM FLOOD RISK

Moderate chance of flooding detected.
Caution advised.

Recommendations:
• Stay alert to weather conditions
• Prepare emergency supplies
• Identify evacuation routes
• Keep important documents safe
• Avoid low-lying areas if possible
• Monitor local authorities
"""
            msg_type = "warning"
            
        else:  # HIGH
            message += """🚨 HIGH FLOOD RISK

Significant flood threat detected!
Immediate action recommended.

URGENT RECOMMENDATIONS:
• Alert local authorities immediately
• Prepare for potential evacuation
• Move to higher ground
• Secure property and valuables
• Keep emergency kit ready
• Monitor water levels closely
• Follow official evacuation orders
• Stay away from flood-prone areas
"""
            msg_type = "error"
        
        # Display result with appropriate styling
        result_window = tk.Toplevel(root)
        result_window.title("Flood Risk Assessment")
        result_window.geometry("450x550")
        result_window.config(bg=color)
        
        # Title
        title_label = tk.Label(result_window, 
                              text=f"{emoji} {risk_level} RISK {emoji}",
                              font=("Arial", 24, "bold"),
                              bg=color, fg="white")
        title_label.pack(pady=20)
        
        # Message box
        text_frame = tk.Frame(result_window, bg="white")
        text_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, font=("Courier", 10), 
                             wrap=tk.WORD, padx=10, pady=10)
        text_widget.insert(1.0, message)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Close button
        close_btn = tk.Button(result_window, text="Close", 
                             command=result_window.destroy,
                             bg="white", fg=color,
                             font=("Arial", 12, "bold"),
                             padx=30, pady=10)
        close_btn.pack(pady=20)
    
    except ValueError as e:
        messagebox.showerror("Input Error", f"Please enter valid numeric values.\n\nError: {str(e)}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred:\n{str(e)}")

# Fill sample data functions
def fill_sample_high_risk():
    """Fill form with sample high-risk data"""
    sample_data = {
        'Latitude': '25.36',
        'Longitude': '85.61',
        'Rainfall (mm)': '198.98',
        'Temperature (°C)': '32.63',
        'Humidity (%)': '74.45',
        'River Discharge (m³/s)': '5589.20',
        'Water Level (m)': '13.18',
        'Elevation (m)': '2512.27',
        'Land Cover': 'Desert',
        'Soil Type': 'Sandy',
        'Population Density': '6163.06',
        'Infrastructure': '1 (Yes)',
        'Historical Floods': '1 (Yes)'
    }
    for key, value in sample_data.items():
        if isinstance(entries[key], ttk.Combobox):
            entries[key].set(value)
        else:
            entries[key].delete(0, tk.END)
            entries[key].insert(0, value)

def fill_sample_medium_risk():
    """Fill form with sample medium-risk data"""
    sample_data = {
        'Latitude': '20.52',
        'Longitude': '70.92',
        'Rainfall (mm)': '179.72',
        'Temperature (°C)': '34.84',
        'Humidity (%)': '63.63',
        'River Discharge (m³/s)': '3038.20',
        'Water Level (m)': '6.97',
        'Elevation (m)': '5106.69',
        'Land Cover': 'Agricultural',
        'Soil Type': 'Peat',
        'Population Density': '1427.95',
        'Infrastructure': '0 (No)',
        'Historical Floods': '0 (No)'
    }
    for key, value in sample_data.items():
        if isinstance(entries[key], ttk.Combobox):
            entries[key].set(value)
        else:
            entries[key].delete(0, tk.END)
            entries[key].insert(0, value)

def fill_sample_low_risk():
    """Fill form with sample low-risk data"""
    sample_data = {
        'Latitude': '25.43',
        'Longitude': '90.12',
        'Rainfall (mm)': '212.87',
        'Temperature (°C)': '35.60',
        'Humidity (%)': '60.76',
        'River Discharge (m³/s)': '1867.75',
        'Water Level (m)': '2.02',
        'Elevation (m)': '7287.01',
        'Land Cover': 'Forest',
        'Soil Type': 'Silt',
        'Population Density': '6782.94',
        'Infrastructure': '0 (No)',
        'Historical Floods': '0 (No)'
    }
    for key, value in sample_data.items():
        if isinstance(entries[key], ttk.Combobox):
            entries[key].set(value)
        else:
            entries[key].delete(0, tk.END)
            entries[key].insert(0, value)

def clear_all():
    """Clear all input fields"""
    for key, widget in entries.items():
        if isinstance(widget, ttk.Combobox):
            widget.set(widget['values'][0] if widget['values'] else '')
        else:
            widget.delete(0, tk.END)

# Buttons
button_frame = tk.Frame(root, bg="#e0f7fa")
button_frame.pack(pady=20)

btn_predict = tk.Button(button_frame, text="🔍 Predict Flood Risk", command=predict_flood, 
                       bg="#00796b", fg="white", font=("Arial", 12, "bold"), 
                       padx=20, pady=10, cursor="hand2")
btn_predict.grid(row=0, column=0, columnspan=3, pady=10, padx=5)

btn_sample_high = tk.Button(button_frame, text="🔴 High Risk Sample", 
                           command=fill_sample_high_risk, bg="#f44336", fg="white", 
                           font=("Arial", 10), padx=15, pady=8, cursor="hand2")
btn_sample_high.grid(row=1, column=0, padx=5)

btn_sample_medium = tk.Button(button_frame, text="🟡 Medium Risk Sample", 
                             command=fill_sample_medium_risk, bg="#ff9800", fg="white", 
                             font=("Arial", 10), padx=15, pady=8, cursor="hand2")
btn_sample_medium.grid(row=1, column=1, padx=5)

btn_sample_low = tk.Button(button_frame, text="🟢 Low Risk Sample", 
                          command=fill_sample_low_risk, bg="#4caf50", fg="white", 
                          font=("Arial", 10), padx=15, pady=8, cursor="hand2")
btn_sample_low.grid(row=1, column=2, padx=5)

btn_clear = tk.Button(button_frame, text="🗑️ Clear All", command=clear_all,
                     bg="#607d8b", fg="white", font=("Arial", 10),
                     padx=15, pady=8, cursor="hand2")
btn_clear.grid(row=2, column=0, columnspan=3, pady=5)

btn_exit = tk.Button(button_frame, text="❌ Exit", command=root.destroy, 
                    bg="#d32f2f", fg="white", font=("Arial", 12, "bold"), 
                    padx=20, pady=10, cursor="hand2")
btn_exit.grid(row=3, column=0, columnspan=3, pady=10)

# Footer
footer = tk.Label(root, text="Flood Risk Prediction System | Three-Level Risk Assessment", 
                 font=("Arial", 9), bg="#e0f7fa", fg="gray")
footer.pack(side=tk.BOTTOM, pady=10)

# Risk level legend
legend_frame = tk.Frame(root, bg="#e0f7fa", relief=tk.RIDGE, borderwidth=2)
legend_frame.pack(side=tk.BOTTOM, pady=5)

tk.Label(legend_frame, text="Risk Levels: ", font=("Arial", 9, "bold"), 
        bg="#e0f7fa").pack(side=tk.LEFT, padx=5)
tk.Label(legend_frame, text="🟢 LOW (0-33%)", font=("Arial", 9), 
        bg="#e0f7fa", fg="#4caf50").pack(side=tk.LEFT, padx=5)
tk.Label(legend_frame, text="🟡 MEDIUM (34-66%)", font=("Arial", 9), 
        bg="#e0f7fa", fg="#ff9800").pack(side=tk.LEFT, padx=5)
tk.Label(legend_frame, text="🔴 HIGH (67-100%)", font=("Arial", 9), 
        bg="#e0f7fa", fg="#f44336").pack(side=tk.LEFT, padx=5)

root.mainloop()