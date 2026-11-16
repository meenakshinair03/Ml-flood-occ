

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flood Risk Assessment · India", page_icon="🌊", layout="wide")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
MODEL_DIRS = [BASE / "models", BASE.parent / "models", Path("models")]

INK = "#0f2a43"
RISK = {
    "LOW": {"color": "#2e9e6b", "label": "Low risk", "range": "below 34%"},
    "MEDIUM": {"color": "#e0a21b", "label": "Medium risk", "range": "34% to 66%"},
    "HIGH": {"color": "#d9453b", "label": "High risk", "range": "67% and above"},
}

ADVICE = {
    "LOW": (
        "The area looks relatively safe from flooding right now.",
        [
            "Carry on with normal activities",
            "Keep an eye on weather forecasts",
            "Keep basic emergency supplies at home",
            "Stay subscribed to local alerts",
        ],
    ),
    "MEDIUM": (
        "There is a moderate chance of flooding. Stay cautious.",
        [
            "Watch weather and water-level updates",
            "Prepare emergency supplies",
            "Identify your evacuation routes",
            "Keep important documents somewhere safe",
            "Avoid low-lying areas where possible",
            "Follow local authority announcements",
        ],
    ),
    "HIGH": (
        "A significant flood threat is indicated. Act now.",
        [
            "Alert local authorities immediately",
            "Prepare for possible evacuation",
            "Move to higher ground",
            "Secure property and valuables",
            "Keep an emergency kit ready",
            "Monitor water levels closely",
            "Follow official evacuation orders",
            "Stay away from flood-prone areas",
        ],
    ),
}

# Defaults and one-click presets (same sample values as the desktop app)
DEFAULTS = {
    "lat": 20.52, "lon": 70.92, "rain": 179.72, "temp": 34.84, "hum": 63.63,
    "discharge": 3038.20, "level": 6.97, "elev": 5106.69, "pop": 1427.95,
    "land": "Agricultural", "soil": "Peat", "infra": False, "hist": False,
}
PRESETS = {
    "High risk": {
        "lat": 25.36, "lon": 85.61, "rain": 198.98, "temp": 32.63, "hum": 74.45,
        "discharge": 5589.20, "level": 13.18, "elev": 2512.27, "pop": 6163.06,
        "land": "Desert", "soil": "Sandy", "infra": True, "hist": True,
    },
    "Medium risk": DEFAULTS,
    "Low risk": {
        "lat": 25.43, "lon": 90.12, "rain": 212.87, "temp": 35.60, "hum": 60.76,
        "discharge": 1867.75, "level": 2.02, "elev": 7287.01, "pop": 6782.94,
        "land": "Forest", "soil": "Silt", "infra": False, "hist": False,
    },
}

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, label, input {{ font-family: 'IBM Plex Sans', sans-serif; }}
h1, h2, h3, .hero-title {{ font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}

.hero {{ background: {INK}; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; color: #eaf3fa; }}
.hero-title {{ font-size: 2.1rem; font-weight: 700; margin: 0; color: #ffffff; }}
.hero p {{ margin: 6px 0 0 0; color: #a9c4da; font-size: 1.02rem; max-width: 62ch; }}

.group-title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 1.05rem; margin: 0 0 2px 0; }}
.group-note {{ color: #7b8b99; font-size: 0.85rem; margin: 0 0 10px 0; }}

.result {{ background: {INK}; border-radius: 16px; padding: 22px 24px 20px 24px; color: #eaf3fa; border-top: 6px solid var(--risk); }}
.result .verdict {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 700; color: var(--risk); margin: 0; }}
.result .summary {{ color: #c6d9e8; margin: 4px 0 8px 0; }}
.result svg {{ width: 100%; height: auto; display: block; margin: 0 auto; }}
.result .split {{ display: flex; gap: 12px; margin: 4px 0 14px 0; }}
.result .split div {{ flex: 1; background: rgba(255,255,255,.07); border-radius: 10px; padding: 10px 12px; }}
.result .split b {{ display: block; font-size: 1.25rem; }}
.result .split span {{ font-size: 0.8rem; color: #9fb8cc; }}
.result h4 {{ margin: 0 0 6px 0; font-size: 0.95rem; color: #ffffff; }}
.result ul {{ margin: 0; padding-left: 1.1rem; color: #d6e5f1; }}
.result li {{ margin-bottom: 3px; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.85rem; color: #7b8b99; margin-top: 12px; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    for d in MODEL_DIRS:
        m, e = d / "flood_model.pkl", d / "label_encoders.pkl"
        if m.exists() and e.exists():
            return joblib.load(m), joblib.load(e)
    return None, None


model, encoders = load_artifacts()
if model is None:
    st.error(
        "Model files not found. Train the model first (run `main.py`) and make sure "
        "`models/flood_model.pkl` and `models/label_encoders.pkl` are in the repository."
    )
    st.stop()


def options_for(encoder_name, fallback):
    enc = encoders.get(encoder_name) if hasattr(encoders, "get") else None
    return [str(c) for c in enc.classes_] if enc is not None else fallback


LAND_OPTIONS = options_for("Land Cover", ["Agricultural", "Forest", "Urban", "Water Bodies", "Desert"])
SOIL_OPTIONS = options_for("Soil Type", ["Clay", "Loam", "Sandy", "Silt", "Peat"])


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def get_risk_level(p):
    if p < 0.34:
        return "LOW"
    if p < 0.67:
        return "MEDIUM"
    return "HIGH"


def gauge_svg(p, level):
    cx, cy, r = 150, 150, 120

    def pt(frac, radius=r):
        a = math.pi * (1 - frac)
        return cx + radius * math.cos(a), cy - radius * math.sin(a)

    def arc(f0, f1, color, opacity):
        x0, y0 = pt(f0)
        x1, y1 = pt(f1)
        return (f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}" '
                f'stroke="{color}" stroke-width="20" fill="none" opacity="{opacity}"/>')

    bands = [(0.0, 0.34, "LOW"), (0.34, 0.67, "MEDIUM"), (0.67, 1.0, "HIGH")]
    arcs = "".join(arc(a, b, RISK[n]["color"], 1 if n == level else 0.28) for a, b, n in bands)
    nx, ny = pt(p, r - 34)
    return (
        '<svg viewBox="0 0 300 200" role="img" aria-label="Flood probability gauge">'
        f"{arcs}"
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#ffffff" stroke-width="4" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="8" fill="#ffffff"/>'
        f'<text x="{cx}" y="190" text-anchor="middle" font-size="34" font-weight="700" fill="#ffffff">{p*100:.1f}%</text>'
        '<text x="30" y="172" text-anchor="middle" font-size="11" fill="#9fb8cc">0%</text>'
        '<text x="270" y="172" text-anchor="middle" font-size="11" fill="#9fb8cc">100%</text>'
        "</svg>"
    )


def apply_preset(name):
    for k, v in PRESETS[name].items():
        if k == "land" and v not in LAND_OPTIONS:
            v = LAND_OPTIONS[0]
        if k == "soil" and v not in SOIL_OPTIONS:
            v = SOIL_OPTIONS[0]
        st.session_state[k] = v


for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
if st.session_state["land"] not in LAND_OPTIONS:
    st.session_state["land"] = LAND_OPTIONS[0]
if st.session_state["soil"] not in SOIL_OPTIONS:
    st.session_state["soil"] = SOIL_OPTIONS[0]


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <p class="hero-title">Flood risk assessment</p>
  <p>Enter rainfall, river, terrain and community readings for a location in India
  to estimate how likely a flood is. The result updates as you change the inputs.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Layout: inputs (left) / result (right)
# ----------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.caption("Start from an example, then adjust the values.")
    b1, b2, b3, b4 = st.columns(4)
    b1.button("High risk example", on_click=apply_preset, args=("High risk",), use_container_width=True)
    b2.button("Medium risk example", on_click=apply_preset, args=("Medium risk",), use_container_width=True)
    b3.button("Low risk example", on_click=apply_preset, args=("Low risk",), use_container_width=True)
    b4.button("Reset", on_click=apply_preset, args=("Medium risk",), use_container_width=True)

    with st.container(border=True):
        st.markdown('<p class="group-title">Location</p><p class="group-note">Decimal degrees</p>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", -90.0, 90.0, step=0.01, format="%.2f", key="lat")
        lon = c2.number_input("Longitude", -180.0, 180.0, step=0.01, format="%.2f", key="lon")
        if not (6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0):
            st.warning("These coordinates look outside India. The model was built for Indian conditions, "
                       "so the estimate may be unreliable.")

    with st.container(border=True):
        st.markdown('<p class="group-title">Weather</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        rain = c1.number_input("Rainfall (mm)", 0.0, 500.0, step=1.0, key="rain")
        temp = c2.number_input("Temperature (°C)", -10.0, 50.0, step=0.1, key="temp")
        hum = c3.number_input("Humidity (%)", 0.0, 100.0, step=0.5, key="hum")

    with st.container(border=True):
        st.markdown('<p class="group-title">Water</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        discharge = c1.number_input("River discharge (m³/s)", 0.0, 10000.0, step=10.0, key="discharge")
        level = c2.number_input("Water level (m)", 0.0, 50.0, step=0.1, key="level")

    with st.container(border=True):
        st.markdown('<p class="group-title">Terrain and land</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        elev = c1.number_input("Elevation (m)", 0.0, 9000.0, step=10.0, key="elev")
        land = c2.selectbox("Land cover", LAND_OPTIONS, key="land")
        soil = c3.selectbox("Soil type", SOIL_OPTIONS, key="soil")

    with st.container(border=True):
        st.markdown('<p class="group-title">Community</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        pop = c1.number_input("Population density", 0.0, 10000.0, step=10.0, key="pop")
        infra = c2.toggle("Infrastructure present", key="infra")
        hist = c3.toggle("Past floods recorded", key="hist")

# ----------------------------------------------------------------------
# Prediction (feature order matches the training pipeline / original app)
# ----------------------------------------------------------------------
try:
    land_enc = encoders["Land Cover"].transform([land])[0]
    soil_enc = encoders["Soil Type"].transform([soil])[0]
    features = np.array(
        [[lat, lon, rain, temp, hum, discharge, level, elev,
          land_enc, soil_enc, pop, int(infra), int(hist)]],
        dtype=float,
    )
    proba = model.predict_proba(features)[0]
    flood_p, safe_p = float(proba[1]), float(proba[0])
    error = None
except Exception as e:  # noqa: BLE001
    error = str(e)

with right:
    if error:
        st.error(f"Could not make a prediction: {error}")
    else:
        level_name = get_risk_level(flood_p)
        info = RISK[level_name]
        summary, tips = ADVICE[level_name]
        tips_html = "".join(f"<li>{t}</li>" for t in tips)
        st.markdown(
            f'<div class="result" style="--risk:{info["color"]}">'
            f'<p class="verdict">{info["label"]}</p>'
            f'<p class="summary">{summary}</p>'
            f"{gauge_svg(flood_p, level_name)}"
            f'<div class="split">'
            f"<div><b>{flood_p*100:.1f}%</b><span>Flood probability</span></div>"
            f"<div><b>{safe_p*100:.1f}%</b><span>No-flood probability</span></div>"
            f"</div>"
            f"<h4>What to do</h4><ul>{tips_html}</ul>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="legend">'
            f'<span><i style="background:{RISK["LOW"]["color"]}"></i>Low: {RISK["LOW"]["range"]}</span>'
            f'<span><i style="background:{RISK["MEDIUM"]["color"]}"></i>Medium: {RISK["MEDIUM"]["range"]}</span>'
            f'<span><i style="background:{RISK["HIGH"]["color"]}"></i>High: {RISK["HIGH"]["range"]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Values used for this estimate"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Input": ["Latitude", "Longitude", "Rainfall (mm)", "Temperature (°C)", "Humidity (%)",
                                  "River discharge (m³/s)", "Water level (m)", "Elevation (m)", "Land cover",
                                  "Soil type", "Population density", "Infrastructure", "Historical floods"],
                        "Value": [lat, lon, rain, temp, hum, discharge, level, elev, land, soil, pop,
                                  "Yes" if infra else "No", "Yes" if hist else "No"],
                    }
                ).astype(str),
                hide_index=True,
                use_container_width=True,
            )
        st.caption("This is a model estimate, not an official warning. For real events, follow "
                   "alerts from IMD, CWC and your state disaster management authority.")

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
