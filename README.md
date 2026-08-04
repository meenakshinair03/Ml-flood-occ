# 🌊 Flood Risk Prediction System (India)

A machine learning pipeline that predicts flood risk across locations in India using environmental, geographic, and infrastructure data. The system cleans and explores the data, trains multiple classification models, automatically selects the best performer, and exposes predictions through both a command-line tool and a desktop GUI.

## Features

- **Data preprocessing** — loads and cleans the raw dataset, handles missing values, and label-encodes categorical features (`Land Cover`, `Soil Type`)
- **Exploratory Data Analysis (EDA)** — summary statistics, correlation heatmap, feature-vs-target plots, and an interactive **Folium map** of flood occurrences (`flood_occurrence_map.html`)
- **Model training** — trains and compares four models (Random Forest, Gradient Boosting, Logistic Regression, SVM), selects the best by F1-score, and saves evaluation plots (confusion matrix, ROC curves, feature importance, model comparison)
- **Prediction interface**
  - `predict.py` — command-line predictions via arguments
  - `flood_gui.py` — a Tkinter desktop GUI with color-coded Low/Medium/High risk results and one-click sample data
- **Diagnostics** — `debug_check.py` verifies your environment, files, packages, and dataset structure before you run the pipeline

## Project Structure

```
.
├── flood_risk_dataset_india.csv     # Source dataset
├── data_preprossesing.py            # Data loading, cleaning, encoding
├── eda.py                           # Exploratory data analysis + map generation
├── training.py                      # Model training, evaluation, saving artifacts
├── main.py                          # Full pipeline: preprocessing → EDA → training
├── main_simple.py                   # Simplified/verbose pipeline for debugging
├── predict.py                       # CLI prediction tool
├── flood_gui.py                     # Tkinter GUI for predictions
├── extract_sample.py                # Pulls representative low/medium/high risk samples
├── debug_check.py                   # Environment & dataset diagnostic script
├── requriements.txt                 # Python dependencies
│
├── models/                          # Generated after training
│   ├── flood_model.pkl              # Best trained model
│   ├── label_encoders.pkl           # Encoders for categorical features
│   ├── model_metadata.pkl           # Feature order, metrics, scaling flag
│   └── scaler.pkl                   # StandardScaler (only if best model needs scaling)
│
├── data/                            # Generated after preprocessing
│   └── cleaned_flood_data.csv
│
├── plots/                           # Generated after training
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   └── model_comparison.png
│
└── flood_occurrence_map.html        # Generated after EDA
```

## Dataset

`flood_risk_dataset_india.csv` contains the following columns:

| Column | Description |
|---|---|
| `Latitude`, `Longitude` | Location coordinates |
| `Rainfall (mm)` | Rainfall amount |
| `Temperature (°C)` | Temperature |
| `Humidity (%)` | Humidity level |
| `River Discharge (m³/s)` | River discharge rate |
| `Water Level (m)` | Water level |
| `Elevation (m)` | Elevation above sea level |
| `Land Cover` | Category: Agricultural, Forest, Urban, Water Body, Desert |
| `Soil Type` | Category: Clay, Loam, Sandy, Silt, Peat |
| `Population Density` | Population density |
| `Infrastructure` | Binary: 0 (No) / 1 (Yes) |
| `Historical Floods` | Binary: 0 (No) / 1 (Yes) |
| `Flood Occurred` | Target: 0 (No) / 1 (Yes) |

## Installation

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requriements.txt
```

> **Note:** `requriements.txt` currently lists `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`. The scripts also require `folium`, `joblib`, and (for the GUI) `tkinter`. It's recommended to add these to the requirements file:
> ```
> pandas
> numpy
> matplotlib
> seaborn
> scikit-learn
> folium
> joblib
> ```

## Usage

### 1. Run diagnostics (recommended first step)

```bash
python debug_check.py
```

Checks your Python version, required files, directories, installed packages, and dataset structure.

### 2. Run the full pipeline

```bash
python main.py
```

This runs preprocessing → EDA → model training end to end, and saves the trained model, encoders, metadata, and plots.

For a step-by-step version with progress messages and error handling (useful while debugging):

```bash
python main_simple.py
```

### 3. Make predictions

**Command line:**

```bash
python predict.py \
  --lat 18.64 --lon 70.03 \
  --rainfall 23.75 --temp 44.89 --humidity 30.21 \
  --discharge 2609.75 --water-level 4.79 --elevation 3931.98 \
  --land-cover "Water Bodies" --soil-type Sandy \
  --population 9397.87 --infrastructure 0 --historical 1
```

**GUI:**

```bash
python flood_gui.py
```

Enter the environmental parameters (or click a Low/Medium/High sample button) and click **Predict Flood Risk** to see a color-coded risk assessment with recommendations.

### 4. Extract representative samples (optional)

```bash
python extract_sample.py
```

Scans the dataset and prints example rows corresponding to low, medium, and high flood-probability predictions — useful for testing or demos.

## Risk Levels

Flood probability is bucketed into three risk levels:

| Level | Probability |
|---|---|
| 🟢 LOW | ≤ 40% |
| 🟡 MEDIUM | 41% – 60% |
| 🔴 HIGH | > 60% |

## Models Compared

- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)

The best model is selected automatically based on F1-score, then retrained on the full dataset and saved to `models/flood_model.pkl`.

## Notes / Known Issues

- The module is named `data_preprossesing.py` (typo) — imports elsewhere in the codebase use this exact spelling, so keep it consistent if you rename it.
- `debug_check.py` and other scripts reference `data_preprocessing.py` (correct spelling) in a couple of places — double-check the filename matches your imports if you hit `ModuleNotFoundError`.
- `requriements.txt` is misspelled and contains a stray `pip install -r requirements.txt` line inside the file itself — clean this up before publishing (see Installation section above).
- `main.py` creates a `dataset/` directory that isn't actually used (data is saved to `data/` by `data_preprossesing.py`).

## License

Add your preferred license here (e.g., MIT).
