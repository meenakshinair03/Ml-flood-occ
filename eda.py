# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster

def perform_eda(df):
    print("\n📊 Performing Exploratory Data Analysis...\n")

    # 1. Dataset info
    print("🔹 Dataset Info:")
    print(df.info(), "\n")

    # 2. Summary statistics
    print("🔹 Summary Statistics:")
    print(df.describe(include='all'), "\n")

    # 3. Missing values
    print("🔹 Missing Values:")
    print(df.isnull().sum(), "\n")

    # 4. Target distribution
    if 'Flood Occurred' in df.columns:
        print("🔹 Flood Occurred Distribution:")
        print(df['Flood Occurred'].value_counts())
        sns.countplot(x='Flood Occurred', data=df)
        plt.title("Flood Occurred Count")
        plt.show()

    # 5. Correlation heatmap (numerical features)
    plt.figure(figsize=(12,8))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # 6. Numerical features vs Flood Occurred
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols.remove('Flood Occurred')
    for col in numeric_cols:
        plt.figure(figsize=(8,5))
        sns.boxplot(x='Flood Occurred', y=col, data=df)
        plt.title(f"{col} vs Flood Occurred")
        plt.show()

    # 7. Categorical features vs Flood Occurred
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue='Flood Occurred', data=df)
        plt.title(f"{col} vs Flood Occurred")
        plt.xticks(rotation=45)
        plt.show()

    # 8. Map visualization: Flood occurrences
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        print("🔹 Generating Flood Occurrence Map...")
        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=5)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df.iterrows():
            color = 'red' if row['Flood Occurred'] == 1 else 'green'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Rainfall: {row['Rainfall (mm)']}, Water Level: {row['Water Level (m)']}"
            ).add_to(marker_cluster)

        m.save("flood_occurrence_map.html")
        print("🌍 Map saved as flood_occurrence_map.html")

    print("\n✅ EDA Completed.")

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("flood_risk_dataset_india.csv")  # replace with your dataset
    perform_eda(df)
