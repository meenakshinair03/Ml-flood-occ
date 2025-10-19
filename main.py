# main.py

from data_preprossesing import load_and_clean_data
from eda import perform_eda
from training import train_and_evaluate
import os

def main():
    print("🌊 Flood Risk Prediction System\n")
    print("="*50)

    # Create necessary directories
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 1: Load and clean data
    print("\n📂 Step 1: Loading and Cleaning Data...")
    df, encoders = load_and_clean_data("flood_risk_dataset_india.csv")

    # Step 2: Perform EDA
    print("\n📊 Step 2: Performing Exploratory Data Analysis...")
    perform_eda(df)

    # Step 3: Train and evaluate model
    print("\n🤖 Step 3: Training and Evaluating Model...")
    train_and_evaluate(df, target_col='Flood Occurred', encoders=encoders)

    print("\n" + "="*50)
    print("✅ All steps completed successfully!")

if __name__ == "__main__":
    main()