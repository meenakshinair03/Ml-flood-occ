# main.py

from data_preprossesing import load_and_clean_data
from eda import perform_eda
from training import train_and_evaluate

def main():
    print("🌊 Flood Risk Prediction System")

    # Step 1: Load and clean data
    df = load_and_clean_data("flood_risk_dataset_india.csv")

    # Step 2: Perform EDA
    perform_eda(df)

    # Step 3: Train and evaluate model
    train_and_evaluate(df, target_col='Flood_occurred')

if __name__ == "__main__":
    main()
