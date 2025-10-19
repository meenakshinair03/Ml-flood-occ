# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    # Step 1: Load dataset
    df = pd.read_csv(filepath)
    print("✅ Dataset loaded successfully!")
    print(df.head())

    # Step 2: Check missing values
    print("\n🔍 Missing Values Before Cleaning:")
    print(df.isnull().sum())

    # Step 3: Drop or fill missing data (you can modify this based on dataset)
    df = df.dropna()

    # Step 4: Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print("\n✅ Data cleaned and encoded.")
    print(df.info())

    return df


if __name__ == "__main__":
    # 👇 Step 5: Define file path and run the function
    filepath = "flood_risk_dataset_india.csv"   # make sure this path is correct

    df_cleaned = load_and_clean_data(filepath)

    # 👇 Step 6: Save cleaned data to new file
    output_path = "data/cleaned_flood_data.csv"
    df_cleaned.to_csv(output_path, index=False)

    print(f"\n💾 Cleaned data saved to: {output_path}")
