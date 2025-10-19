# data_preprocessing.py

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    """Load and clean the flood dataset"""
    
    # Step 1: Load dataset
    print(f"📂 Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print("✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 First few rows:")
    print(df.head())

    # Step 2: Check missing values
    print("\n🔍 Checking for missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("⚠️  Missing values found:")
        print(missing[missing > 0])
    else:
        print("✅ No missing values found")

    # Step 3: Handle missing data (if any)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"🗑️  Dropped {dropped_rows} rows with missing values")
    
    print(f"✅ Clean dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Step 4: Encode categorical columns
    print("\n🔤 Encoding categorical features...")
    categorical_cols = ['Land Cover', 'Soil Type']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Create new encoded column (keep original for EDA)
            df[col + '_Encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
            
            # Show encoding mapping
            mapping = dict(enumerate(le.classes_))
            print(f"   ✅ {col}:")
            for code, value in mapping.items():
                print(f"      {code} → {value}")
        else:
            print(f"   ⚠️  Column '{col}' not found in dataset")

    print("\n✅ Data preprocessing completed!")
    print(f"📊 Final dataset info:")
    print(f"   Rows: {df.shape[0]}")
    print(f"   Columns: {df.shape[1]}")
    print(f"   Encoded columns added: {[col + '_Encoded' for col in categorical_cols]}")
    
    # Return both dataframe and encoders
    return df, label_encoders


if __name__ == "__main__":
    # Test the preprocessing
    print("=" * 60)
    print("TESTING DATA PREPROCESSING")
    print("=" * 60 + "\n")
    
    filepath = "flood_risk_dataset_india.csv"
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found!")
        print("   Make sure the dataset is in the current directory")
        exit(1)
    
    df_cleaned, encoders = load_and_clean_data(filepath)

    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save cleaned data
    output_path = "data/cleaned_flood_data.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n💾 Cleaned data saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING TEST COMPLETED")
    print("=" * 60)