# main_simple.py - Simplified version for debugging

import os
import sys

print("🌊 Flood Risk Prediction System - Simple Version")
print("=" * 60)

# Create directories
print("\n📁 Creating directories...")
for dir_name in ['data', 'models', 'plots']:
    os.makedirs(dir_name, exist_ok=True)
    print(f"   ✅ {dir_name}/")

# Step 1: Data Preprocessing
print("\n" + "=" * 60)
print("STEP 1: DATA PREPROCESSING")
print("=" * 60)

try:
    from data_preprossesing import load_and_clean_data
    print("✅ Imported data_preprocessing module")
    
    print("\n🔄 Loading and cleaning data...")
    df, encoders = load_and_clean_data("flood_risk_dataset_india.csv")
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
except ImportError as e:
    print(f"❌ ERROR: Cannot import data_preprocessing module")
    print(f"   Details: {e}")
    print("\n💡 Make sure data_preprocessing.py exists in the current directory")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"❌ ERROR: Dataset file not found")
    print(f"   Details: {e}")
    print("\n💡 Make sure 'flood_risk_dataset_india.csv' exists in the current directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR during data preprocessing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: EDA (Optional - can skip if causing issues)
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

try:
    from eda import perform_eda
    print("✅ Imported eda module")
    
    response = input("\n❓ Run EDA? This will generate plots (y/n): ").lower()
    if response == 'y':
        print("\n🔄 Performing EDA...")
        perform_eda(df)
        print("✅ EDA completed")
    else:
        print("⏭️  Skipping EDA")
    
except ImportError as e:
    print(f"⚠️  WARNING: Cannot import eda module: {e}")
    print("   Skipping EDA step...")
except Exception as e:
    print(f"⚠️  WARNING: Error during EDA: {e}")
    print("   Continuing with training...")

# Step 3: Model Training
print("\n" + "=" * 60)
print("STEP 3: MODEL TRAINING")
print("=" * 60)

try:
    from training import train_and_evaluate
    print("✅ Imported training module")
    
    print("\n🔄 Training models...")
    print("   (This may take a few minutes...)")
    
    model, results = train_and_evaluate(df, target_col='Flood Occurred', encoders=encoders)
    
    print("✅ Training completed successfully!")
    
except ImportError as e:
    print(f"❌ ERROR: Cannot import training module")
    print(f"   Details: {e}")
    print("\n💡 Make sure training.py exists in the current directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 60)
print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\n📁 Generated files:")
print("   • data/cleaned_flood_data.csv")
print("   • models/flood_model.pkl")
print("   • models/label_encoders.pkl")
print("   • models/model_metadata.pkl")
print("   • plots/*.png")

print("\n🚀 Next steps:")
print("   • Run the GUI: python flood_gui.py")
print("   • Make predictions: python predict.py --help")

print("\n" + "=" * 60)