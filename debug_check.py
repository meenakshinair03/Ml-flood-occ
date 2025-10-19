#!/usr/bin/env python3
# debug_check.py - Diagnostic script to identify issues

import sys
import os

print("🔍 FLOOD PREDICTION SYSTEM - DIAGNOSTIC CHECK")
print("=" * 60)

# 1. Check Python version
print("\n1️⃣ Checking Python Version...")
print(f"   Python version: {sys.version}")
if sys.version_info < (3, 8):
    print("   ⚠️  WARNING: Python 3.8+ recommended")
else:
    print("   ✅ Python version OK")

# 2. Check required files exist
print("\n2️⃣ Checking Required Files...")
required_files = [
    'flood_risk_dataset_india.csv',
    'data_preprocessing.py',
    'eda.py',
    'training.py',
    'main.py'
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} - MISSING!")
        missing_files.append(file)

if missing_files:
    print(f"\n   ⚠️  Missing files: {', '.join(missing_files)}")
    print("   Please ensure all required files are in the current directory.")

# 3. Check directories
print("\n3️⃣ Checking/Creating Directories...")
required_dirs = ['data', 'models', 'plots']
for dir_name in required_dirs:
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print(f"   ✅ Created directory: {dir_name}/")
        except Exception as e:
            print(f"   ❌ Failed to create {dir_name}/: {e}")
    else:
        print(f"   ✅ {dir_name}/ exists")

# 4. Check required packages
print("\n4️⃣ Checking Required Packages...")
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'folium': 'folium',
    'joblib': 'joblib'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠️  Missing packages: {', '.join(missing_packages)}")
    print(f"   Install with: pip install {' '.join(missing_packages)}")

# 5. Check dataset structure
print("\n5️⃣ Checking Dataset...")
if os.path.exists('flood_risk_dataset_india.csv'):
    try:
        import pandas as pd
        df = pd.read_csv('flood_risk_dataset_india.csv')
        print(f"   ✅ Dataset loaded successfully")
        print(f"   📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check for required columns
        required_columns = [
            'Latitude', 'Longitude', 'Rainfall (mm)', 'Temperature (°C)',
            'Humidity (%)', 'River Discharge (m³/s)', 'Water Level (m)',
            'Elevation (m)', 'Land Cover', 'Soil Type', 'Population',
            'Infrastructure', 'Historical Flooding', 'Flood Occurred'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"   ❌ Missing columns: {missing_columns}")
            print(f"\n   Available columns:")
            for col in df.columns:
                print(f"      - {col}")
        else:
            print(f"   ✅ All required columns present")
            
        # Check for missing values
        missing_vals = df.isnull().sum().sum()
        if missing_vals > 0:
            print(f"   ⚠️  Dataset has {missing_vals} missing values")
        else:
            print(f"   ✅ No missing values")
            
    except Exception as e:
        print(f"   ❌ Error reading dataset: {e}")
else:
    print("   ❌ Dataset file not found")

# 6. Test imports from project files
print("\n6️⃣ Testing Project File Imports...")
test_imports = [
    ('data_preprocessing', 'load_and_clean_data'),
    ('eda', 'perform_eda'),
    ('training', 'train_and_evaluate')
]

for module_name, function_name in test_imports:
    try:
        module = __import__(module_name)
        if hasattr(module, function_name):
            print(f"   ✅ {module_name}.{function_name}()")
        else:
            print(f"   ❌ {module_name}.py exists but missing {function_name}()")
    except ImportError as e:
        print(f"   ❌ Cannot import {module_name}: {e}")
    except Exception as e:
        print(f"   ⚠️  {module_name}: {e}")

# Summary
print("\n" + "=" * 60)
print("📋 SUMMARY")
print("=" * 60)

issues = []
if missing_files:
    issues.append(f"Missing files: {', '.join(missing_files)}")
if missing_packages:
    issues.append(f"Missing packages: {', '.join(missing_packages)}")

if issues:
    print("\n❌ ISSUES FOUND:")
    for issue in issues:
        print(f"   • {issue}")
    print("\n📝 NEXT STEPS:")
    if missing_packages:
        print(f"   1. Install packages: pip install {' '.join(missing_packages)}")
    if missing_files:
        print(f"   2. Create missing files: {', '.join(missing_files)}")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("\n🚀 You can now run: python main.py")

print("\n" + "=" * 60)