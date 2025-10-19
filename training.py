# training.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
import os
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df, target_col='Flood Occurred'):
    """Prepare features and target for training"""
    
    # Define feature columns (using actual column names from dataset)
    feature_cols = [
        'Latitude', 'Longitude', 'Rainfall (mm)', 'Temperature (°C)', 
        'Humidity (%)', 'River Discharge (m³/s)', 'Water Level (m)', 
        'Elevation (m)', 'Land Cover_Encoded', 'Soil Type_Encoded',
        'Population Density', 'Infrastructure', 'Historical Floods'
    ]
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target distribution:\n{y.value_counts()}")
    
    return X, y

def train_and_evaluate(df, target_col='Flood Occurred', encoders=None, test_size=0.2, random_state=42):
    """Train multiple models and select the best one"""
    
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING AND EVALUATION")
    print("="*60 + "\n")
    
    # Prepare features
    X, y = prepare_features(df, target_col)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Training set size: {X_train.shape[0]}")
    print(f"📊 Test set size: {X_test.shape[0]}")
    print(f"📊 Train class distribution:\n{y_train.value_counts()}\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to test
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=random_state
        )
    }
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Use scaled data for models that benefit from scaling
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ {name} Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        # Track best model (using F1-score as primary metric)
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_model_name = name
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model_name} (F1-Score: {best_score:.4f})")
    print(f"{'='*60}\n")
    
    # Detailed evaluation of best model
    print(f"\n📊 Detailed Classification Report for {best_model_name}:")
    print(classification_report(y_test, results[best_model_name]['predictions'], 
                               target_names=['No Flood', 'Flood']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Flood', 'Flood'],
                yticklabels=['No Flood', 'Flood'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    print("✅ Confusion matrix saved to plots/confusion_matrix.png")
    plt.close()
    
    # Feature Importance (for tree-based models)
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top 10 Feature Importances ({best_model_name}):")
        print(feature_importance.head(10))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title(f'Top 10 Feature Importances - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        print("✅ Feature importance saved to plots/feature_importance.png")
        plt.close()
    
    # ROC Curve
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png')
    print("✅ ROC curves saved to plots/roc_curves.png")
    plt.close()
    
    # Model comparison
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1-Score': [r['f1'] for r in results.values()],
        'ROC-AUC': [r['roc_auc'] for r in results.values()]
    })
    
    print("\n📊 Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Visualize model comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    comparison_df_melted = comparison_df.melt(id_vars='Model', 
                                              value_vars=metrics,
                                              var_name='Metric', 
                                              value_name='Score')
    
    sns.barplot(data=comparison_df_melted, x='Metric', y='Score', hue='Model', ax=axes[0])
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot for metric distribution
    sns.boxplot(data=comparison_df[metrics], ax=axes[1])
    axes[1].set_title('Metric Distribution Across Models')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    print("✅ Model comparison saved to plots/model_comparison.png")
    plt.close()
    
    # Save the best model and scaler
    os.makedirs("models", exist_ok=True)
    
    # Retrain best model on full dataset for production
    if best_model_name in ['Logistic Regression', 'SVM']:
        X_scaled = scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        joblib.dump(scaler, "models/scaler.pkl")
        print("✅ Scaler saved to models/scaler.pkl")
    else:
        best_model.fit(X, y)
    
    joblib.dump(best_model, "models/flood_model.pkl")
    print(f"✅ Best model ({best_model_name}) saved to models/flood_model.pkl")
    
    # Save encoders if provided
    if encoders:
        joblib.dump(encoders, "models/label_encoders.pkl")
        print("✅ Label encoders saved to models/label_encoders.pkl")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'f1_score': best_score,
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'roc_auc': results[best_model_name]['roc_auc'],
        'feature_columns': X.columns.tolist(),
        'target_column': target_col,
        'uses_scaling': best_model_name in ['Logistic Regression', 'SVM']
    }
    
    joblib.dump(metadata, "models/model_metadata.pkl")
    print("✅ Model metadata saved to models/model_metadata.pkl")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return best_model, results


if __name__ == "__main__":
    # Test training independently
    print("Testing training module...")
    print("="*60)
    
    # Try to load cleaned data first
    if os.path.exists("data/cleaned_flood_data.csv"):
        print("✅ Found cleaned data, loading...")
        df = pd.read_csv("data/cleaned_flood_data.csv")
        
        # Check if it has encoded columns
        if 'Land Cover_Encoded' not in df.columns:
            print("⚠️  Cleaned data doesn't have encoded columns.")
            print("   Running preprocessing first...\n")
            from data_preprossesing import load_and_clean_data
            df, encoders = load_and_clean_data("flood_risk_dataset_india.csv")
        else:
            encoders = None
            
        train_and_evaluate(df, target_col='Flood Occurred', encoders=encoders)
        
    elif os.path.exists("flood_risk_dataset_india.csv"):
        print("⚠️  Cleaned data not found. Using original dataset...")
        print("   Running preprocessing first...\n")
        from data_preprossesing import load_and_clean_data
        df, encoders = load_and_clean_data("flood_risk_dataset_india.csv")
        train_and_evaluate(df, target_col='Flood Occurred', encoders=encoders)
        
    else:
        print("❌ Error: No dataset found!")
        print("   Please ensure either:")
        print("   • data/cleaned_flood_data.csv exists, OR")
        print("   • flood_risk_dataset_india.csv exists")
        print("\n💡 Recommended: Run 'python main.py' instead")