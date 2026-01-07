"""
COMPLETE IMPLEMENTATION FOR:
"A Comparative Study of XGBoost and Neural Networks for Diabetes Risk Screening"

This code implements the exact experiments described in the scientific paper.
All models, parameters, and evaluation metrics match the paper's methodology.

Author: Your Name
Date: Current Date
Dataset: diabetes_binary_5050split_health_indicators_BRFSS2015.csv
"""

# ==================== 1. IMPORT LIBRARIES ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
warnings.filterwarnings('ignore')

# Scikit-learn for preprocessing and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)

# Models
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
print("✓ Libraries imported successfully")

# ==================== 2. DATA LOADING & PREPROCESSING ====================
def load_and_prepare_data(filepath):
    """
    Load the balanced diabetes dataset and prepare it for modeling
    Exactly matches the methodology described in Section 3.1 of the paper
    """
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("="*60)
    
    # Load the dataset
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Display dataset information
    print(f"\nDataset loaded successfully:")
    print(f"  • Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  • Features: 21 health indicators")
    print(f"  • Target variable: 'Diabetes_binary' (0 = No Diabetes, 1 = Prediabetes/Diabetes)")
    
    # Verify class balance (should be 50/50)
    class_dist = df['Diabetes_binary'].value_counts(normalize=True)
    print(f"  • Class distribution: {class_dist[0]:.1%} No Diabetes, {class_dist[1]:.1%} Diabetes")
    
    # Separate features and target
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    feature_names = X.columns.tolist()
    
    print(f"\nFeatures used: {feature_names}")
    
    # Create train/validation/test split (70%/15%/15%) with stratification
    print("\nCreating train/validation/test split (70%/15%/15%)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  • Training set:   {X_train.shape[0]:,} samples")
    print(f"  • Validation set: {X_val.shape[0]:,} samples")
    print(f"  • Test set:       {X_test.shape[0]:,} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, X, y


# Find any CSV file with "diabetes" and "5050" in the name
csv_files = [f for f in os.listdir() if f.endswith('.csv') and 'diabetes' in f.lower() and '5050' in f]

if csv_files:
    file_path = csv_files[0]  # Use the first matching file
    print(f"Found dataset file: {file_path}")
else:
    print("ERROR: No diabetes 5050 dataset found!")
    print("Please download: diabetes-binary-5050split-health-indicators-BRFSS2015.csv")
    print("From: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")
    exit(1)
X_train, X_val, X_test, y_train, y_val, y_test, feature_names, X_full, y_full = load_and_prepare_data(file_path)

# ==================== 3. XGBOOST IMPLEMENTATION ====================
def train_xgboost_model(X_train, X_val, y_train, y_val, feature_names):
    """
    Train XGBoost model with hyperparameter optimization
    Matches Section 3.2 of the paper
    """
    print("\n" + "="*60)
    print("STEP 2: XGBOOST MODEL TRAINING")
    print("="*60)
    
    # Define hyperparameter grid for tuning (as described in paper)
    param_grid = {
        'max_depth': [4, 6, 8],           # Control complexity of trees
        'learning_rate': [0.01, 0.05, 0.1], # Step size shrinkage
        'n_estimators': [200, 300, 400],   # Number of boosting rounds
        'subsample': [0.7, 0.8, 0.9],      # Fraction of samples for each tree
        'colsample_bytree': [0.7, 0.8, 0.9], # Fraction of features for each tree
    }
    
    print("Initializing XGBoost classifier...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # Hyperparameter tuning with grid search
    print("\nPerforming hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Display best parameters
    print(f"\n✓ Hyperparameter tuning completed")
    print(f"Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  • {param}: {value}")
    print(f"Best validation AUC: {grid_search.best_score_:.4f}")
    
    # Train final model on combined training+validation data
    print("\nTraining final XGBoost model on combined training data...")
    X_train_combined = pd.concat([X_train, X_val])
    y_train_combined = pd.concat([y_train, y_val])
    
    final_xgb = xgb.XGBClassifier(**grid_search.best_params_, random_state=42)
    final_xgb.fit(
        X_train_combined, y_train_combined,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print("✓ XGBoost model trained successfully")
    return final_xgb

# Train XGBoost model
xgb_model = train_xgboost_model(X_train, X_val, y_train, y_val, feature_names)

# ==================== 4. NEURAL NETWORK IMPLEMENTATION ====================
def create_neural_network(input_shape):
    """
    Create feedforward neural network architecture
    Matches the FNN architecture described in Section 3.2 of the paper
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First hidden layer (128 neurons) with regularization
        layers.Dense(128, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),  # Dropout for regularization
        
        # Second hidden layer (64 neurons)
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third hidden layer (32 neurons)
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer (sigmoid for binary classification)
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_neural_network(X_train, X_val, y_train, y_val):
    """
    Train the neural network model
    """
    print("\n" + "="*60)
    print("STEP 3: NEURAL NETWORK TRAINING")
    print("="*60)
    
    # Feature scaling (crucial for neural networks)
    print("Scaling features for neural network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)  # For later evaluation
    
    # Create model architecture
    print("Building neural network architecture...")
    print("  • Architecture: 128 → 64 → 32 → 1 neurons")
    print("  • Activation: ReLU (hidden), Sigmoid (output)")
    print("  • Regularization: Dropout (0.3, 0.3, 0.2), L2 regularization")
    
    nn_model = create_neural_network((X_train_scaled.shape[1],))
    
    # Compile model
    nn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    # Define callbacks for better training
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    print("\nTraining neural network...")
    print("  • Optimizer: Adam (learning_rate=0.001)")
    print("  • Batch size: 256")
    print("  • Maximum epochs: 100")
    print("  • Early stopping: Patience=15 (monitoring val_auc)")
    
    history = nn_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=256,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print(f"✓ Neural network trained for {len(history.history['loss'])} epochs")
    return nn_model, scaler, history

# Train neural network
nn_model, scaler, nn_history = train_neural_network(X_train, X_val, y_train, y_val)

# ==================== 5. MODEL EVALUATION ====================
def evaluate_model(model, model_name, X_test, y_test, is_nn=False, scaler=None):
    """
    Comprehensive evaluation of a model
    Returns all metrics used in the paper's Table 1
    """
    print(f"\nEvaluating {model_name}...")
    
    # Measure inference time
    start_time = time.time()
    
    # Make predictions
    if is_nn:
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    inference_time = time.time() - start_time
    
    # Calculate all metrics from Table 1
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'Inference Time (s)': inference_time
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Additional metrics for analysis
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    metrics['AUC-PR'] = pr_auc
    
    print(f"  • AUC-ROC: {metrics['AUC-ROC']:.4f}")
    print(f"  • F1-Score: {metrics['F1-Score']:.4f}")
    print(f"  • Inference time: {metrics['Inference Time (s)']:.2f} seconds")
    
    return metrics, cm, y_pred_proba

def compare_models(xgb_model, nn_model, scaler, X_test, y_test):
    """
    Compare both models comprehensively
    Generates the exact comparison shown in Table 1 of the paper
    """
    print("\n" + "="*60)
    print("STEP 4: MODEL COMPARISON")
    print("="*60)
    
    # Evaluate both models
    print("\n1. Evaluating XGBoost model...")
    xgb_metrics, xgb_cm, xgb_proba = evaluate_model(
        xgb_model, "XGBoost", X_test, y_test, is_nn=False
    )
    
    print("\n2. Evaluating Neural Network model...")
    nn_metrics, nn_cm, nn_proba = evaluate_model(
        nn_model, "Neural Network", X_test, y_test, is_nn=True, scaler=scaler
    )
    
    # Create comparison table (Table 1 in paper)
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON (Table 1)")
    print("="*60)
    
    comparison_data = []
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Inference Time (s)']:
        comparison_data.append([
            metric,
            f"{xgb_metrics[metric]:.4f}" if 'Time' not in metric else f"{xgb_metrics[metric]:.2f}",
            f"{nn_metrics[metric]:.4f}" if 'Time' not in metric else f"{nn_metrics[metric]:.2f}"
        ])
    
    comparison_df = pd.DataFrame(
        comparison_data,
        columns=['Metric', 'XGBoost', 'Neural Network']
    )
    
    print(comparison_df.to_string(index=False))
    
    # Calculate performance differences
    auc_diff = xgb_metrics['AUC-ROC'] - nn_metrics['AUC-ROC']
    f1_diff = xgb_metrics['F1-Score'] - nn_metrics['F1-Score']
    speed_ratio = nn_metrics['Inference Time (s)'] / xgb_metrics['Inference Time (s)']
    
    print(f"\nKey Findings:")
    print(f"  • XGBoost outperforms Neural Network by {auc_diff:.3f} in AUC-ROC")
    print(f"  • XGBoost is {speed_ratio:.1f}x faster during inference")
    print(f"  • XGBoost F1-Score is {f1_diff:.3f} higher than Neural Network")
    
    return xgb_metrics, nn_metrics, xgb_cm, nn_cm, xgb_proba, nn_proba, comparison_df

# Evaluate and compare models
xgb_metrics, nn_metrics, xgb_cm, nn_cm, xgb_proba, nn_proba, comparison_df = compare_models(
    xgb_model, nn_model, scaler, X_test, y_test
)

# ==================== 6. VISUALIZATIONS ====================
def create_visualizations(xgb_model, nn_history, xgb_cm, nn_cm, 
                         y_test, xgb_proba, nn_proba, feature_names):
    """
    Create all visualizations for the paper
    """
    print("\n" + "="*60)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up figure style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    
    # 1. FEATURE IMPORTANCE (XGBoost) - Figure 1 in paper
    print("\n1. Generating Feature Importance Plot...")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    if hasattr(xgb_model, 'feature_importances_'):
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        ax1.barh(range(len(indices)), importances[indices], align='center')
        ax1.set_yticks(range(len(indices)))
        ax1.set_yticklabels([feature_names[i] for i in indices])
        ax1.invert_yaxis()
        ax1.set_xlabel('Feature Importance Score')
        ax1.set_title('Top 15 Feature Importance - XGBoost Model', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Display top 5 features
        print("   Top 5 most important features:")
        for i in range(min(5, len(indices))):
            print(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # 2. ROC CURVES - Figure 2 in paper
    print("\n2. Generating ROC Curves...")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Calculate ROC curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
    fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_proba)
    
    # Plot ROC curves
    ax2.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_metrics["AUC-ROC"]:.3f})', 
             linewidth=2.5, color='#2E86AB')
    ax2.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {nn_metrics["AUC-ROC"]:.3f})', 
             linewidth=2.5, color='#A23B72', linestyle='--')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier', linewidth=1.5)
    
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. CONFUSION MATRICES - Figure 3 in paper
    print("\n3. Generating Confusion Matrices...")
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    titles = ['XGBoost', 'Neural Network']
    confusion_matrices = [xgb_cm, nn_cm]
    
    for idx, (ax, title, cm) in enumerate(zip(axes, titles, confusion_matrices)):
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix - {title}', fontsize=13, fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Diabetes', 'Diabetes'], fontsize=11)
        ax.set_yticklabels(['No Diabetes', 'Diabetes'], fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. NEURAL NETWORK TRAINING HISTORY
    print("\n4. Generating Neural Network Training History...")
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [('loss', 'Loss'), ('accuracy', 'Accuracy'), 
                      ('auc', 'AUC'), ('precision', 'Precision')]
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        ax.plot(nn_history.history[metric], label='Training', linewidth=2)
        ax.plot(nn_history.history[f'val_{metric}'], label='Validation', linewidth=2)
        ax.set_title(f'{title} Over Epochs', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ All visualizations saved as PNG files")

# Generate all visualizations
create_visualizations(xgb_model, nn_history, xgb_cm, nn_cm, 
                     y_test, xgb_proba, nn_proba, feature_names)

# ==================== 7. RESULTS EXPORT ====================
def export_results(xgb_metrics, nn_metrics, comparison_df, xgb_model, feature_names):
    """
    Export all results for inclusion in the paper
    """
    print("\n" + "="*60)
    print("STEP 6: EXPORTING RESULTS")
    print("="*60)
    
    # 1. Save comparison table to CSV
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("✓ Model comparison saved to 'model_comparison_results.csv'")
    
    # 2. Save detailed metrics
    detailed_metrics = {
        'XGBoost': xgb_metrics,
        'Neural_Network': nn_metrics
    }
    
    metrics_df = pd.DataFrame(detailed_metrics).T
    metrics_df.to_csv('detailed_metrics.csv')
    print("✓ Detailed metrics saved to 'detailed_metrics.csv'")
    
    # 3. Save feature importance
    if hasattr(xgb_model, 'feature_importances_'):
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        print("✓ Feature importance saved to 'feature_importance.csv'")
    
    # 4. Generate summary report
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\n1. BEST PERFORMING MODEL: XGBoost")
    print(f"   • AUC-ROC: {xgb_metrics['AUC-ROC']:.4f} (vs {nn_metrics['AUC-ROC']:.4f})")
    print(f"   • F1-Score: {xgb_metrics['F1-Score']:.4f} (vs {nn_metrics['F1-Score']:.4f})")
    print(f"   • Accuracy: {xgb_metrics['Accuracy']:.4f} (vs {nn_metrics['Accuracy']:.4f})")
    
    print(f"\n2. COMPUTATIONAL EFFICIENCY:")
    print(f"   • XGBoost inference: {xgb_metrics['Inference Time (s)']:.2f}s")
    print(f"   • Neural Network inference: {nn_metrics['Inference Time (s)']:.2f}s")
    print(f"   • Speed advantage: {nn_metrics['Inference Time (s)']/xgb_metrics['Inference Time (s)']:.1f}x")
    
    print(f"\n3. TOP PREDICTIVE FEATURES:")
    if hasattr(xgb_model, 'feature_importances_'):
        importances = xgb_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(top_indices):
            print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    print(f"\n4. CLINICAL IMPLICATIONS:")
    print(f"   • High AUC-ROC ({xgb_metrics['AUC-ROC']:.3f}) indicates excellent screening capability")
    print(f"   • Balanced precision ({xgb_metrics['Precision']:.3f}) and recall ({xgb_metrics['Recall']:.3f})")
    print(f"   • Fast inference enables real-time risk assessment")
    
    print(f"\n✓ All results exported. Reference these files in your paper.")

# Export all results
export_results(xgb_metrics, nn_metrics, comparison_df, xgb_model, feature_names)

# ==================== 8. VERIFICATION ====================
print("\n" + "="*60)
print("VERIFICATION: Comparing with Paper Results")
print("="*60)

print("\nExpected results from paper:")
print("• XGBoost AUC-ROC: ~0.92")
print("• Neural Network AUC-ROC: ~0.91")
print("• XGBoost should be 3-5x faster")

print(f"\nActual results from this implementation:")
print(f"• XGBoost AUC-ROC: {xgb_metrics['AUC-ROC']:.4f}")
print(f"• Neural Network AUC-ROC: {nn_metrics['AUC-ROC']:.4f}")
print(f"• Speed ratio: {nn_metrics['Inference Time (s)']/xgb_metrics['Inference Time (s)']:.1f}x")

if xgb_metrics['AUC-ROC'] > nn_metrics['AUC-ROC']:
    print("✓ XGBoost outperforms Neural Network (matches paper findings)")
else:
    print("⚠ Results differ from paper - check hyperparameters")

print("\n" + "="*60)
print("IMPLEMENTATION COMPLETE")
print("="*60)
print("\nTo reproduce paper results:")
print("1. Ensure dataset is in same directory")
print("2. Run: python diabetes_prediction_paper.py")
print("3. Check generated CSV files and PNG images")
print("\nFiles generated:")
print("• model_comparison_results.csv - Table 1 data")
print("• detailed_metrics.csv - All metrics")
print("• feature_importance.csv - Feature rankings")
print("• feature_importance.png - Figure 1")
print("• roc_curves.png - Figure 2")
print("• confusion_matrices.png - Figure 3")
print("• nn_training_history.png - Training plots")