import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize, StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_CLASSES = 4
CLASS_NAMES = ['Abnormal Heartbeat', 'Myocardial Infarction', 'Normal', 'History of MI']

def load_main_model():
    """Load the pre-trained main model and PCA transformer."""
    try:
        model_path = r"Cardiovascular-Disease-Detection-From-ECG-Images\Deployment\Heart_Disease_Prediction_using_ECG.pkl"
        pca_path = r"Cardiovascular-Disease-Detection-From-ECG-Images\Deployment\PCA_ECG.pkl"
        
        model = joblib.load(model_path)
        pca = joblib.load(pca_path)
        print("Main model and PCA loaded successfully!")
        return model, pca
    except Exception as e:
        print(f"Error loading main model or PCA: {e}")
        return None, None

def create_1d_cnn_model(input_shape, n_classes):
    """Create a 1D CNN model for ECG classification."""
    model = Sequential([
        # First Conv1D layer
        Conv1D(filters=64, kernel_size=3, activation='relu', 
              input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.3),
        
        # Second Conv1D layer
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.3),
        
        # Third Conv1D layer
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.3),
        
        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def generate_synthetic_data(n_samples=2000, n_features=400, n_classes=4, random_state=42):
    """Generate synthetic ECG-like data for demonstration."""
    np.random.seed(random_state)
    
    # Generate random data with some structure
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Add some signal to the data
    for i in range(n_classes):
        class_indices = np.where(y == i)[0]
        X[class_indices, i*100:(i+1)*100] += 2.0  # Add class-specific signal
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def plot_model_comparison(results, class_names):
    """Plot comparison of models' performance."""
    # Plot ROC curves
    plt.figure(figsize=(15, 6))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    for model_name, metrics in results.items():
        fpr, tpr, roc_auc = metrics['roc']
        plt.plot(fpr, tpr, lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    
    # Confusion Matrix for each model
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, metrics) in zip(axes, results.items()):
        cm = metrics['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_comparison_1dcnn.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load the main model and PCA
    main_model, pca = load_main_model()
    if main_model is None or pca is None:
        print("Failed to load main model or PCA. Exiting...")
        return
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    
    # Apply PCA transformation
    try:
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("Applied PCA transformation to data.")
    except Exception as e:
        print(f"Error applying PCA: {e}")
        X_train_pca = X_train
        X_test_pca = X_test
    
    # Dictionary to store results
    results = {}
    
    # Evaluate main model
    print("\nEvaluating main model...")
    if hasattr(main_model, "predict_proba"):
        y_scores = main_model.predict_proba(X_test_pca)
    else:
        y_scores = main_model.decision_function(X_test_pca)
        if len(y_scores.shape) == 1:  # Binary classification
            y_scores = np.column_stack([-y_scores, y_scores])
    
    # Calculate metrics for main model
    y_pred = np.argmax(y_scores, axis=1)
    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=range(N_CLASSES)).ravel(), 
                           y_scores.ravel())
    roc_auc = auc(fpr, tpr)
    
    results['Main Model'] = {
        'roc': (fpr, tpr, roc_auc),
        'cm': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    }
    
    # Train and evaluate 1D CNN model
    print("\nTraining 1D CNN model...")
    
    # Reshape data for CNN (samples, timesteps, features)
    X_train_cnn = np.expand_dims(X_train_pca, axis=2)
    X_test_cnn = np.expand_dims(X_test_pca, axis=2)
    
    # Convert labels to one-hot encoding
    y_train_oh = to_categorical(y_train, N_CLASSES)
    y_test_oh = to_categorical(y_test, N_CLASSES)
    
    # Create and train 1D CNN model
    cnn_model = create_1d_cnn_model((X_train_cnn.shape[1], 1), N_CLASSES)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = cnn_model.fit(
        X_train_cnn, y_train_oh,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate 1D CNN model
    print("\nEvaluating 1D CNN model...")
    y_scores_cnn = cnn_model.predict(X_test_cnn)
    y_pred_cnn = np.argmax(y_scores_cnn, axis=1)
    
    fpr_cnn, tpr_cnn, _ = roc_curve(label_binarize(y_test, classes=range(N_CLASSES)).ravel(), 
                                   y_scores_cnn.ravel())
    roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
    
    results['1D CNN'] = {
        'roc': (fpr_cnn, tpr_cnn, roc_auc_cnn),
        'cm': confusion_matrix(y_test, y_pred_cnn),
        'report': classification_report(y_test, y_pred_cnn, target_names=CLASS_NAMES)
    }
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_model_comparison(results, CLASS_NAMES)
    
    # Print classification reports
    print("\n" + "="*80)
    print("MAIN MODEL CLASSIFICATION REPORT")
    print("="*80)
    print(results['Main Model']['report'])
    
    print("\n" + "="*80)
    print("1D CNN CLASSIFICATION REPORT")
    print("="*80)
    print(results['1D CNN']['report'])
    
    # Save the trained CNN model
    cnn_model.save('1d_cnn_ecg_model.h5')
    print("\n1D CNN model saved as '1d_cnn_ecg_model.h5'")

if __name__ == "__main__":
    main()
