import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up the figure
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12

# Data from the analysis
models = ['SVM', 'KNN', 'Logistic\nRegression', 'XGBoost', 'Ensemble\nVoting']
test_accuracies = [90.5, 79.3, 77.9, 85.1, 92.8]
cross_val_scores = [85.2, 78.1, 76.8, 82.3, 90.5]

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model Performance Comparison
bars = ax1.bar(models, test_accuracies, color=['#c38888', '#d4a5a5', '#e5c2c2', '#f6dfdf', '#ff6b6b'])
ax1.set_title('Model Performance Comparison (Test Accuracy)', fontsize=14, fontweight='bold', color='#c38888')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim(70, 95)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, test_accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{acc}%', ha='center', va='bottom', fontweight='bold')

# 2. Training vs Validation Accuracy
x = np.arange(len(models))
width = 0.35

bars1 = ax2.bar(x - width/2, test_accuracies, width, label='Test Accuracy', color='#c38888', alpha=0.8)
bars2 = ax2.bar(x + width/2, cross_val_scores, width, label='Cross-Validation Score', color='#ff6b6b', alpha=0.8)

ax2.set_title('Training vs Validation Performance', fontsize=14, fontweight='bold', color='#c38888')
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_xlabel('Models', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(70, 95)

# 3. Class-wise Performance for Ensemble Model
classes = ['Abnormal\nHeartbeat', 'Myocardial\nInfarction', 'Normal', 'History\nof MI']
precision = [100, 100, 84, 86]
recall = [95, 100, 92, 79]
f1_score = [97, 100, 88, 83]

x = np.arange(len(classes))
width = 0.25

bars1 = ax3.bar(x - width, precision, width, label='Precision', color='#c38888', alpha=0.8)
bars2 = ax3.bar(x, recall, width, label='Recall', color='#ff6b6b', alpha=0.8)
bars3 = ax3.bar(x + width, f1_score, width, label='F1-Score', color='#90EE90', alpha=0.8)

ax3.set_title('Ensemble Model: Class-wise Performance', fontsize=14, fontweight='bold', color='#c38888')
ax3.set_ylabel('Performance (%)', fontsize=12)
ax3.set_xlabel('Cardiac Conditions', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(classes)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(70, 105)

# 4. Dataset Split and Training Progress
# Simulated training progress
epochs = np.arange(1, 11)
train_acc = np.array([75, 78, 82, 85, 87, 89, 90, 91, 91.5, 92])
val_acc = np.array([73, 76, 80, 83, 85, 87, 88, 89, 89.5, 90])

ax4.plot(epochs, train_acc, 'o-', label='Training Accuracy', color='#c38888', linewidth=2, markersize=6)
ax4.plot(epochs, val_acc, 's-', label='Validation Accuracy', color='#ff6b6b', linewidth=2, markersize=6)
ax4.set_title('Training Progress (Simulated)', fontsize=14, fontweight='bold', color='#c38888')
ax4.set_xlabel('Epochs', fontsize=12)
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(70, 95)

# Add main title
fig.suptitle('ECG Classification System: Training & Validation Metrics\nMyocardia - Cardiovascular Disease Detection', 
             fontsize=16, fontweight='bold', color='#c38888', y=0.95)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90)

# Save the plot
plt.savefig('ECG_Training_Validation_Metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Create performance summary table
fig2, ax = plt.subplots(figsize=(12, 6))

# Performance summary table
data = {
    'Model': ['SVM', 'K-Nearest Neighbors', 'Logistic Regression', 'XGBoost', 'Ensemble Voting'],
    'Test Accuracy (%)': [90.5, 79.3, 77.9, 85.1, 92.8],
    'CV Score (%)': [85.2, 78.1, 76.8, 82.3, 90.5],
    'Training Time': ['Medium', 'Fast', 'Fast', 'Slow', 'Very Slow'],
    'Complexity': ['High', 'Low', 'Low', 'High', 'Very High']
}

df = pd.DataFrame(data)

# Create table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style the table
for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#c38888')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(df) + 1):
    for j in range(len(df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('white')

ax.set_title('ECG Classification Models: Performance Summary', fontsize=14, fontweight='bold', color='#c38888', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig('ECG_Models_Performance_Summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Graphs generated successfully!")
print("Files saved:")
print("1. ECG_Training_Validation_Metrics.png - Comprehensive metrics visualization")
print("2. ECG_Models_Performance_Summary.png - Performance summary table")
