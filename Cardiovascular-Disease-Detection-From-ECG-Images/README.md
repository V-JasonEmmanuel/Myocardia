# Myocardia - Cardiovascular Disease Detection from ECG Images <img src="https://images.emojiterra.com/google/noto-emoji/unicode-15/animated/1fac0.gif" width="50" height="50" alt="Alt text">

**GitHub Repository:** [Myocardia on GitHub](https://github.com/V-JasonEmmanuel/Myocardia)

**Maintainer:** V-JasonEmmanuel

### Abstract:

The electrocardiogram (ECG) is an important tool for identifying cardiovascular issues. Traditionally, ECG records were paper-based, making manual analysis difficult and time-consuming. By digitizing these records, we can automate diagnosis and analysis processes. This project aims to utilize image processing and machine learning to convert ECG images into 1-D signals. It focuses on extracting key components like P, QRS, and T waves, representing heart electrical activity. This automated feature extraction facilitates the diagnosis of various cardiac conditions, enhancing medical analysis and decision-making.

### Key Features:

- **4-Class Classification:** Normal, Abnormal Heartbeat, Myocardial Infarction (MI), History of MI
- **929 Training Samples:** Complete ECG dataset from Mendeley
- **92.8% Accuracy:** Ensemble Voting Classifier (SVM + KNN + Logistic Regression)
- **Web Interface:** Streamlit-based application for easy ECG image analysis
- **Pre-trained Models:** Includes 1D CNN and ensemble ML models
- **Production-Ready:** Dockerized for cloud deployment

### Dataset Information:

**Included in Repository:**
- Normal Person ECG Images: 284 samples
- Abnormal Heartbeat ECG Images: 233 samples
- Myocardial Infarction ECG Images: 240 samples
- History of MI ECG Images: 172 samples

**Source:** [Mendeley ECG Dataset](https://data.mendeley.com/datasets/gwbz3fsgp8/2)

### Getting Started:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/V-JasonEmmanuel/Myocardia.git
   cd Myocardia
   ```

2. **Set Up Environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   pip install -r Cardiovascular-Disease-Detection-From-ECG-Images/Deployment/requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run Cardiovascular-Disease-Detection-From-ECG-Images/Deployment/final_app.py
   ```

4. **Upload ECG Images:**
   - Open the Streamlit app in your browser
   - Upload an ECG image in JPG/PNG format
   - Receive instant cardiac health predictions with confidence scores

### Project Structure:

```
Myocardia/
├── Cardiovascular-Disease-Detection-From-ECG-Images/
│   ├── Deployment/          # Streamlit web app & deployment files
│   ├── colabs/              # Jupyter notebooks for development
│   ├── Combined1d_csv/      # Processed ECG data
│   ├── model_pkl/           # Trained ML models
│   ├── ECG_IMAGES_DATASET/  # Complete ECG training dataset
│   └── README.md
├── compare_with_1dcnn.py    # Model comparison script
├── plot_ecg_metrics.py      # Performance visualization
└── .gitignore, .gitattributes
```

### Algorithms & Models:

**Machine Learning Models (Tested & Deployed):**
- **Ensemble Voting Classifier** (Production) - 92.8% Accuracy
  - Combines SVM, KNN, and Logistic Regression
  - Soft voting for robust predictions
  
- **Support Vector Machine (SVM)** - 90.5% Accuracy
- **K-Nearest Neighbors (KNN)** - 79.3% Accuracy
- **Logistic Regression** - 77.9% Accuracy
- **XGBoost** - 85.1% Accuracy

**Deep Learning:**
- **1D Convolutional Neural Network (CNN)** - TensorFlow/Keras
  - Conv1D layers (64→128→256 filters)
  - MaxPooling + Batch Normalization + Dropout
  - 4-class classification with Softmax output

**Feature Engineering:**
- PCA Dimensionality Reduction: 3,060 features → 400 components (99% variance retained)
- Gaussian Filtering (σ=1) for noise reduction
- Otsu Adaptive Thresholding for binary conversion
- Contour detection and 1D signal extraction

### Hardware & Software Requirements:

**Hardware:**
- Minimum: CPU (4GB RAM)
- Recommended: GPU with 8GB+ VRAM
- Production: 2 CPU cores + 2GB RAM (Docker)

**Software Stack:**
- Python 3.7-3.8
- TensorFlow/Keras (Deep Learning)
- scikit-learn (ML algorithms)
- Streamlit (Web Interface)
- OpenCV/scikit-image (Image Processing)
- Git LFS (Large File Management)

### Architecture:

The project follows a modular architecture:

1. **Image Processing Layer** (Ecg.py)
   - 13-lead ECG segmentation
   - Preprocessing (blur, thresholding)
   - Signal extraction from images

2. **Feature Engineering Layer**
   - Signal combining (3,060 features)
   - PCA transformation (400 components)

3. **Inference Engine**
   - Pre-trained ensemble classifier
   - Multi-class probability output

4. **User Interface**
   - Streamlit web application
   - Real-time ECG analysis
   - Confidence visualization

### Results & Performance:

| Metric | Value |
|--------|-------|
| Best Model | Ensemble Voting |
| Accuracy | 92.8% |
| Training Samples | 929 |
| Test Accuracy | Validated |
| Deployment | Production-Ready |

### Docker Deployment:

Build and run using Docker:
```bash
docker build -t myocardia .
docker run -p 8080:8080 myocardia
```

### License & Attribution:

Original concept by: Gummadavelli Sandeep
Enhanced & Maintained by: V-JasonEmmanuel
Dataset Source: [Mendeley ECG Dataset](https://data.mendeley.com/datasets/gwbz3fsgp8/2)

### Contributing:

Feel free to fork, submit issues, and create pull requests. All contributions are welcome!
