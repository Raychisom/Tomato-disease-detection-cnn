# Tomato-disease-detection-cnn
Tomato Disease Detection &amp; Pesticide Recommendation System > A deep learning-powered solution designed to identify tomato plant diseases from leaf images and automatically recommend precise, effective pesticide treatments to optimize crop yield.

# 🍅 Disease Detection & Pesticide Recommendation in Tomatoes Using Imaging & Deep Learning

[![Academic Project](https://img.shields.io/badge/Academic-B.Sc.%20Project-blue.svg)](https://github.com/)
[![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange?logo=tensorflow)](https://tensorflow.org)
[![Language: Python](https://img.shields.io/badge/Language-Python%203.12-3776AB?logo=python&logoColor=white)](https://python.org)

An automated agricultural diagnostic pipeline engineered to identify plant pathologies from tomato leaf images and map them to actionable targeted pesticide treatments. Developed as a Bachelor of Science research project in Computer Science at the **Federal University Wukari**.

---

## 📖 Project Overview
*   **Author:** Amadi Benjamin Chisom
*   **Supervisor:** Prof. Asaju La’aro Bolaji
*   **Institution:** Federal University Wukari, Taraba State
*   **Core Problem:** Traditional manual inspection of crops is slow and highly prone to misdiagnosis, leading to improper chemical use or catastrophic yield loss.
*   **Solution:** A supervised deep learning pipeline utilizing a custom **Convolutional Neural Network (CNN)** built to process tomato leaf images, combat training class imbalances, classify plant health statuses, and supply exact crop protection recommendations.

---

## ⚙️ Core Architecture & Methodology

The research methodology adheres to a clean three-tiered execution framework:

```
[ Raw Kaggle Leaf Images ] ──► [ Preprocessing & Scaling ] ──► [ Custom CNN Classifier ] ──► [ Evaluation & Recommendations ]
```

### 1. Data Processing & Augmentation Pipeline
*   **Dataset Sourced:** Sourced via Kaggle (`kaustubh999/tomatoleaf`), containing $11,000$ images ($1,000$ healthy samples, $10,000$ disease cases) at $224 	imes 224$ pixels.
*   **Data Scaling:** Pixel intensity values are scaled uniformly to a floating-point target distribution of $[0, 1]$ by dividing values by $255$ via NumPy and TensorFlow.
*   **Augmentation Strategy:** Implemented using Keras `ImageDataGenerator` to resolve major dataset class imbalance issues. Transformations include horizontal flips, width/height shifts ($0.2$), zoom ($0.2$), shear ($0.2$), and rotations ($20^\circ$).

### 2. Neural Network Hyperparameters
*   **Loss Function:** Categorical Cross-Entropy
*   **Optimizer:** Adam with a learning rate of $0.0001$
*   **Batch Size & Epochs:** Batch size of $32$ across $10$ max epochs
*   **Activation Functions:** Rectified Linear Unit (ReLU) for hidden convolution filters and Softmax for the multi-class output projection.
*   **Overfitting Mitigation:** Batch Normalization layers coupled with an explicit $0.5$ Dropout sequence after conv steps.

---

## 📊 Experimental Results

Following a strict percentage split methodology ($10,000$ samples allocated for network instruction and $1,000$ isolated for validation arrays), the model outputs are characterized below:

*   **Training Accuracy:** $75.6\%$
*   **Validation / Test Accuracy:** $65.2\%$
*   **Inference Performance:** High baseline precision, recall, and balanced F1-scores across all evaluated diagnostic matrices.

---

## 📁 Repository Directory Map

```hltext
├── data/
│   ├── raw/                 # Original 11k tomato leaves from Kaggle
│   └── processed/           # Scaled matrices [0, 1]
├── notebooks/
│   └── notebook.ipynb       # Jupyter training log with Keras image generation loops
├── src/
│   ├── app.py               # Front-end system interface for image uploads
│   └── cnn_model.py         # CNN layer definitions (Conv2D, MaxPool2D, Dropout, Dense)
├── requirements.txt         # Core dependencies: tensorflow, opencv-python, numpy, pandas
└── README.md
```

---

## ⚡ Quick Deployment & Verification

1. **Clone and Install:**
   ```bash
   git clone https://github.com/yourusername/tomato-disease-detection.git
   cd tomato-disease-detection
   pip install -r requirements.txt
   ```

2. **Run Image Inference Routine:**
   ```python
   import tensorflow as tf
   import numpy as np

   # Mathematical prediction map implementation
   # ReLU: z = max(0, i)
   model = tf.keras.models.load_model('models/tomato_disease_classifier.h5')
   img = tf.keras.preprocessing.image.load_img('test_leaf.jpg', target_size=(256, 256))
   tensor = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0) / 255.0

   prediction = model.predict(tensor)
   print(f"Pathology Output Probability Distribution Array: {prediction}")
   ```

---

## 📄 Reference Citation
```bibtex
@thesis{Amadi2025,
  author       = {Amadi Benjamin Chisom},
  title        = {Disease Detection and Pesticide Recommendation in Tomatoes Using Imaging and Deep Learning},
  school       = {Federal University Wukari, Department of Computer Science},
  year         = {2025},
  type         = {B.Sc. Thesis},
  supervisor   = {Prof. Asaju La’aro Bolaji}
}
```
