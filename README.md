# 🫁 Pneumonia Detection Using CNN & VGG16 Transfer Learning

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)
[![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

---

## 📌 Problem

Pneumonia is one of the leading causes of death worldwide, especially among children under 5 and adults over 65. Radiologists diagnose pneumonia by examining chest X-rays, but this process is time-consuming and prone to human error, particularly in high-volume clinical settings.

**The question:** Can deep learning models classify chest X-ray images as **Normal** or **Pneumonia** with high enough accuracy to assist radiologists in faster, more reliable diagnosis?

---

## 🎯 Approach

I built and compared two approaches on the [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia):

### Model 1: Custom CNN (Built from Scratch)

A 5-block convolutional neural network designed to learn directly from grayscale X-ray images:

```
Input (150×150×1)
  → Conv2D(32) → BatchNorm → MaxPool
  → Conv2D(64) → Dropout(0.1) → BatchNorm → MaxPool
  → Conv2D(64) → BatchNorm → MaxPool
  → Conv2D(128) → Dropout(0.2) → BatchNorm → MaxPool
  → Conv2D(256) → Dropout(0.2) → BatchNorm → MaxPool
  → Flatten → Dense(128) → Dropout(0.2) → Dense(1, sigmoid)
```

- **Optimizer:** RMSProp
- **Loss:** Binary Cross-Entropy
- **Augmentation:** ImageDataGenerator (rotation, zoom, shift, flip)
- **Learning Rate:** ReduceLROnPlateau (patience=2, factor=0.3)
- **Epochs:** 15

### Model 2: VGG16 Transfer Learning

Pre-trained VGG16 (ImageNet weights) with custom classification head:

- VGG16 base → Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(2, Softmax)
- **Optimizer:** Adam (lr=0.0001)
- **Epochs:** 20

---

## 📊 Results

| Model | Test Accuracy | Pneumonia Precision | Pneumonia Recall | Pneumonia F1 |
|-------|:------------:|:-------------------:|:----------------:|:------------:|
| **Custom CNN** | 90.76% | 0.99 | 0.88 | 0.93 |
| **VGG16 (Transfer Learning)** | **95.03%** | 0.96 | 0.91 | 0.93 |

### Key Findings

- **VGG16 outperformed the custom CNN by ~4.3%** in overall accuracy, demonstrating the power of transfer learning even on grayscale medical images.
- **Custom CNN had higher precision (0.99)** — when it predicted pneumonia, it was almost always right. But it missed more actual pneumonia cases (recall = 0.88).
- **VGG16 had better balance** — 0.96 precision with 0.91 recall, meaning fewer missed pneumonia cases. In a clinical setting, this balance matters more because a missed diagnosis is more dangerous than a false alarm.
- The custom CNN confusion matrix showed 522 false negatives vs VGG16's more balanced error distribution.

---

## 🗂 Dataset

- **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Total Images:** ~5,856 chest X-ray images
- **Classes:** NORMAL, PNEUMONIA
- **Split:** Train / Validation / Test
- **Class Imbalance:** ~73% Pneumonia, ~27% Normal in training set (addressed via augmentation)

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Pre-trained Model** | VGG16 (ImageNet) |
| **ML Evaluation** | Scikit-learn |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Data Source** | Kaggle (OpenDatasets) |
| **Environment** | Google Colab (GPU) |

---

## 📁 Project Structure

```
Pneumonia-Detection-CNN-VGG16/
│
├── notebooks/
│   └── AI_Project_CNN_VGG_ENSEM_Final.ipynb   # Full training & evaluation notebook
│
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/CHANx0000/Pneumonia-Detection-CNN-VGG16.git
   cd Pneumonia-Detection-CNN-VGG16
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   - Open `notebooks/AI_Project_CNN_VGG_ENSEM_Final.ipynb` in Google Colab or Jupyter
   - The notebook will automatically download the dataset from Kaggle (requires Kaggle API credentials)

---

## 💡 What I Learned

- **Transfer learning works even when domains don't match perfectly.** VGG16 was trained on color ImageNet photos, yet it transferred well to grayscale medical X-rays — the low-level features (edges, textures, patterns) are universal.
- **Precision vs. recall trade-off matters in healthcare.** A model that misses pneumonia (low recall) is more dangerous than one that over-flags it (low precision). VGG16's better recall made it the stronger clinical candidate.
- **Data augmentation is essential with imbalanced medical datasets.** Without augmentation, the model would simply learn to predict the majority class.

---

## 📬 Contact

- **GitHub:** [CHANx0000](https://github.com/CHANx0000)
