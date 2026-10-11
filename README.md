# 🧠 AI Testing & PyTorch Foundations

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-2980B9?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![Open In Colab (Perceptron)](https://img.shields.io/badge/Colab-Perceptron-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aziz-belloumi/AI_testing/blob/main/perceptron.ipynb)
[![Open In Colab (CNN)](https://img.shields.io/badge/Colab-CNN-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aziz-belloumi/AI_testing/blob/main/convolutionalNeuralNetwork.ipynb)

A hands-on collection of deep learning implementations and experiments built with **PyTorch**. This repository explores the core pillars of computer vision and neural networks: dataset preparation, fully connected networks (Multi-Layer Perceptron), and Convolutional Neural Networks (CNN) for image classification.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Modules & Experiments](#-modules--experiments)
  - [1. Data Pipeline & Custom Datasets (`dataLoader.py`)](#1-data-pipeline--custom-datasets-dataloaderpy)
  - [2. Multi-Layer Perceptron (`perceptron.ipynb`)](#2-multi-layer-perceptron-perceptronipynb)
  - [3. Convolutional Neural Network (`convolutionalNeuralNetwork.ipynb`)](#3-convolutional-neural-network-convolutionalneuralnetworkipynb)
- [Performance & Benchmark Summary](#-performance--benchmark-summary)
- [Prerequisites & Installation](#-prerequisites--installation)
- [How to Run](#-how-to-run)
  - [Running the Data Loader Script](#running-the-data-loader-script)
  - [Running the Jupyter Notebooks](#running-the-jupyter-notebooks)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🔭 Overview

The goal of this repository is to demonstrate foundational machine learning workflows in PyTorch:
- Building robust data pipelines with `torch.utils.data.Dataset` and `DataLoader`.
- Training and evaluating a Multi-Layer Perceptron (MLP) on grayscale fashion items (**FashionMNIST**).
- Designing, training, and benchmarking a 2D Convolutional Neural Network (LeNet-inspired architecture) on complex color images (**CIFAR-10**).
- Monitoring training convergence with cross-entropy loss and evaluating test set accuracy both globally and per-class.

---

## 📁 Project Structure

```text
AI_testing/
├── dataLoader.py                     # Dataset loading, custom dataset definition & visualization
├── perceptron.ipynb                  # Multi-layer Perceptron (MLP) trained on FashionMNIST
├── convolutionalNeuralNetwork.ipynb  # 2D CNN (ConvNet) trained and evaluated on CIFAR-10
└── README.md                         # Project documentation and guide
```

---

## 🔬 Modules & Experiments

### 1. Data Pipeline & Custom Datasets (`dataLoader.py`)

This script illustrates how PyTorch handles data ingestion, transformation, and batching:

- **Built-in Dataset Ingestion**: Downloads and caches the **FashionMNIST** dataset.
- **Normalization**: Applies `transforms.ToTensor()` to map pixel values from $[0, 255]$ to floating-point tensors in the range $[0.0, 1.0]$.
- **Data Inspection**: Samples 9 random items and renders them in a $3 \times 3$ grid using `matplotlib`, displaying human-readable class names (*T-Shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot*).
- **Custom Dataset Pattern**: Implements a boilerplate `CustomImageDataset` subclassing `torch.utils.data.Dataset` (`__init__`, `__len__`, `__getitem__`) for loading CSV-annotated image collections.
- **Batched Iteration**: Wraps datasets inside `DataLoader(batch_size=64, shuffle=True)` for efficient mini-batch generation during training.

```python
# Batch generation example:
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
# Feature shape: [64, 1, 28, 28] | Labels shape: [64]
```

---

### 2. Multi-Layer Perceptron (`perceptron.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aziz-belloumi/AI_testing/blob/main/perceptron.ipynb)

A fully connected deep neural network trained to classify 10 apparel categories in **FashionMNIST**.

#### Network Architecture

```
Input (1 x 28 x 28)
       │
   [Flatten] ──> (784 features)
       │
   [Linear (784 ──> 512)] + [ReLU]
       │
   [Linear (512 ──> 512)] + [ReLU]
       │
   [Linear (512 ──> 10)] ──> Logits Output
```

#### Training Configuration & Results

- **Device**: Automatic device selection (`cuda` / `mps` / `cpu`).
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.SGD(lr=0.001, momentum=0.9)`
- **Epochs**: `5`
- **Batch Size**: `64`
- **Training Progression**:
  - Epoch 1 Loss: `0.4652`
  - Epoch 2 Loss: `0.4483`
  - Epoch 3 Loss: `0.4361`
  - Epoch 4 Loss: `0.4250`
  - Epoch 5 Loss: `0.4149`
- **Evaluation**: Evaluated on the 10,000-sample test set.
  - **Test Accuracy: 84.23%**

---

### 3. Convolutional Neural Network (`convolutionalNeuralNetwork.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aziz-belloumi/AI_testing/blob/main/convolutionalNeuralNetwork.ipynb)

A classical 2D Convolutional Neural Network trained on the **CIFAR-10** benchmark dataset (10 natural image classes: *plane, car, bird, cat, deer, dog, frog, horse, ship, truck*).

#### Preprocessing
- Converts images to tensors and normalizes RGB channels to mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`.

#### Network Architecture

```
Input Image (3 x 32 x 32)
       │
   [Conv2d (3 ──> 6, kernel=5x5)] + [ReLU]
       │
   [MaxPool2d (2x2, stride=2)] ──> (6 x 14 x 14)
       │
   [Conv2d (6 ──> 16, kernel=5x5)] + [ReLU]
       │
   [MaxPool2d (2x2, stride=2)] ──> (16 x 5 x 5 = 400 features)
       │
   [Flatten]
       │
   [Linear (400 ──> 120)] + [ReLU]
       │
   [Linear (120 ──> 84)] + [ReLU]
       │
   [Linear (84 ──> 10)] ──> Class Logits
```

#### Training Configuration & Results

- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.SGD(lr=0.001, momentum=0.9)`
- **Epochs**: `2`
- **Batch Size**: `4`
- **Mini-batch Loss Convergence**:
  - Epoch 1: `2.221` $\rightarrow$ `1.514`
  - Epoch 2: `1.437` $\rightarrow$ `1.313`
- **Overall Test Accuracy**: **54.6%** (after 2 baseline epochs)
- **Per-Class Accuracy Breakdown**:

| Class | Accuracy | Class | Accuracy |
|:------|:--------:|:------|:--------:|
| 🚗 **Car** | **83.4%** | 🚢 **Ship** | **71.3%** |
| 🐸 **Frog** | **64.0%** | ✈️ **Plane** | **60.8%** |
| 🐴 **Horse** | **59.0%** | 🦌 **Deer** | **52.1%** |
| 🐶 **Dog** | **47.8%** | 🚚 **Truck** | **47.6%** |
| 🐱 **Cat** | **42.6%** | 🐦 **Bird** | **17.4%** |

---

## 📊 Performance & Benchmark Summary

| Model | Architecture | Dataset | Input Dimension | Optimizer | Epochs | Test Accuracy |
|:------|:-------------|:--------|:----------------|:----------|:------:|:-------------:|
| **MLP** | Fully Connected (3 Linear layers) | FashionMNIST | $1 \times 28 \times 28$ | SGD ($lr=0.001$, $mom=0.9$) | 5 | **84.23%** |
| **CNN** | ConvNet (2 Conv + 3 Linear layers) | CIFAR-10 | $3 \times 32 \times 32$ | SGD ($lr=0.001$, $mom=0.9$) | 2 | **54.60%** |

---

## ⚙️ Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/aziz-belloumi/AI_testing.git
cd AI_testing
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision matplotlib pandas numpy jupyter
```

*(Note: For GPU acceleration with CUDA, follow the official installation guide at [pytorch.org](https://pytorch.org/get-started/locally/).)*

---

## 🚀 How to Run

### Running the Data Loader Script
Executes data download, visual inspection, and batch generation:
```bash
python dataLoader.py
```

### Running the Jupyter Notebooks
Start the local Jupyter notebook server:
```bash
jupyter notebook
```
Then open either:
- `perceptron.ipynb`
- `convolutionalNeuralNetwork.ipynb`

Or run them directly in your browser using **Google Colab** via the badges at the top of each notebook.

---

## 💡 Future Enhancements

- [ ] Add data augmentation transforms (`RandomHorizontalFlip`, `RandomCrop`, `ColorJitter`).
- [ ] Incorporate `BatchNorm2d` and `Dropout` layers to regularize the CNN and prevent overfitting.
- [ ] Implement modern architectures (ResNet-18, VGG, MobileNet) via `torchvision.models`.
- [ ] Add learning rate scheduling (e.g., `CosineAnnealingLR` or `ReduceLROnPlateau`).
- [ ] Introduce a unified training and validation loop with TensorBoard / Weights & Biases logging.

---

## 👤 Author

Developed by **[Aziz Belloumi](https://github.com/aziz-belloumi)**.
Feel free to open issues or submit pull requests for suggestions and enhancements!
