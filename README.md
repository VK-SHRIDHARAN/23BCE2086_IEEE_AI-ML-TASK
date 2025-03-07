# 23BCE2086_IEEE_AI-ML-TASK
# Fashion MNIST Classification

## Project Overview

This project focuses on classifying fashion items using both traditional machine learning and deep learning approaches. The models are trained on a given dataset of grayscale images representing various clothing items. The goal is to compare the performance of Logistic Regression and Neural Networks in image classification.

## Features Implemented

- **Data Loading & Inspection**: Load and visualize the given dataset.
- **Exploratory Data Analysis (EDA)**: Statistical analysis and category-wise image visualization.
- **Logistic Regression Model**: A basic classifier to predict clothing items.
- **Neural Network Model**: A deep learning model for improved classification.
- **Explainability**: Interpretation of model decisions and feature importance.

## Technologies Used

- **Python**
- **Google Colab**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas**
- **Matplotlib**
- **NumPy**

## Setup Instructions

### 1. Upload Dataset

GitHub restricts large files, so manually upload the dataset to Google Colab under the `/content/` directory before running the notebook.

**Note:** If the dataset is not available, the code will not execute properly. Ensure the dataset is uploaded before proceeding. This dataset is crucial for training and evaluation as it contains structured fashion images specifically curated for classification tasks.

### 2. Run the Notebook

Open Google Colab and execute the provided notebook sequentially.

### 3. Install Dependencies

Ensure required libraries are installed using:

```sh
pip install tensorflow scikit-learn pandas matplotlib numpy
```

## Implementation Details

### Levels Completed

✅ Level 0: Data loading and visualization\
✅ Level 1: EDA and statistical analysis\
✅ Level 2: Logistic Regression classifier\
✅ Level 3: Neural Network implementation

### 1️⃣ Data Loading & Initial Inspection

- The dataset is read using **pandas**.
- Image samples are displayed using **matplotlib** to ensure correct loading.
- Pixel values are normalized by dividing by 255.

### 2️⃣ Exploratory Data Analysis (EDA)

- Dataset statistics such as class distribution are examined.
- Sample images from each category are visualized.
- A bar chart represents the frequency of each clothing category.

## Algorithm Overview

### Logistic Regression Classifier

**Mathematical Representation:**
The logistic regression model is defined as:

$$
P(y=1|X) = \frac{1}{1 + e^{-\theta^T X}}
$$

where:

- \(X\) is the input feature vector (flattened image pixels).
- \(\theta\) is the weight vector.
- \(e\) is the natural exponential function.

The model is trained using **Gradient Descent**, where weights are updated as:

$$
\theta = \theta - \alpha \nabla J(\theta)
$$

where \(\alpha\) is the learning rate, and \(J(\theta)\) is the cost function based on **cross-entropy loss**:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(h_{\theta}(x_i)) + (1 - y_i) \log(1 - h_{\theta}(x_i))]
$$

- **Train-Test Split**: Dataset is divided into 80% training and 20% testing.
- **Model Training**: Uses `LogisticRegression` from `sklearn`.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-score.

### Neural Network Model

**Architecture:**

- **Input Layer**: 784 neurons (flattened 28x28 grayscale image)
- **Hidden Layers**:
  - 128 neurons with ReLU activation
  - 64 neurons with ReLU activation
  - Dropout layers to prevent overfitting
- **Output Layer**: 10 neurons with softmax activation for multi-class classification

**Mathematical Representation:**
Each hidden layer applies the following transformation:

$$
Z^{(l)} = W^{(l)} A^{(l-1)} + b^{(l)}
$$

where:

- \(W^{(l)}\) and \(b^{(l)}\) are the weight and bias for layer \(l\).
- \(A^{(l-1)}\) is the activation from the previous layer.
- **Activation Function (ReLU):**
  $$
  f(x) = \max(0, x)
  $$
- **Output Layer (Softmax):**
  $$
  P(y=k|X) = \frac{e^{z_k}}{\sum_{j=1}^{10} e^{z_j}}
  $$

**Loss Function:**
The model is optimized using the **categorical cross-entropy** loss function:

$$
L = -\sum_{i=1}^{m} \sum_{j=1}^{10} y_{ij} \log(\hat{y}_{ij})
$$

where \(y_{ij}\) is the true label and \(\hat{y}_{ij}\) is the predicted probability.

**Training Details:**

- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Cross-Entropy**
- Epochs: **20**
- Evaluation on the test dataset

## Evaluation & Predictions

- The logistic regression model achieves **\~83% accuracy**.
- The neural network model achieves **\~90% accuracy**.
- Custom image predictions can be made by reshaping and normalizing the input images.

## Future Improvements

- Implement **CNNs** for improved accuracy.
- Experiment with **data augmentation** techniques.
- Train with **a larger dataset** for better generalization.




##

