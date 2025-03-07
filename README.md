# 23BCE2086_IEEE_AI-ML-TASK
# **Fashion MNIST Classification using Logistic Regression and Neural Networks**

## **📌 Project Overview**
This project focuses on **classifying images from the Fashion MNIST dataset**, which contains **70,000 grayscale images (28x28 pixels)**, categorized into **10 different clothing items**. The classification task is approached using **two different models**:
- **Logistic Regression** – A simple linear model for baseline classification.
- **Neural Network (MLP)** – A multi-layer perceptron implemented using TensorFlow/Keras.

### **Key Features**
- **Data Loading & Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Logistic Regression Classifier**
- **Neural Network Model (Multi-layer Perceptron)**
- **Performance Evaluation (Accuracy, Loss, Classification Report)**
- **Explainable AI (Model Interpretation)**

---

## **📁 Features Implemented**
✔ **Data Handling** – Load, inspect, and visualize dataset images.  
✔ **Exploratory Data Analysis (EDA)** – Statistical analysis & visualization.  
✔ **Logistic Regression Classifier** – Train & evaluate a simple classifier.  
✔ **Neural Network Model** – Implement a multi-layer perceptron using TensorFlow/Keras.  
✔ **Performance Metrics** – Accuracy, loss analysis, and classification reports.  
✔ **Explainable AI** – Understand model decisions using feature importance analysis.  

---

## **🛠️ Technologies Used**
- **Python 3.x**  
- **Google Colab** *(For development & execution)*  
- **NumPy, Pandas** *(Data processing)*  
- **Matplotlib, Seaborn** *(Data visualization)*  
- **Scikit-learn** *(Logistic Regression, Data Preprocessing)*  
- **TensorFlow/Keras** *(Neural Networks)*  
- **Git & GitHub** *(Version control)*  

---

## **⚙️ Setup Instructions**
### **🔹 Running the Project in Google Colab**
1. Open [Google Colab](https://colab.research.google.com/).  
2. Upload the dataset manually in Colab (Since GitHub doesn’t support large files).  
3. Clone this repository into Colab:
   ```sh
   !git clone https://github.com/VK-SHRIDHARAN/23BCE2086_IEEE_AI-ML-TASK.git
Open and run the fashion_mnist_analysis.ipynb notebook.
🔹 Running Locally (Optional)
Install dependencies:
sh
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
Clone the repository:
sh
Copy
Edit
git clone https://github.com/VK-SHRIDHARAN/23BCE2086_IEEE_AI-ML-TASK.git
cd 23BCE2086_IEEE_AI-ML-TASK
Run the notebook in Jupyter Notebook or VS Code.
📌 Approach & Implementation
This project is divided into four key levels, each progressively building on the previous steps:

1️⃣ Data Loading & Preprocessing
The dataset is loaded from a CSV file containing flattened 28x28 pixel grayscale images.
Each image is reshaped to 28x28 pixels for visualization.
Pixel values are normalized (0 to 1 range) to improve model performance.
The dataset is split into training (80%) and testing (20%) subsets.
2️⃣ Exploratory Data Analysis (EDA)
Sample images from each class are displayed to understand the dataset.
A class distribution plot is generated to check the balance of categories.
Summary statistics of pixel values are computed.
3️⃣ Logistic Regression Model
Algorithm Used: Logistic Regression (Multinomial)
Input: Flattened image vectors (28x28 → 784 pixels).
Optimization Method: L-BFGS solver (efficient for multi-class classification).
Training: Model is trained using labeled images.
Evaluation: Accuracy, classification report, and confusion matrix.
4️⃣ Neural Network Model (MLP - Multi-layer Perceptron)
Algorithm Used: Fully Connected Neural Network (MLP).
Architecture:
Input Layer: 784 neurons (flattened image)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 64 neurons (ReLU activation)
Output Layer: 10 neurons (Softmax activation)
Loss Function: Categorical Crossentropy (since it's a multi-class problem).
Optimizer: Adam (efficient adaptive learning rate optimization).
Training: Model trained using 10 epochs, batch size = 32.
Evaluation: Accuracy, loss analysis, and misclassification patterns.
📏 Algorithm Explanation
📌 Logistic Regression (Baseline Model)
Concept: Logistic Regression is a linear classifier that applies the softmax function to compute probabilities for each category.

Given an input X (image pixels), it calculates:
Copy
Edit
P(Y=k | X) = softmax(WX + b)
The model learns weights W and bias b using Gradient Descent.
📌 Neural Network (MLP)
Concept: A deep learning model consisting of fully connected layers.

Activation Functions: ReLU (hidden layers), Softmax (output layer).
Loss Function: Categorical Crossentropy.
Optimization: Adam optimizer for efficient learning.
Training Process:
Forward pass: Computes predicted outputs.
Loss calculation: Measures prediction error.
Backpropagation: Adjusts weights to minimize loss.
Repeat for multiple epochs.
📂 Project Structure
bash
Copy
Edit
📦 23BCE2086_IEEE_AI-ML-TASK
├── 📄 README.md          # Project documentation
├── 📂 dataset/           # Folder where the dataset is uploaded manually
│   ├── data.csv
│   ├── labels.csv
├── 📄 fashion_mnist_analysis.ipynb  # Main notebook for execution
├── 📂 models/           # Saved trained models (optional)
│   ├── logistic_model.pkl
│   ├── nn_model.h5
├── 📂 images/           # Visualization outputs
│   ├── sample_images.png
│   ├── class_distribution.png
📝 Code Quality & Version Control
Code Quality: Clean, modular, well-commented Python scripts.
Version Control: Tracked using Git with structured commits.
GitHub Repository: GitHub Repo Link
🚀 Conclusion
This project successfully demonstrates Fashion MNIST image classification using Logistic Regression and a Neural Network. The models are trained, evaluated, and interpreted to understand their decision-making.

For any queries, feel free to open an issue or contribute to improvements! 🎯

pgsql
Copy
Edit

This is your full **README.md** file. Just copy-paste it into your GitHub repository. Let me know if
