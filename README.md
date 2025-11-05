# MNIST Digit Classification with Simple Neural Network  
**Author:** MARK YOSINAO  
**Machine Learning** — Neural Networks, Data Preprocessing, and Model Evaluation

---

## Environment

- Python 3.11.3  
- TensorFlow 2.13.0  
- NumPy 1.26.0  
- Matplotlib 3.8.0  

---

## Project Overview

This project applies the principles of the machine learning lifecycle to classify handwritten digits using a simple feedforward neural network. The goal is to demonstrate a complete deep learning workflow: data inspection, preprocessing, model design, training, evaluation, and prediction — all implemented in a Jupyter Notebook using the MNIST dataset.

---

## Dataset

The MNIST dataset contains grayscale images of handwritten digits (0–9), each sized 28×28 pixels. Key components used:

- `x_train`: 60,000 training images  
- `y_train`: Corresponding digit labels  
- `x_test`: 10,000 test images  
- `y_test`: Labels for evaluation  

---

## Workflow Summary

### 1. Data Inspection  
- Loaded MNIST dataset using TensorFlow  
- Verified image shapes and label distribution  
- Inspected raw pixel values and label formats  

### 2. Data Preprocessing  
- Normalized pixel values from 0–255 to [0, 1]  
- Flattened 28×28 images into 784-length vectors  
- One-hot encoded labels for multi-class classification  

### 3. Visualization  
- Displayed original, normalized, and flattened images  
- Visualized one-hot encoded labels as bar charts  
- Explained format transitions: raw, normalized, flattened, and one-hot encoded  

### 4. Model Training  
- Defined a neural network with two hidden layers (128 and 64 neurons, ReLU)  
- Output layer: 10 neurons with Softmax activation  
- Compiled with Adam optimizer and categorical crossentropy loss  
- Trained for 5 epochs with batch size of 32  
- Used 20% of training data for validation  

### 5. Evaluation  
- Final test accuracy: 97.28%  
- Evaluated model on unseen test data  
- Plotted accuracy and loss curves to assess performance  

### 6. Custom Predictions  
- Selected a random test image  
- Compared true label vs predicted label  
- Displayed image with prediction result  

---

## Results

- The model achieved 97.28% accuracy on the test set  
- Visual inspection confirms correct classification of unseen digits  
- Training and validation curves show stable learning with no major signs of overfitting  
