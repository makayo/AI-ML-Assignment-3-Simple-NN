MNIST Digit Classification with a Simple Neural Network
Author: M Machine Learning_Neural Networks, Data Preprocessing, and Model Evaluation

Environment
Python 3.11.3 TensorFlow 2.13.0 NumPy 1.26.0 Matplotlib 3.8.0

Project Overview
This project demonstrates a complete deep learning workflow using a simple feedforward neural network to classify handwritten digits from the MNIST dataset. It covers data inspection, preprocessing, model architecture, training, evaluation, and prediction — all implemented in a Jupyter Notebook.

Dataset
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels. Key components used:

x_train: 60,000 training images

y_train: Corresponding digit labels (0–9)

x_test: 10,000 test images

y_test: Test labels for evaluation

Workflow Summary

1. Data Inspection and Preparation

Loaded MNIST dataset using TensorFlow

Verified image shape and label distribution

Inspected raw pixel values and label formats

2. Data Preprocessing

Normalized pixel values from 0–255 to [0, 1]

Flattened 28×28 images into 784-length vectors

One-hot encoded labels for multi-class classification

3. Visualization

Displayed original, normalized, and flattened images

Visualized one-hot encoded label as a bar chart

Explained differences between formats:

Raw: 2D grayscale image with pixel values 0–255

Normalized: Same image scaled to [0, 1]

Flattened: Reshaped into a 784-length vector

One-hot encoded: Label converted to a binary vector for classification

4. Model Architecture

Input layer: 784 features

Hidden layers: 128 and 64 neurons with ReLU activation

Output layer: 10 neurons with Softmax activation

Compiled with Adam optimizer and categorical crossentropy loss

5. Model Training

Trained for 5 epochs with batch size of 32

Used 20% of training data for validation

Monitored accuracy and loss during training

6. Evaluation

Final test accuracy: 97.28%

Evaluated model on unseen test data

Plotted accuracy and loss curves to assess performance

7. Prediction

Selected a random test image

Compared true label vs predicted label

Displayed image with prediction result

8. Visual Prediction Test

Used Matplotlib to show the digit image

Annotated with true and predicted labels

Confirmed model’s ability to generalize to unseen data

Results
The model achieved 97.28% accuracy on the test set, demonstrating strong generalization.
Visual inspection of predictions confirms the model’s ability to correctly classify unseen digits.
Training and validation curves show stable learning with no major signs of over-fitting.
