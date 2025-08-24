# Gait Condition Classification using LSTMs

This project demonstrates the use of a Long Short-Term Memory (LSTM) neural network to classify human walking conditions based on multivariate time-series data of joint angles. The model predicts whether a person is walking normally (unbraced), with a knee brace, or with an ankle brace with high accuracy.

This is a portfolio project designed to showcase a complete machine learning workflow for time-series classification, from data exploration to model deployment.

<p align="center">Figure 1: Visualization of the average joint angle patterns for each walking condition, revealing distinct profiles that the model learns to classify.</p>

---

## Table of Contents
- Project Overview  
- Dataset  
- Methodology  
- Results  
- How to Run  
- Dependencies  

---

## Project Overview
The primary goal of this project is to build and evaluate a deep learning model capable of distinguishing between three different gait conditions using biomechanical data. This is a classic example of a multivariate time-series classification problem, with applications in clinical biomechanics, rehabilitation monitoring, and sports science.

The project follows these key steps:

1. **Data Loading and Exploration**: Fetching the dataset and performing Exploratory Data Analysis (EDA) to understand its structure and uncover patterns.  
2. **Data Preprocessing**: Transforming the raw data from a long format into sequences suitable for an LSTM model. This step also includes feature scaling and label encoding.  
3. **Model Building**: Designing and implementing an LSTM-based neural network architecture using TensorFlow/Keras.  
4. **Training and Evaluation**: Training the model on the preprocessed data and rigorously evaluating its performance using metrics like accuracy, classification report, and a confusion matrix.  

---

## Dataset
The project uses the **Multivariate Gait Data dataset** from the UCI Machine Learning Repository.  

**Source**: [UCI ML Repo: Multivariate Gait Data](https://archive.ics.uci.edu/ml/datasets)  

**Characteristics**:
- **Instances**: 181,800  
- **Features**: 7 (subject, condition, replication, leg, joint, time, angle)  
- **Abstract**: The dataset contains bilateral joint angle time series (ankle, knee, hip) from 10 healthy subjects under 3 walking conditions: unbraced, knee braced, and ankle braced. Each gait cycle is normalized to 101 time points.  

---

## Methodology

### 1. Data Preprocessing & Sequencing
The raw data is in a "long" format, where each row represents a single time point for a single joint. To prepare it for the LSTM model, the data was reshaped into sequences with the following structure:  

- **Samples**: 600 unique gait cycles (10 subjects × 3 conditions × 10 replications × 2 legs).  
- **Timesteps**: 101 data points, representing 0–100% of a gait cycle.  
- **Features**: 3 joint angles (ankle, knee, hip) measured at each timestep.  

The features were scaled using `StandardScaler` to normalize the data, which is crucial for the performance of neural networks.  

### 2. Model Architecture
A Sequential model was built using `tensorflow.keras` with the following layers:  

- **Input Layer**: Defined to accept sequences of shape (101, 3).  
- **LSTM Layer**: 64 units to learn temporal dependencies in the data.  
- **Dropout Layers**: Two dropout layers with a rate of 0.3 for regularization.  
- **Dense Layer**: 32 units with ReLU activation.  
- **Output Layer**: 3 units (one for each class) with softmax activation.  

The model was compiled with the **Adam** optimizer and **categorical_crossentropy** loss function.  

---

## Results
The model achieved excellent performance on the unseen test set, demonstrating its ability to accurately classify the walking conditions.  

- **Test Accuracy**: 99.17%  

The classification report shows high precision, recall, and F1-scores for all three classes, indicating a well-balanced and robust model.  

**Confusion Matrix**:  
The model made only one misclassification on the test set of 120 samples, incorrectly identifying one *Knee Brace* sample as *Unbraced*.  

These results confirm that the LSTM model successfully learned the distinct temporal patterns associated with each walking condition.  

---

## How to Run
The entire project is contained within a single Google Colab notebook (`Gait_Condition_Classification.ipynb`).  

1. **Open in Google Colab**: Upload the `.ipynb` file to your Google Drive and open it with Google Colab.  
2. **Install Dependencies**: The first code cell in the notebook installs the required `ucimlrepo` library. All other dependencies are pre-installed in the Colab environment.  
3. **Run All Cells**: Click `Runtime > Run all` in the Colab menu. The notebook will automatically fetch the data, preprocess it, build the model, train it, and display the results.  

---

## Dependencies
- Python 3.x  
- ucimlrepo  
- pandas  
- numpy  
- scikit-learn  
- tensorflow  
- matplotlib  
- seaborn  

All required libraries can be installed via `pip`. The notebook is configured to handle the installation automatically.  
