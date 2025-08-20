# Machine Learning Models from Scratch in R  

## Project Overview  
This project explores three machine learning algorithms — **k-nearest neighbors (KNN)**, **decision trees**, and **neural networks** — implemented from scratch in **R** and applied to three types of tasks:  
- **Binary classification**  
- **Multiclass classification**  
- **Regression**  

The custom implementations are systematically compared with models built using well-known R packages:  
- `class` (for KNN)  
- `rpart` (for decision trees)  
- `nnet` (for neural networks)  

The project emphasizes both **model implementation** and **evaluation**, providing insights into how self-written algorithms perform against standard library functions.  

---

## Datasets  
The project uses well-known benchmark datasets from the **UCI Machine Learning Repository**:  

- **Binary classification** → *Wisconsin Breast Cancer Diagnostic* dataset  
  - 569 samples, 30 numerical features  
  - Target: malignant vs. benign tumors  

- **Multiclass classification** → *Glass Identification* dataset  
  - 214 samples, 9 numerical features  
  - 7 glass types (e.g., building windows, containers, headlamps)  

- **Regression** → *Abalone* dataset  
  - 4177 samples, 8 features  
  - Target: age estimation via shell rings  

All datasets are preprocessed for training and validation without missing values.  

---

## Evaluation Metrics  
Different evaluation metrics were implemented to measure model performance across tasks:  

- **Binary classification:** AUC, sensitivity, specificity, classification accuracy  
- **Multiclass classification:** Confusion matrix, macro-averaged F1 score, accuracy  
- **Regression:** MAE, MSE, MAPE  

Metrics are calculated on both **training** and **validation** sets.  

---

## Project Structure  
- **`funkcje.R`** → Contains all custom implementations:  
  - Model evaluation metrics (classification and regression)  
  - Functions for KNN, decision trees, and neural networks  
  - Cross-validation and hyperparameter tuning (`CrossValidTune`)  

- **`Glowny.R`** → Main script coordinating the workflow:  
  - Data loading and preprocessing  
  - Model training, prediction, and evaluation  
  - Visualization of results  
  - Comparison with R’s built-in packages (`class`, `rpart`, `nnet`)  

---

## Key Features  
- End-to-end pipeline: from **data preparation** to **model training**, **evaluation**, and **visualization**  
- Custom **from-scratch implementations** of KNN, decision trees, and neural networks  
- Side-by-side comparison with **R package implementations**  
- Cross-validation and hyperparameter tuning support  
- Modular structure for easy extension and reuse  

---

👉 This repository is a great resource for anyone who wants to:  
- Learn how to implement machine learning algorithms from scratch in R  
- Understand how evaluation metrics differ across binary, multiclass, and regression tasks  
- Compare custom implementations against standard R libraries  
