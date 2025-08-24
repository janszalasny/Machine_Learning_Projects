# MAGIC Gamma Telescope: Particle Classification

This project demonstrates a complete machine learning workflow to classify high-energy particles detected by the **MAGIC (Major Atmospheric Gamma-ray Imaging Cherenkov)** telescope.  
The goal is to build a model that can accurately distinguish between **gamma rays (g)** and **hadronic showers (h)** based on the image parameters of the light they produce in the atmosphere.

This repository contains a **Google Colab notebook (.ipynb)** that walks through the entire process, from data exploration to model evaluation.

---

## Dataset

The data for this project is the **MAGIC Gamma Telescope Data Set** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope).  
It consists of Monte Carlo generated data to simulate the registration of high-energy particles.

- **Source:** UCI Machine Learning Repository  
- **Instances:** 19,020  
- **Features:** 10 numerical features describing the shape and orientation of the event image  
- **Target:** A binary class label (`g` for gamma, `h` for hadron)  
- **Missing Values:** None  

**Features include:** `fLength`, `fWidth`, `fSize`, `fConc`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`.  
The **target variable** is `class`.

---

## Project Workflow

The project follows a standard machine learning pipeline:

1. **Setup and Data Loading**  
   - The environment is prepared, and the dataset is loaded directly from the UCI repository.

2. **Exploratory Data Analysis (EDA)**  
   - Analyze feature distributions, class balance, and correlations between variables.

3. **Data Preprocessing**  
   - **Label Encoding:** Convert categorical target (`g`, `h`) into numerical format (`1`, `0`).  
   - **Feature Scaling:** Standardize numerical features using `StandardScaler`.

4. **Model Training**  
   Train three different classification models:
   - Logistic Regression (baseline)  
   - Random Forest Classifier  
   - XGBoost Classifier  

5. **Model Evaluation**  
   - Evaluate models on a held-out test set using metrics: **accuracy, precision, recall, F1-score**, and a **confusion matrix**.

