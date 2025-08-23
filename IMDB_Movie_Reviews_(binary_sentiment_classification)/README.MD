# Sentiment Analysis of IMDB Movie Reviews

This project demonstrates an end-to-end workflow for a classic Natural Language Processing (NLP) problem: binary text classification. A neural network is built using TensorFlow and Keras to classify IMDB movie reviews as either positive or negative.

The project is presented in a Google Colab notebook, walking through each step from data loading and cleaning to model training, evaluation, and visualization.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Learnings](#key-learnings)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Project Overview

The goal of this project is to build and train a deep learning model that can accurately predict the sentiment of a movie review. We use the well-known IMDB dataset, which contains 25,000 highly polar movie reviews for training and 25,000 for testing.

The notebook covers:

- Loading data from the Hugging Face datasets library.
- Performing Exploratory Data Analysis (EDA) to understand the data's structure and content.
- Cleaning and preprocessing text data to make it suitable for a neural network.
- Building a sequential model with an Embedding layer to learn word representations.
- Training the model with validation and EarlyStopping to prevent overfitting.
- Evaluating the model's performance on unseen test data.

---

## Key Learnings

This project serves as a practical demonstration of the following machine learning concepts and skills:

- **End-to-End NLP Workflow:** From raw text data to a fully evaluated predictive model.
- **Text Preprocessing:** Techniques for cleaning text data, including removing HTML tags and special characters.
- **Text Vectorization:** Using TensorFlow's `TextVectorization` layer to create a data pipeline that converts raw text into integer sequences.
- **Word Embeddings:** Understanding and implementing Keras Embedding layers to create dense vector representations of words, which capture semantic meaning.
- **Model Building:** Constructing a neural network with `Embedding`, `GlobalAveragePooling1D`, and `Dense` layers for efficient classification.
- **Model Training & Regularization:** Implementing best practices like using a validation set, Dropout, and EarlyStopping callbacks.
- **Performance Evaluation:** Analyzing model performance using metrics like accuracy, precision, recall, F1-score, and visualizing a confusion matrix.

---

## Technologies Used

- Python 3  
- TensorFlow & Keras: For building and training the deep learning model.  
- Hugging Face datasets: To load the IMDB dataset.  
- Scikit-learn: For performance evaluation metrics (`classification_report`, `confusion_matrix`).  
- Pandas: For data manipulation and exploration.  
- NumPy: For numerical operations.  
- Matplotlib & Seaborn: For data visualization.  
- Google Colab: As the development environment.  

---

## How to Run

This project is designed to be run in a cloud environment like Google Colab, requiring no local setup.

**Open in Google Colab:**

1. Navigate to the `.ipynb` notebook file in this repository.
2. Click the "Open in Colab" button at the top of the file.

**Run the Cells:**

1. Once the notebook is open, you can run all the cells sequentially.
2. Go to `Runtime -> Run all` in the top menu.

The necessary libraries will be installed, the data downloaded, and the model will be trained and evaluated.

---

## Project Workflow

The notebook is structured to follow a standard machine learning project lifecycle:

1. **Setup & Imports:** Install and import all necessary libraries.  
2. **Load Data:** Load the `stanfordnlp/imdb` dataset and convert it to Pandas DataFrames.  
3. **Exploratory Data Analysis (EDA):** Inspect the data, check for missing values, and analyze the class distribution.  
4. **Data Preprocessing:**  
   - Define a function to clean the text (remove HTML, punctuation).  
   - Use the `TextVectorization` layer to create a vocabulary and map text to integer sequences.  
5. **Model Building:** Construct a Sequential Keras model with `Embedding`, `GlobalAveragePooling1D`, and `Dense` layers.  
6. **Model Training:** Train the model on the training data while monitoring performance on a validation set.  
7. **Evaluation:** Evaluate the final model on the unseen test set and visualize the results, including training history and a confusion matrix.  

---

## Results

The model achieved a final accuracy of approximately **87%** on the test set.

The training history shows that the model learned effectively without significant overfitting, as the validation accuracy tracked the training accuracy closely before EarlyStopping halted the process.


