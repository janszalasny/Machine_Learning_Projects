# Anachronism Detection using Zero-Shot Classification

## Project Overview
This project explores the use of **Zero-Shot Text Classification** to identify anachronisms in sentences.  
An **anachronism** is an element, typically an object or concept, that is misplaced in a particular historical context.  
The goal is to leverage the implicit knowledge of large, pre-trained transformer models to perform this classification task without any task-specific training or fine-tuning.

The primary tool used is the **Hugging Face `transformers` library**, specifically its **zero-shot-classification pipeline**.  
This demonstrates a modern, efficient approach to solving novel NLP classification problems where labeled training data may be scarce.

---

## Dataset
The project utilizes the **anachronisms** subset of the **tasksource/bigbench** dataset, available on the Hugging Face Hub.

- **Dataset:** `tasksource/bigbench`  
- **Subset:** `anachronisms`

This dataset contains sentences that are either historically consistent or contain one or more anachronistic elements.  
The original labels are:

- `Yes` → *anachronistic*  
- `No` → *not anachronistic*

---

## Methodology
The core of this project is **Zero-Shot Classification**, a technique where a model can classify data into categories it has not explicitly been trained on.  
This is achieved by reframing the classification problem as a **Natural Language Inference (NLI)** task.  
The model determines the likelihood that a given sentence (*premise*) entails a candidate label (*hypothesis*).

### Models Used
- **Baseline Model:** `facebook/bart-large-mnli`  
- **Improved Model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`

### Workflow
1. **Data Loading:** Load dataset from Hugging Face Hub using the `datasets` library.  
2. **Preprocessing:** Extract sentences, map labels (`Yes`/`No`) → (`anachronistic`/`not anachronistic`).  
3. **Inference:** Use zero-shot classification pipeline to predict the most likely category.  
4. **Evaluation:** Assess with precision, recall, F1-score, and confusion matrix.

---

## Setup and Usage

### Prerequisites
- Python **3.8+**  
- Google Colab (recommended) or a local machine with GPU support

### Installation
Install dependencies:

```bash
pip install transformers datasets torch pandas scikit-learn seaborn
