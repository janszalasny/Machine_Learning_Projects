# Fine-Tuning a Sentence Transformer for Semantic Similarity

This project demonstrates how to fine-tune a pre-trained Sentence Transformer model (**all-MiniLM-L6-v2**) for a semantic similarity task. By training on a specialized dataset of sentence pairs, we significantly improve the model's ability to generate meaningful embeddings that capture nuanced semantic relationships.

This project is structured as a Google Colab notebook, making it fully reproducible and easy to run.

---

## Project Overview

The core objective is to enhance a base Transformer model's understanding of sentence similarity. We use a **Siamese network architecture** with a **MultipleNegativesRankingLoss** function to teach the model that semantically equivalent sentences should have very close vector representations in the embedding space.

The final fine-tuned model shows a quantifiable improvement over the base model when evaluated on the standard **Semantic Textual Similarity (STS)** benchmark.

---

## Features

- **Efficient Fine-Tuning**: Utilizes the powerful `sentence-transformers` library for a streamlined training process.  
- **Modern Architecture**: Implements a Siamese network to learn from sentence pairs.  
- **Advanced Loss Function**: Employs `MultipleNegativesRankingLoss` to effectively learn sentence embeddings by using in-batch negatives.  
- **Reproducible Environment**: A complete, end-to-end Google Colab notebook is provided.  
- **Clear Evaluation**: Performance is measured against the standard STS benchmark, showing a clear improvement in Spearman correlation.  

---

## Dataset

We use the **embedding-data/sentence-compression** dataset, which is publicly available on the Hugging Face Hub.

- **Content**: The dataset consists of pairs of sentences that are semantically equivalent.  
- **Purpose**: These pairs serve as positive examples for our Siamese network. The model learns to minimize the distance between the embeddings of these sentence pairs.  

**Example Pair:**

- Sentence 1:  
  *"The Andromeda Galaxy, also known as Messier 31, M31, or NGC 224, is a spiral galaxy approximately 2.5 million light-years from Earth in the Andromeda constellation."*  

- Sentence 2:  
  *"The Andromeda Galaxy is a spiral galaxy located about 2.5 million light-years away from our planet."*  

---

## Methodology

1. **Setup**: Install and import the necessary libraries, including `sentence-transformers`, `datasets`, and `torch`. The environment is configured to leverage a GPU for accelerated training.  
2. **Data Preparation**: The training data is loaded and transformed into a list of `InputExample` objects, the required format for the `sentence-transformers` library.  
3. **Model Architecture**: Load the pre-trained model `sentence-transformers/all-MiniLM-L6-v2` as the base. This model is then used within a Siamese network structure, where two sentences are processed in parallel to compare their embeddings.  
4. **Training**: Train the model for one epoch using the `MultipleNegativesRankingLoss` function. This loss is highly effective as it takes a batch of positive pairs `(a_i, p_i)` and, for each anchor sentence `a_i`, treats all other sentences `p_j (j != i)` in the batch as hard negatives. This forces the model to learn fine-grained distinctions.  

---

## Evaluation & Results

We evaluated both the base model and our fine-tuned model on the development set of the **Semantic Textual Similarity Benchmark (STSb)**. The evaluation metric is the **Spearman correlation** between the cosine similarity of the model's embeddings and the human-annotated similarity scores.

| Model                   | Spearman Correlation on STSb-dev |
|--------------------------|----------------------------------|
| all-MiniLM-L6-v2 (Base)  | ~85.26                           |
| fine-tuned-mini-lm (Ours)| ~87.53                           |

Our fine-tuned model achieved a significant improvement, demonstrating its enhanced ability to capture semantic similarity.

