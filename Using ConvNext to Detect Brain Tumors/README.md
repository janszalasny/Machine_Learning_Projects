* * * * *

Brain Scan Classification using Transfer Learning 
====================================================

This project provides a comprehensive Jupyter Notebook for classifying brain MRI scans into multiple categories, including different types of tumors (Glioma, Meningioma, Pituitary), Multiple Sclerosis (MS), and Normal cases. The solution leverages **transfer learning** with a state-of-the-art **ConvNeXt** model, automates **data augmentation**, and performs a **hyperparameter grid search** to find the optimal training configuration.

The entire workflow is designed to be run in a Google Colab environment, utilizing Google Drive for data storage and retrieval.

Features & Key Concepts
-----------------------

This notebook demonstrates a complete end-to-end deep learning pipeline:

-   ** Automated Data Preparation:** Scripts to organize and subset the initial dataset into a clean working directory structure.

-   ** Advanced Data Augmentation:** Programmatically expands the training dataset to a target size by applying a variety of random transformations, including rotations, blurring, noise injection, shearing, and cropping. This helps prevent overfitting and improves model generalization.

-   ** Transfer Learning:** Utilizes the powerful `ConvNeXt-Atto` model, pre-trained on ImageNet. Only the final classification layer is trained initially (layer freezing), allowing for rapid adaptation to the medical imaging task.

-   ** Hyperparameter Tuning:** Implements a **grid search** to systematically test different combinations of learning rates, dropout rates, and batch sizes to identify the most effective model configuration.

-   ** Comprehensive Evaluation & Logging:**

    -   Tracks key metrics like **Accuracy**, **Loss**, and weighted **F1-Score** for both training and validation sets.

    -   Automatically generates and saves plots for accuracy and loss curves after each epoch.

    -   Creates and saves a **confusion matrix** visualization each epoch to monitor classification performance across all classes.

    -   Logs all training metrics to a `.csv` file for easy analysis and comparison across different runs.

Project Workflow
----------------

The notebook is structured to guide you through the entire process from data setup to model training and evaluation.

1.  **Environment Setup:** The notebook begins by checking for GPU (CUDA) availability and mounting your Google Drive to access the dataset and save results.

2.  **Dataset Preparation:** It creates the necessary directory structure in your Google Drive and copies a specified number of images from your source folder into a new, organized dataset for the experiment.

3.  **Data Augmentation:** Before training, each class-specific folder is processed. The script applies random transformations to existing images until a target number of files is reached, ensuring a balanced and larger dataset.

4.  **Model Configuration:** A `ConvNeXt-Atto` model is loaded from the `timm` library. The feature extraction layers are frozen (`requires_grad = False`), and only the final classification head is left trainable.

5.  **Grid Search & Training Loop:**

    -   The notebook iterates through every combination of predefined hyperparameters (learning rate, dropout, batch size).

    -   For each combination, it splits the data, creates `DataLoaders`, and initializes a fresh model.

    -   The model is trained for a set number of epochs. Inside the loop, it performs forward/backward passes, calculates loss, and updates weights.

6.  **Live Evaluation & Artifacts:** After each epoch, the model's performance is measured on the validation set. All metrics are logged to a CSV file, and performance plots (accuracy, loss, confusion matrix) are saved to a unique directory corresponding to the hyperparameter set being tested.

Getting Started
---------------

To run this notebook on your own data, follow these steps.

### Prerequisites

-   A Google Account with Google Drive storage.

-   The project is designed for **Google Colab**.

-   Required Python libraries: `torch`, `torchvision`, `timm`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`, `opencv-python`, `Pillow`.

### 1\. Prepare Your Data

1.  Create a base folder in your Google Drive (e.g., `DL`).

2.  Inside this folder, create a source folder (e.g., `MRI_scans`).

3.  Organize your images into subfolders within the source folder, where each subfolder name corresponds to a class. For example:

    ```
    /content/drive/MyDrive/DL/MRI_scans/
    ├── BT_glioma/
    │   ├── image1.jpg
    │   └── ...
    ├── BT_meningioma/
    ├── MS/
    └── Normal/

    ```

### 2\. Set Up the Notebook

1.  Upload the notebook to your Google Colab environment.

2.  Install the `timm` library, as it's not included in Colab by default:

    Python

    ```
    !pip install timm

    ```

3.  Modify the path variables at the beginning of the relevant cells to match your Google Drive folder structure (e.g., `base_path`, `data_path`).

### 3\. Run the Analysis

1.  Run the cells sequentially.

2.  The notebook will first set up folders, then perform data augmentation, and finally begin the grid search and training process.

3.  After the training completes, you will find new folders in your `base_path` directory. Each folder will be named after the hyperparameters used (e.g., `lr_0.001_drop_0.3_bs_16`) and will contain the training metrics CSV file and all the generated plots.
