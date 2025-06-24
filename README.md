# Streamify Movie Recommendation System - Model Training

This section of the Streamify project focuses on the training and evaluation of the deep learning models used for personalized movie recommendations.

## Overview

Our recommendation system leverages advanced machine learning techniques to understand user preferences and suggest relevant movies. This repository contains the code and resources necessary to reproduce our model training process.

## Models Implemented

We explore and compare the performance of the following models:

1.  *Neural Collaborative Filtering (NCF)*: Our primary proposed model, implemented using PyTorch. This model aims to capture complex non-linear user-item interactions.
2.  *Deep Autoencoder (DAE)*: A benchmark model implemented with Keras/TensorFlow, used to learn compressed representations of user interaction profiles.
3.  *Singular Value Decomposition (SVD)*: A classic matrix factorization benchmark model, implemented using the Surprise library.

## Datasets

All models are trained and evaluated on variations of the [MovieLens dataset](https://grouplens.org/datasets/movielens/), specifically:
* MovieLens 100K Dataset
* MovieLens 1M Dataset

## Evaluation Metrics

Model performance is rigorously assessed using:
* *Training Loss & Test Loss*: To monitor learning progress and generalization.
* *Hit Rate @ 10 (HR@10)*: Measures the proportion of relevant items retrieved in the top 10 recommendations.
* *Normalized Discounted Cumulative Gain @ 10 (NDCG@10)*: Evaluates the ranking quality, assigning higher scores to relevant items at higher ranks.

## Getting Started (Model Training)

To run the model training scripts:

1.  *Environment Setup*: Ensure you have Python 3.7+ and Anaconda (or a virtual environment) installed.
2.  *Install Dependencies*: Navigate to the root of your project and install the required libraries:
    bash
    pip install -r requirements.txt
    
    (Note: The requirements.txt should contain PyTorch, TensorFlow, Surprise, Pandas, NumPy, Matplotlib, etc.)
3.  *Access Jupyter Notebook*: Launch Jupyter Notebook (recommended IDE for this part) through Anaconda Navigator.
4.  *Execute Training Scripts*: Open and run the relevant .ipynb (Jupyter Notebook) files located in the model_implementation/ or scripts/ directory (as per your project structure) to perform data preprocessing, exploratory data analysis, model training, and evaluation. These notebooks will guide you through:
    * Data Unpacking and Loading.
    * Data Cleaning and Splitting (using Leave-One-Out methodology).
    * Exploratory Data Analysis (EDA) visualizations.
    * NCF Model definition, negative sampling, hyperparameter tuning, and training.
    * DAE Model definition and training.
    * SVD Model definition and training.
    * Performance evaluation and comparison of all models.

## Results

Our comparative analysis indicates that the *NCF Model 5* configuration consistently achieves superior performance in terms of both HR@10 and NDCG@10 across different datasets, significantly outperforming DAE and SVD benchmarks. This model's ability to learn complex non-linear user-item relationships proves highly effective for generating high-quality, accurately ranked recommendations.

For detailed performance figures and visualizations, please refer to the evaluation sections within the Jupyter notebooks.

## ✍️ Author

**Jee Peng Chai (79639)**
