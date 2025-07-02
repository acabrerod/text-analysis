# Text Analysis with R - Bachelor's Thesis in Statistics

This repository contains the code developed for my Bachelor's Thesis in Statistics titled **"Text Analysis with R"**. The project focuses on the application of statistical and machine learning techniques to textual data using **R**.

It includes a practical demonstration of key text mining tasks such as:

1. **Sentiment Analysis**
2. **Text Clustering**
3. **Text Classification**


## 1. Project Overview

The main objectives of this project are:

- To explore statistical methods for analyzing textual data.
- To apply data preprocessing techniques such as tokenization, normalization and text cleaning.
- To apply symbolic and distributed representation methods, including:
    - Bag of Words (BoW), TF-IDF, and feature extraction techniques.
    - Latent Semantic Analysis (LSA), Word2Vec and GloVe

- To apply supervised and unsupervised techniques, covering both text clustering, and text classification.
- To use the [IMDb Movie Reviews dataset](https://ai.stanford.edu/%7Eamaas/data/sentiment/) to demonstrate the application of said techniques.


## 2. Repository structure

- **`data/`** – Contains the data used for this project, the IMDb Movie Reviews dataset.

- **`visualization/`** – Contains code to generate graphical representations for the IMDb Movie Reviews dataset.
  
- **`sentiment-analysis/`** – Contains code and results for sentiment analysis. The following R packages are used: *syuzhet*, *tidytext*, *sentimentr*, and *SentimentAnalysis*.

- **`clustering/`** – Contains code and results for text clustering. Among the techniques applied are the feature extraction methods Entropy-based Ranking, Term Strength and Term Contribution. For clustering validation, the Dunn and Silhouette indices are used.

- **`classification/`** – Contains code and results for text classification.
