# 🎬 IMDb Movie Review Sentiment Analysis

A Natural Language Processing (NLP) project that classifies IMDb movie reviews as **Positive** or **Negative** using a fine-tuned **DistilBERT** transformer model.

## 📌 Project Overview

The goal of this project is to build a sentiment classification system that understands the meaning and context of movie reviews.

Instead of using traditional NLP techniques such as Bag-of-Words or TF-IDF, this project uses **DistilBERT**, a pretrained Transformer model, and fine-tunes it on the IMDb movie review dataset.

The trained model can predict the sentiment of a new movie review and provide a confidence score.

---

## 🚀 Features

- 🧹 Basic text cleaning
- 🔤 DistilBERT tokenization
- 🤗 Pretrained DistilBERT model
- 🧠 Fine-tuning for binary sentiment classification
- 📊 Validation and test evaluation
- 📈 Classification report
- 🔲 Confusion matrix
- 🎯 Sentiment prediction with confidence score
- 💾 Saved model and tokenizer
- 🌐 Streamlit application
- 🤗 Model hosted on Hugging Face

---

## 📂 Dataset

The project uses the **IMDb Movie Review Dataset**.

The original dataset contains:

- **50,000 movie reviews**
- **25,000 Positive reviews**
- **25,000 Negative reviews**

After removing duplicate reviews:

- **49,582 reviews** remained

### Dataset Columns

| Column | Description |
|---|---|
| `review` | Movie review text |
| `sentiment` | Positive or Negative |

The dataset was divided using a stratified split:

| Dataset | Samples |
|---|---:|
| Training | 39,665 |
| Validation | 4,958 |
| Test | 4,959 |

---

## 🧹 Text Preprocessing

Only light preprocessing was performed because Transformer models such as DistilBERT benefit from retaining the original language structure.

The preprocessing included:

- Removing HTML tags
- Removing extra whitespace
- Removing duplicate reviews
- Converting sentiment labels into numerical values

The following traditional preprocessing techniques were **not** applied:

- Stopword removal
- Stemming
- Lemmatization
- Punctuation removal

This allows DistilBERT to use the original context of the review.

---

## 🤖 Model

The project uses:

**DistilBERT (`distilbert-base-uncased`)**

DistilBERT is a smaller and faster version of BERT that retains much of BERT's language understanding capability.

The pretrained DistilBERT model was fine-tuned for a **binary classification task**:

```text
0 → Negative
1 → Positive


## 🚀 Live Demo

[https://imdb-sentiment-distilbert.streamlit.app/](https://imdb-sentiment-distilbert.streamlit.app/)
