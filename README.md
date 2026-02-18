# IMDB Movie Review Sentiment Analysis

A deep learning project that classifies IMDB movie reviews as **positive** or **negative** using a Simple RNN (Recurrent Neural Network). Includes a Streamlit web application for interactive predictions.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Notebooks](#notebooks)

## Overview

This project builds a sentiment analysis model trained on the IMDB movie review dataset (50,000 reviews). It uses a SimpleRNN architecture with word embeddings to process sequential text data and predict whether a review expresses positive or negative sentiment. A Streamlit web app provides an interactive interface for real-time predictions.

## Project Structure

```
movie-review-imdb/
├── app.py                  # Streamlit web application
├── simple_rnn_imdb.h5      # Pre-trained model weights
├── simplernn.ipynb         # Model training notebook
├── embedding.ipynb         # Word embedding tutorial notebook
├── prediction.ipynb        # Prediction examples notebook
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Tech Stack

- **TensorFlow / Keras** - Deep learning framework
- **Streamlit** - Web application for inference
- **NumPy / Pandas** - Data manipulation
- **Scikit-learn** - ML utilities
- **Matplotlib** - Visualization
- **TensorBoard** - Training monitoring

## Setup

### Prerequisites

- Python 3.9+

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd movie-review-imdb
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Web Application

```bash
streamlit run app.py
```

This opens a web interface (default: `http://localhost:8501`) where you can:

1. Enter a movie review in the text area
2. Click **Classify**
3. View the predicted sentiment (Positive/Negative) and confidence score

### Retrain the Model (Optional)

Open `simplernn.ipynb` in Jupyter Notebook and run all cells. The trained model will be saved as `simple_rnn_imdb.h5`.

```bash
jupyter notebook simplernn.ipynb
```

## Model Architecture

| Layer            | Details                                        |
|------------------|-------------------------------------------------|
| Embedding        | Vocab size: 10,000 → 128-dimensional vectors   |
| SimpleRNN        | 128 hidden units, ReLU activation               |
| Dense (Output)   | 1 neuron, Sigmoid activation                    |

**Total Parameters:** 1,313,025

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Max Sequence Length:** 500 tokens
- **Early Stopping:** Patience of 5 epochs

## Model Performance

| Metric              | Value  |
|----------------------|--------|
| Training Accuracy    | ~88%   |
| Validation Accuracy  | ~75%   |

The model was trained for 10 epochs with a 20% validation split and a batch size of 32.

## How It Works

```
User Input (raw text)
        │
        ▼
Preprocessing
  ├── Lowercase conversion
  ├── Tokenization (split into words)
  ├── Word-to-index encoding (IMDB vocabulary)
  └── Padding to 500 tokens
        │
        ▼
Embedding Layer (word → 128-dim vector)
        │
        ▼
SimpleRNN Layer (sequential pattern extraction)
        │
        ▼
Dense Layer + Sigmoid (probability output)
        │
        ▼
Sentiment: Positive (>0.5) or Negative (≤0.5)
```

## Notebooks

### `embedding.ipynb`
Tutorial notebook demonstrating word embedding concepts including one-hot encoding and Keras Embedding layers.

### `simplernn.ipynb`
End-to-end model training pipeline: loads the IMDB dataset, builds the SimpleRNN model, trains it with early stopping, and saves the trained weights.

### `prediction.ipynb`
Examples of loading the trained model and running sentiment predictions on sample reviews.
