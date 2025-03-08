IMDB Sentiment Analysis using RNN (LSTM)

Overview

This project applies a Recurrent Neural Network (RNN) using Long Short-Term Memory (LSTM) layers to perform sentiment analysis on the IMDB movie reviews dataset. The dataset is provided by Keras and consists of 50,000 movie reviews labeled as positive or negative. The goal is to build a deep learning model capable of classifying the sentiment of reviews.

Features

Utilizes the IMDB dataset from Keras.

Implements an LSTM-based Recurrent Neural Network.

Performs text preprocessing, including tokenization and padding.

Trains and evaluates the model for sentiment classification.

Dataset

The IMDB dataset is a widely used dataset for binary sentiment classification. It consists of:

25,000 training samples (positive/negative reviews)

25,000 testing samples (positive/negative reviews)

Reviews preprocessed as sequences of word indexes

Installation

To run this project, install the required dependencies:

pip install tensorflow keras numpy matplotlib

Implementation Steps

Load Dataset: Import the IMDB dataset from Keras.

Preprocess Data: Tokenize and pad sequences for uniform input length.

Build Model: Define an RNN using LSTM layers.

Train Model: Train the network using labeled data.

Evaluate Model: Measure accuracy and loss on the test dataset.

Make Predictions: Test the model on sample reviews.

Usage

Run the script using Python:

python train_model.py

Model Architecture

Embedding Layer

LSTM Layer

Dense Output Layer (Sigmoid activation for binary classification)

Evaluation Metrics

Accuracy

Loss

Confusion Matrix

Results

The trained model achieves an accuracy of approximately 85% on the test dataset.

Future Improvements

Experimenting with different hyperparameters.

Using Bidirectional LSTMs for better context understanding.

Applying attention mechanisms to enhance model performance.
