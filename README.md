# Speech Emotion Based Music Recommendation System

THis project is a speech-based music recommendation system that detects human emotions from voice and recommends music accordingly. It integrates speech emotion recognition and Spotify music recommendations into one seamless user experience. The project provides two alternative methods for emotion recognition and also includes a Streamlit web application for interactive use.

## Overview

Moodify aims to enhance music personalization using AI-driven speech emotion recognition. Based on the mood of the user inferred from their voice, it maps emotions to predefined genres and fetches real-time Spotify track suggestions.

## Features

- Emotion detection using CNN-based models

- Feature extraction using librosa

- SMOTE for class balancing

- 5-fold cross-validation

- Music genre mapping based on predicted emotion

- Integration with Spotify API

- Confusion matrix and training visualizations

- Deployable via Streamlit app

## Method 1

This method uses a detailed feature extraction pipeline including MFCCs, Chroma, Mel spectrogram, Tonnetz, ZCR, RMSE, and spectral features. It implements a 1D CNN model with dropout and batch normalization. Stratified 5-fold cross-validation is used to evaluate performance. The best model is saved and evaluated using a classification report and confusion matrix.

The model is trained using augmented audio and SMOTE-balanced datasets. After prediction, the emotion is mapped to a genre and recommended tracks are fetched from Spotify.

## Method 2

This method includes additional optimizations:

- Data augmentation: pitch shift, noise, time stretch

- Log-mel spectrogram feature extraction

- Cosine annealing learning rate scheduler

- Optional use of SMOTE and class weights

The CNN model follows a similar architecture and is trained on augmented, balanced data. Cross-validation ensures robustness, and final predictions are made using the best model.

## Web Application

The Streamlit web app allows users to upload a .wav file and receive an emotion classification and corresponding music recommendations from Spotify. It uses the model and scalers saved during method 2's training.

### Features

Upload .wav audio file

Plays back uploaded file

Shows predicted emotion

Displays recommended tracks via Spotify embeds

## Requirements

`Python 3.7+`
`librosa`
`spotipy`
`tensorflow`
`scikit-learn`
`seaborn`
`matplotlib`
`imbalanced-learn`
`streamlit`
`python-dotenv`

## Installation

`pip install -r requirements.txt`

## How to Use

`CLI / Notebook`

Place .wav files in the AudioWAV/ directory.

Run either `moodify-method_1.ipynb` or `moodify-method_2.ipynb`.

Use the test_audio_prediction function to evaluate results.

### Web App
In order to run web app, we need  `best_model.h5`(mode),`laber_encoder.pkl`(label encoder) and `scaler.pkl`(scaler) files.
`streamlit run app.py`

Upload a .wav file to get started.


## 📁 Project Structure

```
Moodify/
├── AudioWAV/                         # Contains all audio samples (train/test)
├── Code/                             # Source code for training, evaluation, and app
│   ├── moodify-method_1.ipynb        # Method 1 - Without data augmentation
│   ├── moodify-method_2.ipynb        # Method 2 - With data augmentation
│   ├── requirements.txt              # Required Python libraries
│   └── App/
│       ├── app.py                    # Streamlit web application
│       ├── best_model.h5             # Saved best model
│       ├── label_encoder.pkl         # Label encoder for predictions
│       └── scaler.pkl                # Data scaler (pickle)
├── Presentation.pdf         # Final presentation slides
├── Report.pdf  # Final report with ideas and outcomes
└── README.md                         # You are here
```

