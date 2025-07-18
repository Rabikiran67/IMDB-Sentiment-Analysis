# Data Directory

This directory contains the datasets used for sentiment analysis.

## Required Files:

### IMDB Dataset.csv
- **Description**: IMDB movie reviews dataset for training sentiment analysis models
- **Source**: https://www.kaggle.com/lakshmi25npathi/imdb-movie-reviews
- **Format**: CSV file with columns:
  - `review`: Movie review text
  - `sentiment`: Sentiment label (positive/negative)
- **Size**: ~50,000 reviews
- **Usage**: Training and evaluation of machine learning models

### processed_data.csv
- **Description**: Preprocessed version of the IMDB dataset
- **Generated by**: `train_model.py` script
- **Format**: CSV file with additional columns:
  - `processed_text`: Cleaned and preprocessed review text
  - `sentiment_binary`: Binary sentiment labels (1=positive, 0=negative)
- **Usage**: Ready-to-use data for model training

## Instructions:

1. Download the IMDB Dataset.csv from Kaggle
2. Place it in this directory
3. Run `python train_model.py` to generate processed_data.csv
4. The processed data will be automatically created and saved here

## Data Preprocessing Steps:

The preprocessing pipeline includes:
- Text cleaning (remove HTML tags, URLs, special characters)
- Tokenization
- Stop word removal
- Lemmatization
- TF-IDF vectorization

## File Structure:
```
data/
├── IMDB Dataset.csv          # Original dataset (download required)
├── processed_data.csv        # Preprocessed dataset (auto-generated)
└── README.md                 # This file
```