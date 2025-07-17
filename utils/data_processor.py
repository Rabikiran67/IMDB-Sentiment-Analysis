"""
Data processing utilities for sentiment analysis.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataProcessor:
    """
    A class to handle data preprocessing for sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the data processor with necessary NLTK components."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to be cleaned
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize text and apply lemmatization.
        
        Args:
            text (str): Cleaned text
            
        Returns:
            list: List of lemmatized tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def preprocess_text(self, text):
        """
        Complete text preprocessing pipeline.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text ready for analysis
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, file_path, text_column='review', label_column='sentiment'):
        """
        Load and preprocess dataset.
        
        Args:
            file_path (str): Path to the dataset file
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        # Load data
        df = pd.read_csv(file_path)
        
        # Preprocess text
        print("Preprocessing text data...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Convert sentiment labels to binary (if needed)
        if label_column in df.columns:
            df['sentiment_binary'] = df[label_column].map({
                'positive': 1, 'negative': 0,
                'pos': 1, 'neg': 0,
                1: 1, 0: 0
            })
        
        return df
    
    def create_features(self, texts, max_features=10000, fit_vectorizer=True):
        """
        Create TF-IDF features from text data.
        
        Args:
            texts (list): List of preprocessed texts
            max_features (int): Maximum number of features
            fit_vectorizer (bool): Whether to fit the vectorizer
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        if fit_vectorizer or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)
        
        return features
    
    def prepare_train_test_data(self, df, text_column='processed_text', 
                               label_column='sentiment_binary', test_size=0.2):
        """
        Prepare training and testing data.
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            text_column (str): Name of the processed text column
            label_column (str): Name of the label column
            test_size (float): Proportion of test data
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Create features
        X = self.create_features(df[text_column].tolist())
        y = df[label_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, df, output_path):
        """
        Save processed data to CSV file.
        
        Args:
            df (pd.DataFrame): Processed dataset
            output_path (str): Output file path
        """
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")