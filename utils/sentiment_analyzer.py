"""
Sentiment analysis utilities using different approaches.
"""

import pickle
import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.data_processor import DataProcessor

class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis class supporting multiple methods.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.data_processor = DataProcessor()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.ml_model = None
        self.vectorizer = None
        self.model_trained = False
    
    def analyze_with_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert polarity to sentiment label
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'method': 'TextBlob',
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': abs(polarity)
        }
    
    def analyze_with_vader(self, text):
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Determine sentiment based on compound score
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'method': 'VADER',
            'sentiment': sentiment,
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'confidence': abs(compound)
        }
    
    def train_ml_model(self, X_train, y_train, model_type='logistic_regression'):
        """
        Train a machine learning model for sentiment analysis.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type (str): Type of model to train
            
        Returns:
            dict: Training results
        """
        # Select model
        if model_type == 'logistic_regression':
            self.ml_model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.ml_model = SVC(kernel='linear', random_state=42, probability=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        print(f"Training {model_type} model...")
        self.ml_model.fit(X_train, y_train)
        self.model_trained = True
        
        return {
            'model_type': model_type,
            'training_samples': X_train.shape[0],
            'features': X_train.shape[1]
        }
    
    def evaluate_ml_model(self, X_test, y_test):
        """
        Evaluate the trained ML model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation results
        """
        if not self.model_trained:
            raise ValueError("Model not trained yet. Call train_ml_model first.")
        
        # Make predictions
        y_pred = self.ml_model.predict(X_test)
        y_pred_proba = self.ml_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def analyze_with_ml_model(self, text):
        """
        Analyze sentiment using the trained ML model.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not self.model_trained:
            raise ValueError("Model not trained yet. Call train_ml_model first.")
        
        # Preprocess text
        processed_text = self.data_processor.preprocess_text(text)
        
        # Vectorize text
        if self.data_processor.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Train model first.")
        
        features = self.data_processor.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.ml_model.predict(features)[0]
        probabilities = self.ml_model.predict_proba(features)[0]
        
        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = max(probabilities)
        
        return {
            'method': 'ML Model',
            'sentiment': sentiment,
            'prediction': int(prediction),
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0],
                'positive': probabilities[1]
            }
        }
    
    def analyze_comprehensive(self, text):
        """
        Perform comprehensive sentiment analysis using all methods.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Combined sentiment analysis results
        """
        results = {
            'original_text': text,
            'textblob': self.analyze_with_textblob(text),
            'vader': self.analyze_with_vader(text)
        }
        
        # Add ML model results if available
        if self.model_trained:
            try:
                results['ml_model'] = self.analyze_with_ml_model(text)
            except Exception as e:
                results['ml_model'] = {'error': str(e)}
        
        # Create ensemble prediction
        sentiments = []
        confidences = []
        
        for method, result in results.items():
            if method != 'original_text' and 'error' not in result:
                sentiments.append(result['sentiment'])
                confidences.append(result['confidence'])
        
        # Majority vote for ensemble
        if sentiments:
            sentiment_counts = pd.Series(sentiments).value_counts()
            ensemble_sentiment = sentiment_counts.index[0]
            ensemble_confidence = np.mean(confidences)
            
            results['ensemble'] = {
                'sentiment': ensemble_sentiment,
                'confidence': ensemble_confidence,
                'method_agreement': len(set(sentiments)) == 1
            }
        
        return results
    
    def save_model(self, model_path, vectorizer_path):
        """
        Save the trained model and vectorizer.
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        """
        if not self.model_trained:
            raise ValueError("No model to save. Train a model first.")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.ml_model, f)
        
        # Save vectorizer
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.data_processor.vectorizer, f)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path, vectorizer_path):
        """
        Load a trained model and vectorizer.
        
        Args:
            model_path (str): Path to the saved model
            vectorizer_path (str): Path to the saved vectorizer
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.ml_model = pickle.load(f)
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.data_processor.vectorizer = pickle.load(f)
        
        self.model_trained = True
        print("Model and vectorizer loaded successfully")
    
    def batch_analyze(self, texts, method='comprehensive'):
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts (list): List of texts to analyze
            method (str): Analysis method to use
            
        Returns:
            list: List of analysis results
        """
        results = []
        
        for text in texts:
            if method == 'textblob':
                result = self.analyze_with_textblob(text)
            elif method == 'vader':
                result = self.analyze_with_vader(text)
            elif method == 'ml_model':
                result = self.analyze_with_ml_model(text)
            elif method == 'comprehensive':
                result = self.analyze_comprehensive(text)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            results.append(result)
        
        return results