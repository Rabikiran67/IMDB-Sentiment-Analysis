"""
Script to train sentiment analysis models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from utils.data_processor import DataProcessor
from utils.sentiment_analyzer import SentimentAnalyzer

def main():
    """Main function to train sentiment analysis models."""
    
    print("🚀 Starting Sentiment Analysis Model Training")
    print("=" * 50)
    
    # Initialize components
    data_processor = DataProcessor()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Check if dataset exists
    dataset_path = "data/IMDB Dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please download the IMDB dataset and place it in the data/ directory")
        print("You can download it from: https://www.kaggle.com/lakshmi25npathi/imdb-movie-reviews")
        print("Or run create_sample_data.py to create a sample dataset for testing")
        return
    
    try:
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        df = data_processor.load_and_preprocess_data(dataset_path)
        print(f"✅ Loaded {len(df)} samples")
        
        # Display data info
        print(f"📈 Data distribution:")
        if 'sentiment_binary' in df.columns:
            print(df['sentiment_binary'].value_counts())
        
        # Save processed data
        processed_path = "data/processed_data.csv"
        data_processor.save_processed_data(df, processed_path)
        
        # Prepare training data
        print("🔧 Preparing training and test data...")
        X_train, X_test, y_train, y_test = data_processor.prepare_train_test_data(df)
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        
        # Train different models
        models_to_train = ['logistic_regression', 'random_forest']
        best_model = None
        best_accuracy = 0
        
        for model_type in models_to_train:
            print(f"\n🤖 Training {model_type} model...")
            
            # Create new analyzer for each model
            analyzer = SentimentAnalyzer()
            analyzer.data_processor = data_processor  # Use the same preprocessor
            
            # Train model
            train_info = analyzer.train_ml_model(X_train, y_train, model_type)
            print(f"✅ Model trained with {train_info['training_samples']} samples")
            
            # Evaluate model
            eval_results = analyzer.evaluate_ml_model(X_test, y_test)
            accuracy = eval_results['accuracy']
            
            print(f"📊 {model_type} Accuracy: {accuracy:.4f}")
            print(f"📊 Classification Report:")
            report = eval_results['classification_report']
            print(f"  Precision (Negative): {report['0']['precision']:.4f}")
            print(f"  Recall (Negative): {report['0']['recall']:.4f}")
            print(f"  Precision (Positive): {report['1']['precision']:.4f}")
            print(f"  Recall (Positive): {report['1']['recall']:.4f}")
            print(f"  F1-Score: {report['macro avg']['f1-score']:.4f}")
            
            # Keep track of best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = analyzer
                best_model_type = model_type
        
        # Save the best model
        if best_model:
            print(f"\n💾 Saving best model ({best_model_type}) with accuracy: {best_accuracy:.4f}")
            model_path = "models/sentiment_model.pkl"
            vectorizer_path = "models/vectorizer.pkl"
            
            best_model.save_model(model_path, vectorizer_path)
            print("✅ Model and vectorizer saved successfully!")
            
            # Test the saved model
            print("\n🧪 Testing saved model...")
            test_analyzer = SentimentAnalyzer()
            test_analyzer.load_model(model_path, vectorizer_path)
            
            # Test with sample texts
            test_texts = [
                "This movie is absolutely amazing! I loved every minute of it.",
                "Terrible film. Waste of time and money.",
                "It was okay, nothing special but not bad either."
            ]
            
            print("Sample predictions:")
            for text in test_texts:
                result = test_analyzer.analyze_with_ml_model(text)
                print(f"Text: '{text[:50]}...'")
                print(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.4f})")
                print()
        
        print("🎉 Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()