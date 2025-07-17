#!/usr/bin/env python3
"""
Test script to verify all libraries and NLTK data are properly installed
"""

def test_imports():
    """Test all required library imports"""
    print("Testing library imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
    
    try:
        import seaborn as sns
        print("✅ Seaborn imported successfully")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
    
    try:
        from wordcloud import WordCloud
        print("✅ WordCloud imported successfully")
    except ImportError as e:
        print(f"❌ WordCloud import failed: {e}")
    
    try:
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✅ VADER Sentiment imported successfully")
    except ImportError as e:
        print(f"❌ VADER Sentiment import failed: {e}")
    
    try:
        from textblob import TextBlob
        print("✅ TextBlob imported successfully")
    except ImportError as e:
        print(f"❌ TextBlob import failed: {e}")
    
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
    
    try:
        from bs4 import BeautifulSoup
        print("✅ BeautifulSoup imported successfully")
    except ImportError as e:
        print(f"❌ BeautifulSoup import failed: {e}")
    
    try:
        import requests
        print("✅ Requests imported successfully")
    except ImportError as e:
        print(f"❌ Requests import failed: {e}")

def test_nltk_data():
    """Test NLTK data availability"""
    print("\nTesting NLTK data...")
    
    try:
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        print(f"✅ Stopwords loaded successfully ({len(stop_words)} words)")
    except Exception as e:
        print(f"❌ Stopwords failed: {e}")
    
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize("This is a test sentence.")
        print(f"✅ Punkt tokenizer working ({len(tokens)} tokens)")
    except Exception as e:
        print(f"❌ Punkt tokenizer failed: {e}")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores("This is a great day!")
        print(f"✅ VADER lexicon working (positive score: {scores['pos']})")
    except Exception as e:
        print(f"❌ VADER lexicon failed: {e}")

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\nTesting sentiment analysis...")
    
    test_text = "I love this product! It's amazing and works perfectly."
    
    try:
        # Test VADER
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(test_text)
        print(f"✅ VADER analysis: {vader_scores}")
    except Exception as e:
        print(f"❌ VADER analysis failed: {e}")
    
    try:
        # Test TextBlob
        from textblob import TextBlob
        blob = TextBlob(test_text)
        textblob_sentiment = blob.sentiment
        print(f"✅ TextBlob analysis: polarity={textblob_sentiment.polarity}, subjectivity={textblob_sentiment.subjectivity}")
    except Exception as e:
        print(f"❌ TextBlob analysis failed: {e}")

if __name__ == "__main__":
    print("🔍 Testing Sentiment Analysis Project Setup")
    print("=" * 50)
    
    test_imports()
    test_nltk_data()
    test_sentiment_analysis()
    
    print("\n" + "=" * 50)
    print("✅ Setup test completed!")