#!/usr/bin/env python3
"""
Script to verify the project structure is complete and correct.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status."""
    if os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}/")
        return True
    else:
        print(f"❌ {description}: {dirpath}/ (MISSING)")
        return False

def main():
    """Main function to verify project structure."""
    print("🔍 Verifying Sentiment Analysis Project Structure")
    print("=" * 60)
    
    all_good = True
    
    # Check directories
    print("\n📁 Checking Directories:")
    all_good &= check_directory_exists("data", "Data directory")
    all_good &= check_directory_exists("models", "Models directory")
    all_good &= check_directory_exists("utils", "Utils directory")
    all_good &= check_directory_exists("sentiment_env", "Virtual environment")
    
    # Check main files
    print("\n📄 Checking Main Files:")
    all_good &= check_file_exists("streamlit_app.py", "Streamlit web application")
    all_good &= check_file_exists("train_model.py", "Model training script")
    all_good &= check_file_exists("requirements.txt", "Requirements file")
    all_good &= check_file_exists("README.md", "Main README file")
    all_good &= check_file_exists("test_setup.py", "Setup test script")
    
    # Check utils files
    print("\n🛠️ Checking Utils Files:")
    all_good &= check_file_exists("utils/__init__.py", "Utils package init")
    all_good &= check_file_exists("utils/data_processor.py", "Data processor utility")
    all_good &= check_file_exists("utils/sentiment_analyzer.py", "Sentiment analyzer utility")
    
    # Check models files
    print("\n🤖 Checking Models Files:")
    all_good &= check_file_exists("models/__init__.py", "Models package init")
    all_good &= check_file_exists("models/README.md", "Models README")
    
    # Check data files
    print("\n📊 Checking Data Files:")
    all_good &= check_file_exists("data/README.md", "Data README")
    
    # Check optional files (generated after training)
    print("\n🔄 Checking Generated Files (Optional):")
    check_file_exists("models/sentiment_model.pkl", "Trained sentiment model")
    check_file_exists("models/vectorizer.pkl", "TF-IDF vectorizer")
    check_file_exists("data/IMDB Dataset.csv", "IMDB dataset")
    check_file_exists("data/processed_data.csv", "Processed dataset")
    
    # Test imports
    print("\n🧪 Testing Imports:")
    try:
        from utils.data_processor import DataProcessor
        print("✅ DataProcessor import successful")
    except ImportError as e:
        print(f"❌ DataProcessor import failed: {e}")
        all_good = False
    
    try:
        from utils.sentiment_analyzer import SentimentAnalyzer
        print("✅ SentimentAnalyzer import successful")
    except ImportError as e:
        print(f"❌ SentimentAnalyzer import failed: {e}")
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Project structure is complete and correct!")
        print("\n📋 Next Steps:")
        print("1. Download IMDB Dataset.csv and place it in data/ directory")
        print("2. Run: python train_model.py (to train models)")
        print("3. Run: streamlit run streamlit_app.py (to start web app)")
    else:
        print("⚠️ Some files are missing. Please check the structure.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())