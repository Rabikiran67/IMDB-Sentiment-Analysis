"""
Setup script to download required NLTK data for deployment.
This script should be run during deployment to ensure all NLTK data is available.
"""

import nltk
import ssl

def download_nltk_data():
    """Download required NLTK data."""
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        print("Downloading NLTK data...")
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        print("Continuing without NLTK data...")

if __name__ == "__main__":
    download_nltk_data()