# IMDB-Sentiment-Analysis

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)
[![TextBlob](https://img.shields.io/badge/TextBlob-2E8B57?style=for-the-badge&logo=python&logoColor=white)](https://textblob.readthedocs.io/)
[![VADER](https://img.shields.io/badge/VADER-FF6B6B?style=for-the-badge&logo=python&logoColor=white)](https://github.com/cjhutto/vaderSentiment)
[![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-4B8BBE?style=for-the-badge&logo=python&logoColor=white)](https://www.crummy.com/software/BeautifulSoup/)
[![Requests](https://img.shields.io/badge/Requests-2CA5E0?style=for-the-badge&logo=python&logoColor=white)](https://docs.python-requests.org/)
[![Live Website](https://img.shields.io/badge/Live_Website-Visit-green?style=for-the-badge&logo=streamlit&logoColor=white)](https://imdb-sentiment-analysis-mazt9aoekqckucbjktpg4v.streamlit.app/)

</div>

🎬 A comprehensive sentiment analysis application built with Python for movie review analysis, featuring multiple analysis methods, interactive web interface, and machine learning capabilities trained on IMDB movie reviews.

## 🚀 Features

- **Multiple Analysis Methods**: TextBlob, VADER, and custom ML models
- **Interactive Web Interface**: Built with Streamlit
- **Real-time Analysis**: Instant sentiment prediction
- **Batch Processing**: Analyze multiple reviews at once
- **Comprehensive Visualizations**: Charts, gauges, and word clouds
- **Ensemble Predictions**: Combine multiple methods for better accuracy
- **Model Training**: Train custom ML models on IMDB data

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Libraries**: scikit-learn, NLTK, TextBlob, VADER
- **Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
- **Data Processing**: Pandas, NumPy
- **Web Scraping**: BeautifulSoup, Requests

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Rabikiran67/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis
```

2. **Create virtual environment**
```bash
python -m venv sentiment_env
```

3. **Activate virtual environment**
```bash
# Windows
sentiment_env\Scripts\activate

# macOS/Linux
source sentiment_env/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Download NLTK data**
```bash
python setup_nltk.py
```

6. **Test the setup**
```bash
python test_setup.py
```

## 🚀 Usage

### Running the Web Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Training Custom Models

1. **Download IMDB Dataset**
   - Download from [Kaggle IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-movie-reviews)
   - Place `IMDB Dataset.csv` in the `data/` directory

2. **Train the model**
```bash
python train_model.py
```

3. **Create sample data (for testing)**
```bash
python create_sample_data.py
```

## 📊 Application Features

### Single Review Analysis
- Analyze individual movie reviews
- Multiple analysis methods available
- Interactive visualizations
- Confidence scoring

### Batch Analysis
- Upload CSV files with multiple reviews
- Process hundreds of reviews at once
- Export results
- Summary statistics and visualizations

### Model Information
- View available analysis methods
- Model performance metrics
- Training information

### About
- Application overview
- Technology details
- Usage instructions

## 🔧 Analysis Methods

### TextBlob
- Rule-based sentiment analysis
- Polarity and subjectivity scores
- Simple and fast processing

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Lexicon and rule-based analysis
- Optimized for social media text
- Handles emoticons and slang

### Machine Learning Model
- Custom trained on IMDB data
- TF-IDF feature extraction
- Logistic Regression and Random Forest options
- High accuracy on movie reviews

### Ensemble Method
- Combines all available methods
- Majority voting system
- Improved accuracy and reliability

## 📁 Project Structure

```
IMDB-Sentiment-Analysis/
├── streamlit_app.py          # Main Streamlit application
├── train_model.py            # Model training script
├── test_setup.py            # Setup verification script
├── setup_nltk.py            # NLTK data download script
├── create_sample_data.py    # Sample data generator
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── .gitignore             # Git ignore rules
├── .streamlit/            # Streamlit configuration
│   └── config.toml
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── data_processor.py  # Data processing utilities
│   └── sentiment_analyzer.py # Sentiment analysis core
├── data/                  # Data directory
│   └── README.md         # Data documentation
└── models/               # Trained models directory
    └── README.md        # Models documentation
```

## 🎯 Getting Started

1. **Quick Test**: Run `python test_setup.py` to verify installation
2. **Launch App**: Run `streamlit run streamlit_app.py`
3. **Try Analysis**: Enter a movie review in the Single Review Analysis page
4. **Explore Features**: Navigate through different pages using the sidebar

## 📈 Performance

- **TextBlob**: Fast, good for general sentiment
- **VADER**: Excellent for informal text and social media
- **ML Model**: High accuracy on movie reviews (85%+ accuracy)
- **Ensemble**: Best overall performance combining all methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IMDB for the movie review dataset
- NLTK team for natural language processing tools
- Streamlit team for the amazing web framework
- scikit-learn contributors for machine learning tools

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section
2. Run `python test_setup.py` to verify setup
3. Create an issue in the repository

---

**Crafted with code & music 🎧 by Rabi Kiran 🤍**