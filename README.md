# IMDB-Sentiment-Analysis

🎬 A comprehensive sentiment analysis application built with Python for movie review analysis, featuring multiple analysis methods, interactive web interface, and machine learning capabilities trained on IMDB movie reviews.

## 🌐 **Live Demo**
**Try the app now:** [https://imdb-sentiment-analysis-mazt9aoekqckucbjktpg4v.streamlit.app/](https://imdb-sentiment-analysis-mazt9aoekqckucbjktpg4v.streamlit.app/)

## 🚀 Features

- **🔍 Multiple Analysis Methods**: TextBlob, VADER, and custom ML models
- **💻 Interactive Web Interface**: Built with Streamlit
- **⚡ Real-time Analysis**: Instant sentiment prediction
- **📊 Batch Processing**: Analyze multiple reviews at once
- **📈 Comprehensive Visualizations**: Charts, gauges, and word clouds
- **🤝 Ensemble Predictions**: Combine multiple methods for better accuracy
- **🎯 Model Training**: Train custom ML models on IMDB data

## 🛠️ Technology Stack

- **🎨 Frontend**: Streamlit
- **🤖 ML Libraries**: scikit-learn, NLTK, TextBlob, VADER
- **📊 Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
- **⚙️ Data Processing**: Pandas, NumPy
- **🌐 Web Scraping**: BeautifulSoup, Requests

## 📦 Installation

### 📋 Prerequisites
- 🐍 Python 3.8 or higher
- 📦 pip package manager

### 🔧 Setup Instructions

1. **📥 Clone the repository**
```bash
git clone https://github.com/Rabikiran67/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis
```

2. **🏗️ Create virtual environment**
```bash
python -m venv sentiment_env
```

3. **🔄 Activate virtual environment**
```bash
# Windows 🪟
sentiment_env\Scripts\activate

# macOS/Linux 🐧
source sentiment_env/bin/activate
```

4. **📥 Install dependencies**
```bash
pip install -r requirements.txt
```

5. **📚 Download NLTK data**
```bash
python setup_nltk.py
```

6. **🧪 Test the setup**
```bash
python test_setup.py
```

## 🚀 Usage

### 🌐 Running the Web Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501` 🌍

### 🎯 Training Custom Models

1. **📊 Download IMDB Dataset**
   - Download from [Kaggle IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-movie-reviews) 📈
   - Place `IMDB Dataset.csv` in the `data/` directory 📁

2. **🏋️ Train the model**
```bash
python train_model.py
```

3. **🎲 Create sample data (for testing)**
```bash
python create_sample_data.py
```

## 📊 Application Features

### 🎬 Single Review Analysis
- 🔍 Analyze individual movie reviews
- 🎯 Multiple analysis methods available
- 📈 Interactive visualizations
- 📊 Confidence scoring

### 📈 Batch Analysis
- 📤 Upload CSV files with multiple reviews
- ⚡ Process hundreds of reviews at once
- 💾 Export results
- 📊 Summary statistics and visualizations

### 🤖 Model Information
- 👁️ View available analysis methods
- 📈 Model performance metrics
- 📚 Training information

### ℹ️ About
- 📖 Application overview
- 🛠️ Technology details
- 📋 Usage instructions

## 🎨 App Icons & Stickers Used

### 📊 Sentiment Analysis Results
- **😊 Positive Sentiment**: Green color, happy emojis
- **😢 Negative Sentiment**: Red color, sad emojis  
- **😐 Neutral Sentiment**: Yellow/orange color, neutral emojis
- **📈 Confidence Score**: Gauge charts and progress bars

### 🎬 Navigation Icons
- **🎬 Single Review Analysis**: Movie camera for individual analysis
- **📊 Batch Analysis**: Chart icon for bulk processing
- **🤖 Model Information**: Robot icon for ML model details
- **ℹ️ About**: Information icon for app details

### 📈 Visualization Elements
- **📊 Charts**: Bar charts, pie charts, line graphs
- **🎯 Gauges**: Circular progress indicators
- **☁️ Word Clouds**: Visual text representation
- **📈 Metrics**: KPI cards and statistics

### 🔄 Status Indicators
- **✅ Success**: Green checkmarks for completed tasks
- **❌ Error**: Red X marks for failures
- **⚠️ Warning**: Yellow warning triangles
- **🔄 Loading**: Spinning indicators for processing
- **🤖 ML Model**: Robot icons for model status

### 🎮 Interactive Elements
- **🔘 Buttons**: Action buttons with hover effects
- **📤 File Upload**: Drag & drop file areas
- **🎚️ Sliders**: Parameter adjustment controls
- **📋 Forms**: Input fields and text areas

### 🔧 Analysis Methods

### 📝 TextBlob
- 📏 Rule-based sentiment analysis
- 🎯 Polarity and subjectivity scores
- ⚡ Simple and fast processing

### 😊 VADER (Valence Aware Dictionary and sEntiment Reasoner)
- 📚 Lexicon and rule-based analysis
- 📱 Optimized for social media text
- 😀 Handles emoticons and slang

### 🤖 Machine Learning Model
- 🎬 Custom trained on IMDB data
- 🔤 TF-IDF feature extraction
- 🎯 Logistic Regression and Random Forest options
- 📈 High accuracy on movie reviews

### 🤝 Ensemble Method
- 🔄 Combines all available methods
- 🗳️ Majority voting system
- 📊 Improved accuracy and reliability

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

1. **🧪 Quick Test**: Run `python test_setup.py` to verify installation
2. **🚀 Launch App**: Run `streamlit run streamlit_app.py`
3. **🎬 Try Analysis**: Enter a movie review in the Single Review Analysis page
4. **🔍 Explore Features**: Navigate through different pages using the sidebar

## 📈 Performance

- **📝 TextBlob**: ⚡ Fast, good for general sentiment
- **😊 VADER**: 🌟 Excellent for informal text and social media
- **🤖 ML Model**: 🎯 High accuracy on movie reviews (85%+ accuracy)
- **🤝 Ensemble**: 🏆 Best overall performance combining all methods

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🎬 IMDB for the movie review dataset
- 📚 NLTK team for natural language processing tools
- 🎨 Streamlit team for the amazing web framework
- 🤖 scikit-learn contributors for machine learning tools

## 📞 Support

If you encounter any issues:

1. 🔍 Check the troubleshooting section
2. 🧪 Run `python test_setup.py` to verify setup
3. 🐛 Create an issue in the repository

---

**Crafted with code & music 🎧 by Rabi Kiran 🤍**