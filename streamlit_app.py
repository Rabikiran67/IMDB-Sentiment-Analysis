"""
Streamlit web application for sentiment analysis.
"""

import streamlit as st
import os

# Download NLTK data if not available (for deployment)
try:
    import nltk
    import ssl
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Check if NLTK data exists, if not download it
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt_tab', quiet=True)
except Exception as e:
    st.warning(f"NLTK setup warning: {e}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import os
import pickle
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.data_processor import DataProcessor

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis App",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #ffc107;
        font-weight: bold;
    }
    .nav-button {
        margin-bottom: 0.5rem;
    }
    .active-page {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_analyzer():
    """Load the sentiment analyzer with cached models."""
    analyzer = SentimentAnalyzer()
    
    # Try to load trained model
    model_path = "models/sentiment_model.pkl"
    vectorizer_path = "models/vectorizer.pkl"
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            analyzer.load_model(model_path, vectorizer_path)
            return analyzer, True
        except Exception as e:
            st.warning(f"Could not load trained model: {e}")
            return analyzer, False
    else:
        return analyzer, False

def get_sentiment_color(sentiment):
    """Get color for sentiment display."""
    colors = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#ffc107'
    }
    return colors.get(sentiment.lower(), '#6c757d')

def create_sentiment_gauge(confidence, sentiment):
    """Create a gauge chart for sentiment confidence."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence ({sentiment})"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_sentiment_color(sentiment)},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_comparison_chart(results):
    """Create a comparison chart for different methods."""
    methods = []
    sentiments = []
    confidences = []
    
    for method, result in results.items():
        if method not in ['original_text', 'ensemble'] and 'error' not in result:
            methods.append(method.replace('_', ' ').title())
            sentiments.append(result['sentiment'])
            confidences.append(result['confidence'])
    
    df = pd.DataFrame({
        'Method': methods,
        'Sentiment': sentiments,
        'Confidence': confidences
    })
    
    fig = px.bar(
        df, 
        x='Method', 
        y='Confidence', 
        color='Sentiment',
        title='Sentiment Analysis Comparison',
        color_discrete_map={
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#ffc107'
        }
    )
    return fig

def create_wordcloud(text):
    """Create a word cloud from text."""
    if not text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 IMDB Sentiment Analysis App</h1>', unsafe_allow_html=True)
    
    # Load analyzer
    analyzer, model_loaded = load_sentiment_analyzer()
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Initialize session state for page selection
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Single Review Analysis"
    
    # Navigation buttons with active state indication
    pages = [
        ("🎬 Single Review Analysis", "Single Review Analysis"),
        ("📊 Batch Analysis", "Batch Analysis"),
        ("🤖 Model Information", "Model Information"),
        ("ℹ️ About", "About")
    ]
    
    for button_text, page_name in pages:
        # Show current page with different styling
        if st.session_state.current_page == page_name:
            st.sidebar.markdown(f"**➤ {button_text}**")
        else:
            if st.sidebar.button(button_text, use_container_width=True, key=f"nav_{page_name}"):
                st.session_state.current_page = page_name
                st.rerun()
    
    page = st.session_state.current_page
    
    # Add separator and model status
    st.sidebar.markdown("---")
    if model_loaded:
        st.sidebar.success("🤖 ML Model: Loaded")
    else:
        st.sidebar.warning("🤖 ML Model: Not Available")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Tips:**")
    st.sidebar.markdown("• Use specific movie language for better results")
    st.sidebar.markdown("• Try different analysis methods")
    st.sidebar.markdown("• Compare results across methods")
    
    if page == "Single Review Analysis":
        single_text_analysis(analyzer, model_loaded)
    elif page == "Batch Analysis":
        batch_analysis(analyzer, model_loaded)
    elif page == "Model Information":
        model_information(model_loaded)
    elif page == "About":
        about_page()

def single_text_analysis(analyzer, model_loaded):
    """Single text analysis page."""
    st.markdown('<h2 class="sub-header">📝 Single Movie Review Analysis</h2>', unsafe_allow_html=True)
    
    # Text input
    text_input = st.text_area(
        "Enter movie review to analyze:",
        placeholder="Type or paste your movie review here... (e.g., 'This movie was absolutely fantastic! Great acting and storyline.')",
        height=150
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        analysis_method = st.selectbox(
            "Analysis Method:",
            ["Comprehensive", "TextBlob", "VADER", "ML Model"] if model_loaded else ["Comprehensive", "TextBlob", "VADER"]
        )
    
    with col2:
        show_details = st.checkbox("Show detailed results", value=True)
    
    if st.button("Analyze Sentiment", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    if analysis_method == "Comprehensive":
                        results = analyzer.analyze_comprehensive(text_input)
                        display_comprehensive_results(results, show_details)
                    elif analysis_method == "TextBlob":
                        result = analyzer.analyze_with_textblob(text_input)
                        display_single_result(result, show_details)
                    elif analysis_method == "VADER":
                        result = analyzer.analyze_with_vader(text_input)
                        display_single_result(result, show_details)
                    elif analysis_method == "ML Model" and model_loaded:
                        result = analyzer.analyze_with_ml_model(text_input)
                        display_single_result(result, show_details)
                    
                    # Word cloud
                    if st.checkbox("Show Word Cloud"):
                        fig = create_wordcloud(text_input)
                        if fig:
                            st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter a movie review to analyze.")

def display_comprehensive_results(results, show_details):
    """Display comprehensive analysis results."""
    # Ensemble result
    if 'ensemble' in results:
        ensemble = results['ensemble']
        sentiment = ensemble['sentiment']
        confidence = ensemble['confidence']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Sentiment", sentiment.title())
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        with col3:
            agreement = "✅ Yes" if ensemble['method_agreement'] else "❌ No"
            st.metric("Method Agreement", agreement)
        
        # Gauge chart
        fig = create_sentiment_gauge(confidence, sentiment)
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison chart
    fig = create_comparison_chart(results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    if show_details:
        st.subheader("Detailed Results")
        for method, result in results.items():
            if method not in ['original_text', 'ensemble'] and 'error' not in result:
                with st.expander(f"{method.replace('_', ' ').title()} Results"):
                    st.json(result)

def display_single_result(result, show_details):
    """Display single method analysis result."""
    sentiment = result['sentiment']
    confidence = result.get('confidence', 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment", sentiment.title())
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    
    # Gauge chart
    fig = create_sentiment_gauge(confidence, sentiment)
    st.plotly_chart(fig, use_container_width=True)
    
    if show_details:
        st.subheader("Detailed Results")
        st.json(result)

def batch_analysis(analyzer, model_loaded):
    """Batch analysis page."""
    st.markdown('<h2 class="sub-header">📊 Batch Movie Review Analysis</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV file with movie review data:",
        type=['csv'],
        help="CSV file should have a column with movie review text to analyze"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows")
            
            # Column selection
            text_column = st.selectbox(
                "Select the movie review column:",
                df.columns.tolist()
            )
            
            # Analysis method
            method = st.selectbox(
                "Analysis Method:",
                ["comprehensive", "textblob", "vader", "ml_model"] if model_loaded else ["comprehensive", "textblob", "vader"]
            )
            
            if st.button("Analyze All Reviews", type="primary"):
                with st.spinner("Analyzing movie reviews..."):
                    texts = df[text_column].astype(str).tolist()
                    results = analyzer.batch_analyze(texts[:100], method)  # Limit to 100 for demo
                    
                    # Process results
                    sentiments = []
                    confidences = []
                    
                    for result in results:
                        if method == "comprehensive" and 'ensemble' in result:
                            sentiments.append(result['ensemble']['sentiment'])
                            confidences.append(result['ensemble']['confidence'])
                        else:
                            sentiments.append(result['sentiment'])
                            confidences.append(result.get('confidence', 0))
                    
                    # Add results to dataframe
                    df_results = df.head(len(results)).copy()
                    df_results['sentiment'] = sentiments
                    df_results['confidence'] = confidences
                    
                    # Display results
                    st.subheader("Analysis Results")
                    st.dataframe(df_results)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Analyzed", len(results))
                    with col2:
                        positive_count = sum(1 for s in sentiments if s == 'positive')
                        st.metric("Positive", positive_count)
                    with col3:
                        negative_count = sum(1 for s in sentiments if s == 'negative')
                        st.metric("Negative", negative_count)
                    
                    # Visualization
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_information(model_loaded):
    """Model information page."""
    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)
    
    if model_loaded:
        st.success("✅ Trained ML model is loaded and ready!")
        
        st.subheader("Available Analysis Methods")
        methods = [
            ("TextBlob", "Rule-based sentiment analysis using TextBlob library"),
            ("VADER", "Lexicon and rule-based sentiment analysis tool"),
            ("ML Model", "Custom trained machine learning model"),
            ("Comprehensive", "Ensemble of all methods with majority voting")
        ]
        
        for method, description in methods:
            st.write(f"**{method}**: {description}")
    else:
        st.warning("⚠️ No trained ML model found.")
        st.info("To train a model, run: `python train_model.py`")
        
        st.subheader("Available Analysis Methods")
        methods = [
            ("TextBlob", "Rule-based sentiment analysis using TextBlob library"),
            ("VADER", "Lexicon and rule-based sentiment analysis tool"),
            ("Comprehensive", "Ensemble of available methods")
        ]
        
        for method, description in methods:
            st.write(f"**{method}**: {description}")
    
    st.subheader("Model Training Requirements")
    st.write("To train the ML model, you need:")
    st.write("1. IMDB Dataset.csv in the data/ directory")
    st.write("2. Run the training script: `python train_model.py`")
    st.write("3. The trained model will be saved in the models/ directory")

def about_page():
    """About page."""
    st.markdown('<h2 class="sub-header">ℹ️ About</h2>', unsafe_allow_html=True)
    
    st.write("""
    ## IMDB Sentiment Analysis Application
    
    This application provides comprehensive sentiment analysis for movie reviews using multiple approaches:
    
    ### Features:
    - **Single Review Analysis**: Analyze individual movie reviews with multiple methods
    - **Batch Analysis**: Process multiple movie reviews from CSV files
    - **Multiple Methods**: TextBlob, VADER, and custom ML models trained on IMDB data
    - **Visualizations**: Interactive charts, gauges, and word clouds
    - **Ensemble Predictions**: Combine multiple methods for better accuracy
    
    ### Methods Used:
    
    **TextBlob**: A simple library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks.
    
    **VADER**: A lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
    
    **Machine Learning**: Custom trained models using scikit-learn with TF-IDF features, specifically trained on IMDB movie review data.
    
    ### Technology Stack:
    - **Frontend**: Streamlit
    - **ML Libraries**: scikit-learn, NLTK, TextBlob, VADER
    - **Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
    - **Data Processing**: Pandas, NumPy
    
    ### Developer:
    Crafted with code & music 🎧 by Rabi Kiran 🤍
    """)

if __name__ == "__main__":
    main()