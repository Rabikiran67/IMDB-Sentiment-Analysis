"""
Create sample data for demonstration purposes.
"""

import pandas as pd
import numpy as np

def create_sample_imdb_data():
    """Create a sample IMDB-like dataset for demonstration."""
    
    # Sample positive reviews
    positive_reviews = [
        "This movie is absolutely fantastic! The acting is superb and the plot is engaging.",
        "I loved every minute of this film. Great cinematography and excellent direction.",
        "Amazing storyline with brilliant performances. Highly recommended!",
        "One of the best movies I've ever seen. Outstanding in every aspect.",
        "Incredible film with great character development and stunning visuals.",
        "Masterpiece! The dialogue is witty and the acting is phenomenal.",
        "Excellent movie with a compelling story and great special effects.",
        "Wonderful film that kept me engaged throughout. Brilliant acting!",
        "This is a must-watch movie. Great plot twists and amazing performances.",
        "Fantastic film with excellent direction and outstanding cinematography.",
        "Brilliant movie with great character arcs and emotional depth.",
        "Amazing film that exceeded all my expectations. Highly recommended!",
        "Outstanding movie with incredible acting and a gripping storyline.",
        "This film is a work of art. Beautiful cinematography and great music.",
        "Excellent movie with a unique plot and memorable characters.",
        "Great film with outstanding performances and brilliant direction.",
        "This movie is simply amazing. Great story and excellent execution.",
        "Wonderful film with great emotional impact and stunning visuals.",
        "Incredible movie with fantastic acting and a compelling narrative.",
        "This is an exceptional film with great character development.",
        "Amazing movie with brilliant performances and excellent writing.",
        "Outstanding film with great direction and memorable scenes.",
        "This movie is a masterpiece with incredible attention to detail.",
        "Excellent film with great pacing and outstanding cinematography.",
        "Brilliant movie with amazing performances and a gripping plot.",
        "This film is absolutely wonderful with great character chemistry.",
        "Amazing movie with excellent direction and stunning visual effects.",
        "Outstanding film with brilliant acting and a compelling story.",
        "This movie is fantastic with great dialogue and memorable moments.",
        "Incredible film with amazing performances and excellent production values.",
        "Great movie with outstanding direction and brilliant cinematography.",
        "This film is exceptional with amazing acting and a unique storyline.",
        "Wonderful movie with great emotional depth and stunning visuals.",
        "Amazing film with brilliant performances and excellent character development.",
        "This movie is outstanding with great direction and memorable dialogue.",
        "Excellent film with incredible acting and a compelling narrative.",
        "This movie is simply brilliant with amazing performances throughout.",
        "Outstanding film with great storytelling and excellent production quality.",
        "Amazing movie with brilliant direction and stunning cinematography.",
        "This film is a masterpiece with incredible attention to every detail.",
        "Excellent movie with great character arcs and emotional resonance.",
        "Brilliant film with amazing performances and outstanding direction.",
        "This movie is fantastic with great pacing and memorable scenes.",
        "Incredible film with excellent acting and a gripping storyline.",
        "Amazing movie with brilliant cinematography and great music score.",
        "This film is wonderful with outstanding performances and great writing.",
        "Excellent movie with amazing direction and compelling character development.",
        "Outstanding film with brilliant acting and stunning visual effects.",
        "This movie is incredible with great storytelling and memorable moments.",
        "Amazing film with excellent performances and outstanding production values."
    ]
    
    # Sample negative reviews
    negative_reviews = [
        "This movie is terrible. Poor acting and a confusing plot.",
        "Waste of time and money. The storyline is boring and predictable.",
        "Awful film with bad acting and terrible direction. Not recommended.",
        "One of the worst movies I've ever seen. Complete disaster.",
        "Horrible movie with poor character development and bad cinematography.",
        "Terrible film with awful dialogue and mediocre performances.",
        "Bad movie with a weak plot and disappointing special effects.",
        "Boring film that failed to keep me interested. Poor execution.",
        "This is a terrible movie with bad plot twists and awful acting.",
        "Disappointing film with poor direction and terrible cinematography.",
        "Bad movie with weak character development and boring storyline.",
        "Awful film that was a complete waste of time. Not recommended!",
        "Terrible movie with poor acting and a confusing narrative.",
        "This film is horrible with bad cinematography and awful music.",
        "Bad movie with a predictable plot and forgettable characters.",
        "Terrible film with poor performances and awful direction.",
        "This movie is simply bad with weak story and poor execution.",
        "Disappointing film with terrible emotional impact and bad visuals.",
        "Awful movie with poor acting and a boring narrative.",
        "This is a horrible film with weak character development.",
        "Bad movie with terrible performances and poor writing.",
        "Terrible film with awful direction and forgettable scenes.",
        "This movie is a disaster with poor attention to detail.",
        "Bad film with terrible pacing and awful cinematography.",
        "Horrible movie with poor performances and a weak plot.",
        "This film is absolutely terrible with bad character chemistry.",
        "Awful movie with poor direction and disappointing visual effects.",
        "Terrible film with bad acting and a boring story.",
        "This movie is horrible with poor dialogue and forgettable moments.",
        "Bad film with awful performances and poor production values.",
        "Terrible movie with disappointing direction and bad cinematography.",
        "This film is awful with poor acting and a predictable storyline.",
        "Horrible movie with bad emotional depth and terrible visuals.",
        "Awful film with poor performances and weak character development.",
        "This movie is terrible with bad direction and forgettable dialogue.",
        "Bad film with poor acting and a boring narrative.",
        "This movie is simply awful with terrible performances throughout.",
        "Disappointing film with bad storytelling and poor production quality.",
        "Terrible movie with awful direction and bad cinematography.",
        "This film is a disaster with poor attention to every detail.",
        "Bad movie with weak character arcs and no emotional resonance.",
        "Horrible film with terrible performances and awful direction.",
        "This movie is awful with bad pacing and forgettable scenes.",
        "Terrible film with poor acting and a boring storyline.",
        "Bad movie with awful cinematography and terrible music score.",
        "This film is horrible with disappointing performances and bad writing.",
        "Terrible movie with awful direction and weak character development.",
        "Disappointing film with poor acting and bad visual effects.",
        "This movie is awful with bad storytelling and forgettable moments.",
        "Terrible film with poor performances and disappointing production values."
    ]
    
    # Create DataFrame
    data = []
    
    # Add positive reviews
    for review in positive_reviews:
        data.append({'review': review, 'sentiment': 'positive'})
    
    # Add negative reviews
    for review in negative_reviews:
        data.append({'review': review, 'sentiment': 'negative'})
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def main():
    """Create and save sample dataset."""
    print("Creating sample IMDB dataset...")
    
    df = create_sample_imdb_data()
    
    # Save to CSV
    output_path = "data/IMDB Dataset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Sample dataset created with {len(df)} reviews")
    print(f"📊 Sentiment distribution:")
    print(df['sentiment'].value_counts())
    print(f"💾 Saved to: {output_path}")
    
    # Display sample reviews
    print("\n📝 Sample reviews:")
    for i, row in df.head(3).iterrows():
        print(f"Sentiment: {row['sentiment']}")
        print(f"Review: {row['review'][:100]}...")
        print()

if __name__ == "__main__":
    main()