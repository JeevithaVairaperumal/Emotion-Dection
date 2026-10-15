import os
import sys
import urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.features import clean_text

DATA_URL = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"

# Map the 13 raw sentiments to 6 of the 7 core emotions
# disgust is not represented in this dataset
EMOTION_MAP = {
    'happiness': 'happy',
    'love': 'happy',
    'fun': 'happy',
    'enthusiasm': 'happy',
    'sadness': 'sad',
    'hate': 'angry',
    'anger': 'angry',
    'worry': 'fear',
    'surprise': 'surprise',
    'neutral': 'neutral',
    'relief': 'neutral',
    'boredom': 'neutral',
    'empty': 'neutral'
}

def main():
    print("=" * 60)
    print("Text Emotion Model Training Pipeline")
    print("=" * 60)
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'tweet_emotions.csv')
    
    # 1. Download dataset if not locally present
    if not os.path.exists(csv_path):
        print(f"Downloading dataset from: {DATA_URL} ...")
        try:
            # Add user-agent header to avoid potential github blocks
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(DATA_URL, csv_path)
            print("Download successful.")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            print("Please ensure internet access or place a tweet_emotions.csv in train/data/")
            sys.exit(1)
    else:
        print(f"Dataset already exists locally at: {csv_path}")
        
    # 2. Load and Preprocess Data
    print("Loading and mapping dataset...")
    df = pd.read_csv(csv_path)
    
    # Verify expected columns
    if 'sentiment' not in df.columns or 'content' not in df.columns:
        print("Error: Dataset CSV must contain 'sentiment' and 'content' columns.")
        sys.exit(1)
        
    # Map raw labels
    df['emotion'] = df['sentiment'].map(EMOTION_MAP)
    
    # Drop rows that failed to map
    df = df.dropna(subset=['emotion'])
    
    print(f"Distribution of mapped emotions:")
    print(df['emotion'].value_counts())
    
    # Clean text in content
    print("Applying text cleaning (stopwords, lemmatization)...")
    df['cleaned_content'] = df['content'].apply(clean_text)
    
    # Drop empty contents
    df = df[df['cleaned_content'].str.strip() != '']
    print(f"Remaining samples after text cleaning: {len(df)}")
    
    X = df['cleaned_content']
    y = df['emotion']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Vectorize
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Train Model
    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    labels_order = list(set(EMOTION_MAP.values()))
    print(confusion_matrix(y_test, y_pred, labels=labels_order))
    
    # 7. Save Models
    model_path = os.path.join(models_dir, 'text_emotion_model.pkl')
    vec_path = os.path.join(models_dir, 'text_vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
