import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import librosa

# NLTK resources should be pre-downloaded via training scripts or system setup

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans a text input: lowercases, removes non-alphabetic characters,
    filters stopwords, and lemmatizes words.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(w) for w in words if w not in stop_words
    ]
    return " ".join(cleaned_words)

def extract_audio_features(y=None, sr=22050, file_path=None):
    """
    Extracts MFCC (40), Chroma (12), and Mel-Spectrogram (128) features from audio.
    Can accept either a y (waveform) and sr (sample rate), or a file_path to load.
    Returns a 1D numpy array of concatenated mean features across the time dimension.
    Total length = 40 + 12 + 128 = 180.
    """
    if file_path is not None:
        try:
            # Load with soundfile or audioread automatically handled by librosa
            y, sr = librosa.load(file_path, sr=sr)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")
            
    if y is None or len(y) == 0:
        raise ValueError("Audio waveform y is empty or None")
        
    # Ensure audio length is sufficient to avoid STFT errors
    min_len = 2048
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)), 'constant')
        
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Extract Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    
    # Concatenate features
    features = np.hstack((mfcc_mean, chroma_mean, mel_mean))
    return features
