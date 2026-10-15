import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.features import extract_audio_features

# Target emotion classes
EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'disgust']

def generate_synthetic_wave(emotion, duration=2.5, sr=22050):
    """
    Generates a synthetic audio waveform with intentional acoustic correlates
    characteristic of the specified emotion.
    """
    # Randomize parameters slightly to create variance
    duration *= np.random.uniform(0.9, 1.1)
    
    if emotion == 'happy':
        base_f = np.random.uniform(240.0, 280.0)
        pitch_var = np.random.uniform(35.0, 50.0)
        energy = np.random.uniform(0.7, 0.9)
        jitter_level = np.random.uniform(0.03, 0.07)
        duration *= 0.8
    elif emotion == 'sad':
        base_f = np.random.uniform(120.0, 140.0)
        pitch_var = np.random.uniform(5.0, 15.0)
        energy = np.random.uniform(0.15, 0.3)
        jitter_level = np.random.uniform(0.005, 0.015)
        duration *= 1.4
    elif emotion == 'angry':
        base_f = np.random.uniform(280.0, 320.0)
        pitch_var = np.random.uniform(40.0, 60.0)
        energy = np.random.uniform(0.85, 1.0)
        jitter_level = np.random.uniform(0.12, 0.18)
        duration *= 0.7
    elif emotion == 'fear':
        base_f = np.random.uniform(260.0, 300.0)
        pitch_var = np.random.uniform(25.0, 35.0)
        energy = np.random.uniform(0.4, 0.6)
        jitter_level = np.random.uniform(0.15, 0.25)
        duration *= 0.6
    elif emotion == 'surprise':
        base_f = np.random.uniform(300.0, 340.0)
        pitch_var = np.random.uniform(70.0, 90.0)
        energy = np.random.uniform(0.8, 1.0)
        jitter_level = np.random.uniform(0.03, 0.07)
        duration *= 0.7
    elif emotion == 'disgust':
        base_f = np.random.uniform(130.0, 150.0)
        pitch_var = np.random.uniform(10.0, 20.0)
        energy = np.random.uniform(0.25, 0.35)
        jitter_level = np.random.uniform(0.2, 0.3)
        duration *= 1.3
    else:  # neutral
        base_f = np.random.uniform(170.0, 190.0)
        pitch_var = np.random.uniform(10.0, 20.0)
        energy = np.random.uniform(0.45, 0.55)
        jitter_level = np.random.uniform(0.01, 0.03)
        
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Generate frequency contour with vibrato + random jitter
    vibrato = np.sin(2 * np.pi * np.random.uniform(4.0, 6.0) * t) * pitch_var
    jitter = (np.random.rand(num_samples) - 0.5) * base_f * jitter_level
    freq_contour = base_f + vibrato + jitter
    
    # Synthesize signal
    phase = 2 * np.pi * np.cumsum(freq_contour) / sr
    y = np.sin(phase)
    
    # Harmonics
    y += 0.4 * np.sin(2 * phase)
    y += 0.2 * np.sin(3 * phase)
    
    # Amplitude envelope
    envelope = np.ones(num_samples)
    attack_sec = 0.03 if emotion == 'surprise' else 0.15
    attack_samples = int(sr * attack_sec)
    release_samples = int(sr * 0.25)
    
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    y = y * envelope * energy
    
    # Added noise
    noise_level = 0.01 if emotion not in ['angry', 'fear'] else 0.06
    y += np.random.normal(0, noise_level, num_samples)
    
    # Normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = (y / max_val) * energy
        
    return y

def load_real_dataset(data_dir, sr=22050):
    """
    Looks for real WAV files organized in subdirectories:
    data_dir/real_audio/{emotion}/*.wav
    
    Returns lists of features and labels if found, otherwise (None, None).
    """
    real_audio_dir = os.path.join(data_dir, 'real_audio')
    if not os.path.exists(real_audio_dir):
        return None, None
        
    X, y = [], []
    found_any = False
    
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(real_audio_dir, emotion)
        if os.path.exists(emotion_dir):
            for file_name in os.listdir(emotion_dir):
                if file_name.lower().endswith('.wav'):
                    file_path = os.path.join(emotion_dir, file_name)
                    try:
                        features = extract_audio_features(file_path=file_path, sr=sr)
                        X.append(features)
                        y.append(emotion)
                        found_any = True
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        
    if found_any:
        print(f"Successfully loaded {len(X)} real speech samples.")
        return np.array(X), np.array(y)
        
    return None, None

def main():
    print("=" * 60)
    print("Voice Emotion Model Training Pipeline")
    print("=" * 60)
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'real_audio'), exist_ok=True)
    
    # Make subfolders in real_audio for all target emotions to facilitate future real data drops
    for em in EMOTIONS:
        os.makedirs(os.path.join(data_dir, 'real_audio', em), exist_ok=True)
        
    # 1. Load data
    print("Checking for real audio files in train/data/real_audio/...")
    X, y_labels = load_real_dataset(data_dir)
    
    is_synthetic = False
    if X is None:
        print("No real dataset found in `train/data/real_audio/`. Generating synthetic dataset...")
        is_synthetic = True
        X_list, y_list = [], []
        samples_per_emotion = 100
        sr = 22050
        
        for emotion in EMOTIONS:
            print(f"Generating {samples_per_emotion} synthetic waves for: {emotion}")
            for _ in range(samples_per_emotion):
                wave = generate_synthetic_wave(emotion, sr=sr)
                features = extract_audio_features(y=wave, sr=sr)
                X_list.append(features)
                y_list.append(emotion)
                
        X = np.array(X_list)
        y_labels = np.array(y_list)
        print(f"Synthetic data generation complete. Total samples: {X.shape[0]}")
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    
    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Random Forest Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    if is_synthetic:
        print("(Note: High accuracy is expected on synthetic profiles)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=EMOTIONS))
    
    # 6. Save Artifacts
    model_path = os.path.join(models_dir, 'voice_emotion_model.pkl')
    scaler_path = os.path.join(models_dir, 'voice_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
