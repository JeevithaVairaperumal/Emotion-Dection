# SensaAI - Multimodal Emotion Detection from Voice & Text

SensaAI is a fully running, multimodal AI application that classifies emotional states based on user-typed text messages or voice microphone speech recordings. Built from scratch with a Python Flask server, scikit-learn classifiers, and a modern glassmorphic dark-theme user interface.

---

## Technical Architecture & Pipelines

### 📝 1. Text Emotion Pipeline
- **Preprocessing:** Text input is processed by a shared pipeline where it is:
  1. Low-cased and stripped of special characters or integers.
  2. Filtered to remove standard English NLTK stopwords.
  3. Lemmatized using NLTK's `WordNetLemmatizer` to compute base word roots.
- **Vectorization:** TF-IDF Vectorizer (bi-grams included) fits on cleaned text to extract 5,000 top features.
- **Classification:** Multinomial Logistic Regression model trained on public Twitter sentiment datasets.
- **Dataset:** Mapped 13 Kaggle tweet categories into 6 core emotions (`happy, sad, angry, fear, surprise, neutral`). 
  - *Data Honesty Note:* The `disgust` class is excluded from the text classifier due to a lack of statistically significant samples in the source corpus. It is represented as `0.0%` in text prediction distributions.

### 🎙️ 2. Voice Emotion Pipeline
- **Preprocessing:** Audio samples are read, resampled to $22,050 \text{ Hz}$, padded to a minimum of 2048 samples, and analyzed.
- **Feature Extraction:** Standard acoustic features are extracted and averaged across time frames to create a 1D, 180-dimensional feature vector:
  - **MFCC:** Mel-Frequency Cepstral Coefficients ($40 \text{ features}$) mapping vocal timbres/spectral envelopes.
  - **Chroma STFT:** Chromagram analysis ($12 \text{ features}$) mapping harmonic pitch classes.
  - **Mel-Spectrogram:** Mel-scaled frequency band energies ($128 \text{ features}$) mapping power densities.
- **Classification:** Random Forest Classifier ($100 \text{ estimators}$) trained on a scaled features vector.

---

## ⚠️ Data Honesty: The Voice Model Disclaimer

Because raw academic audio emotion corpora (like RAVDESS, TESS) are extremely large and require custom licensing or mirror pathways, **this voice pipeline utilizes synthetic sound waveforms generated in code for demonstration purposes.** 

- **How it works:** A Python waveform generator synthesizes signals by matching pitch, local pitch jitter, relative speed parameters, envelope shapes, and signal energies to documented acoustic correlates of the 7 core emotions (e.g. Angry = high pitch, fast tempo, high jitter; Sad = low pitch, slow decay, flat volume).
- **Demo Status:** The voice model has learned to classify *these synthetic signals* rather than human speech. Predictions on real microphone recordings should be treated as a **working pipeline demo** rather than a clinically accurate result.

---

## Project Structure

```
scarlet-aphelion/
├── app.py                      # Flask server application and API routes
├── requirements.txt            # Python environments dependencies
├── README.md                   # Setup guide and architecture overview
├── test_app.py                 # Automated unit and integration test script
├── models/                     # Saved model artifacts (.pkl files)
│   ├── text_emotion_model.pkl
│   ├── text_vectorizer.pkl
│   ├── voice_emotion_model.pkl
│   └── voice_scaler.pkl
├── train/                      # Training logic and resources
│   ├── train_text_model.py     # Downloads CSV, trains Logistic Regression model
│   ├── train_voice_model.py    # Generates synthetic data, trains Random Forest model
│   └── data/                   # Train datasets
│       ├── tweet_emotions.csv  # Labeled tweet corpus (downloaded dynamically)
│       └── real_audio/         # Subdirectories for swap-in speech samples
├── utils/                      # Sharing modules
│   └── features.py             # Shared text-cleaner and audio-feature-extractor
├── templates/                  # Frontend structures
│   └── index.html              # Beautiful dark-themed layout
└── static/                     # Styles and client scripts
    ├── style.css               # Glassmorphic dark designs
    └── script.js               # Client controller (Microphone to WAV compilation & AJAX calls)
```

---

## Quick Start Guide

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Environment Setup & Installation
Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Training/Retraining the Models
The project ships with pretrained models pre-packaged in the `models/` directory, so it runs out-of-the-box. To retrain:
```bash
# Train voice model (synthetic base)
python train/train_voice_model.py

# Train text model (fetches Kaggle dataset)
python train/train_text_model.py
```

### 4. Running the Web Application
Start the server:
```bash
python app.py
```
Open a browser and navigate to **`http://127.0.0.1:5000`**.

---

## Swapping In Real Audio Data

To expand this voice model to recognize real human speech:
1. Record a series of WAV voice clips for the 7 emotion classes (`happy, sad, angry, fear, surprise, neutral, disgust`).
2. Save them into the directories structured as follows:
   ```
   train/data/real_audio/happy/*.wav
   train/data/real_audio/sad/*.wav
   # ... [for all 7 emotions]
   ```
3. Run `python train/train_voice_model.py` again.
4. The script will automatically detect the presence of human audio recordings, bypass the synthetic generator, fit the Random Forest model on the real dataset, and update your saved models.
