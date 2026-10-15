
import os

# Restrict BLAS/OMP threading to prevent worker thread
# segmentation faults on Windows.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import tempfile
import uuid
import logging
import wave

import av
import joblib
import numpy as np

from flask import Flask, request, jsonify, render_template

# Add project root to path for imports.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.features import clean_text, extract_audio_features
from faster_whisper import WhisperModel


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
app.config["JSON_SORT_KEYS"] = False

MODELS_DIR = os.path.join(BASE_DIR, "models")

TEXT_MODEL_PATH = os.path.join(
    MODELS_DIR, "text_emotion_model.pkl"
)
TEXT_VEC_PATH = os.path.join(
    MODELS_DIR, "text_vectorizer.pkl"
)
VOICE_MODEL_PATH = os.path.join(
    MODELS_DIR, "voice_emotion_model.pkl"
)
VOICE_SCALER_PATH = os.path.join(
    MODELS_DIR, "voice_scaler.pkl"
)

ALL_EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "neutral",
    "disgust"
]

WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

whisper_model = None


# --------------------------------------------------
# WHISPER MODEL
# --------------------------------------------------

def get_whisper_model():
    """Load Whisper once and reuse it for subsequent requests."""
    global whisper_model

    if whisper_model is None:
        app.logger.info("Loading Whisper speech recognition model...")

        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )

        app.logger.info("Whisper model loaded successfully.")

    return whisper_model


# --------------------------------------------------
# MODEL STATUS AND PROBABILITIES
# --------------------------------------------------

def get_models_status():
    """Check whether the trained models exist."""
    status = {
        "text_model": os.path.exists(TEXT_MODEL_PATH),
        "text_vectorizer": os.path.exists(TEXT_VEC_PATH),
        "voice_model": os.path.exists(VOICE_MODEL_PATH),
        "voice_scaler": os.path.exists(VOICE_SCALER_PATH)
    }

    status["all_trained"] = all(status.values())
    return status


def format_probabilities(classes, probas):
    """Return probabilities in the unified seven-emotion format."""
    probability_pct = {
        str(c).strip().lower(): float(p) * 100
        for c, p in zip(classes, probas)
    }

    return {
        emotion: round(probability_pct.get(emotion, 0.0), 2)
        for emotion in ALL_EMOTIONS
    }


def load_audio_models():
    """Load the trained voice classifier and its scaler."""
    if not (
        os.path.exists(VOICE_MODEL_PATH)
        and os.path.exists(VOICE_SCALER_PATH)
    ):
        raise FileNotFoundError(
            "Voice emotion models are missing. "
            "Run train/train_voice_model.py first."
        )

    model = joblib.load(VOICE_MODEL_PATH)
    scaler = joblib.load(VOICE_SCALER_PATH)

    return model, scaler


# --------------------------------------------------
# AUDIO CONVERSION: WEBM / OGG / OTHER AUDIO TO WAV
# --------------------------------------------------

def convert_audio_to_wav(input_path, output_path):
    """
    Convert browser-recorded audio to mono, 16 kHz,
    16-bit PCM WAV using PyAV.

    This WAV output is suitable for librosa.load().
    """
    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=16000
    )

    with av.open(input_path, mode="r") as container:
        audio_stream = next(
            (
                stream for stream in container.streams
                if stream.type == "audio"
            ),
            None
        )

        if audio_stream is None:
            raise ValueError("No audio stream found in the uploaded file.")

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)

            for frame in container.decode(audio=audio_stream.index):
                converted_frames = resampler.resample(frame)

                for converted in converted_frames:
                    wav_file.writeframes(
                        converted.to_ndarray().tobytes()
                    )

            # Flush any remaining resampler output.
            for converted in resampler.resample(None):
                wav_file.writeframes(
                    converted.to_ndarray().tobytes()
                )

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 44:
        raise ValueError(
            "Audio conversion produced an empty or invalid WAV file."
        )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():
    """Render the existing SensaAI interface."""
    status = get_models_status()

    return render_template(
        "index.html",
        models_status=status
    )


# --------------------------------------------------
# TEXT EMOTION DETECTION
# --------------------------------------------------

@app.route("/predict/text", methods=["POST"])
def predict_text():
    """Analyze written text and predict its emotion."""
    status = get_models_status()

    if not (
        status["text_model"]
        and status["text_vectorizer"]
    ):
        return jsonify({
            "error": (
                "Text emotion classification models are not trained yet. "
                "Run train/train_text_model.py first."
            )
        }), 503

    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({
            "error": 'Missing required JSON key: "text"'
        }), 400

    text_input = data["text"]

    if not isinstance(text_input, str) or not text_input.strip():
        return jsonify({
            "error": "Text entry is empty or invalid. Please type a sentence."
        }), 400

    try:
        model = joblib.load(TEXT_MODEL_PATH)
        vectorizer = joblib.load(TEXT_VEC_PATH)

        cleaned = clean_text(text_input)

        if not cleaned:
            cleaned = text_input.strip().lower()

        vectorized_text = vectorizer.transform([cleaned])

        pred = model.predict(vectorized_text)[0]
        probas = model.predict_proba(vectorized_text)[0]
        classes = model.classes_

        confidence = float(np.max(probas)) * 100
        unified_probabilities = format_probabilities(classes, probas)

        return jsonify({
            "emotion": str(pred).strip().lower(),
            "confidence": round(confidence, 2),
            "probabilities": unified_probabilities,
            "note": (
                "The text model is trained on a public tweet-emotions "
                "dataset. The disgust class is omitted from text "
                "classification due to a lack of samples in the source corpus."
            )
        }), 200

    except Exception as e:
        app.logger.exception("Text emotion prediction failed")

        return jsonify({
            "error": f"An error occurred during text inference: {e}"
        }), 500


# --------------------------------------------------
# REAL SPEECH RECOGNITION + VOICE EMOTION DETECTION
# --------------------------------------------------

@app.route("/predict/voice", methods=["POST"])
def predict_voice():
    """
    1. Accept a microphone recording or uploaded audio.
    2. Transcribe speech with Whisper.
    3. Convert the uploaded audio to WAV for feature extraction.
    4. Extract audio features using the existing utility.
    5. Predict emotion using the trained voice classifier.
    6. Return transcript, language, emotion and probabilities.
    """
    status = get_models_status()

    if not (
        status["voice_model"]
        and status["voice_scaler"]
    ):
        return jsonify({
            "error": (
                "Voice emotion classification models are not trained yet. "
                "Run train/train_voice_model.py first."
            )
        }), 503

    if "file" not in request.files:
        return jsonify({
            "error": 'No audio file part. Use the form-data key "file".'
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "No selected audio file."
        }), 400

    temp_path = None
    wav_path = None

    try:
        # Save the uploaded recording with its original extension.
        suffix = os.path.splitext(uploaded_file.filename)[1].lower()

        if not suffix:
            suffix = ".webm"

        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"uploaded_{uuid.uuid4().hex}{suffix}"
        )

        uploaded_file.save(temp_path)

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({
                "error": "The uploaded audio file is empty or could not be saved."
            }), 400

        # STEP 1: TRANSCRIBE SPEECH USING WHISPER
        recognizer = get_whisper_model()

        segments, info = recognizer.transcribe(
            temp_path,
            beam_size=5,
            vad_filter=True
        )

        transcript_segments = []

        for segment in segments:
            segment_text = segment.text.strip()

            if segment_text:
                transcript_segments.append(segment_text)

        transcript = " ".join(transcript_segments).strip()
        detected_language = getattr(info, "language", None)

        if not transcript:
            return jsonify({
                "error": (
                    "No speech detected. Please speak clearly, "
                    "check your microphone, and try again."
                ),
                "transcript": "",
                "language": detected_language
            }), 422

        app.logger.info(
            "Speech transcription completed. Language: %s",
            detected_language
        )

        # STEP 2: CONVERT AUDIO TO WAV
        # Browser MediaRecorder commonly produces WebM/Opus.
        # Librosa/soundfile may not be able to read that format directly.
        wav_path = os.path.join(
            tempfile.gettempdir(),
            f"converted_{uuid.uuid4().hex}.wav"
        )

        convert_audio_to_wav(temp_path, wav_path)

        app.logger.info("Audio converted to WAV successfully.")

        # STEP 3: LOAD TRAINED VOICE MODEL AND SCALER
        model, scaler = load_audio_models()

        # STEP 4: EXTRACT FEATURES FROM CONVERTED WAV
        features = extract_audio_features(
            file_path=wav_path
        )

        features = np.asarray(
            features,
            dtype=np.float64
        ).reshape(1, -1)

        # STEP 5: SCALE FEATURES
        features_scaled = scaler.transform(features)

        # STEP 6: PREDICT VOICE EMOTION
        pred = model.predict(features_scaled)[0]

        probas = model.predict_proba(features_scaled)[0]
        classes = model.classes_

        confidence = float(np.max(probas)) * 100
        unified_probabilities = format_probabilities(classes, probas)

        # STEP 7: RETURN RESULTS
        response = {
            "transcript": transcript,
            "language": detected_language,
            "emotion": str(pred).strip().lower(),
            "confidence": round(confidence, 2),
            "probabilities": unified_probabilities,
            "note": (
                "Speech was transcribed using Whisper. Voice emotion was "
                "predicted using the existing trained audio classifier. "
                "Prediction reliability depends on the quality and "
                "representativeness of its training data."
            )
        }

        return jsonify(response), 200

    except FileNotFoundError as e:
        app.logger.exception("Voice model files missing")

        return jsonify({
            "error": str(e)
        }), 503

    except Exception as e:
        app.logger.exception("Voice analysis failed")

        return jsonify({
            "error": (
                "Failed to process audio or make prediction: "
                + str(e)
            )
        }), 500

    finally:
        # Remove temporary upload and converted WAV, even if analysis fails.
        for path in (temp_path, wav_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    app.logger.warning(
                        "Could not remove temporary file: %s",
                        path
                    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Check backend and trained model availability."""
    return jsonify({
        "status": "running",
        "models": get_models_status(),
        "speech_recognition": "faster-whisper",
        "whisper_loaded": whisper_model is not None
    }), 200


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Audio file exceeds the 10 MB upload limit."
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Requested API endpoint was not found."
    }), 404


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "error": "An internal server error occurred."
    }), 500


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    app.logger.info("Starting SensaAI Emotion Detection...")
    app.logger.info("Text and voice emotion APIs initialized.")

    # Disable the reloader to avoid interrupting audio processing.
    app.run(
        debug=False,
        use_reloader=False,
        host="0.0.0.0",
        port=5000,
        threaded=False
    )