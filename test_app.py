import os
import sys
import time
import subprocess
import urllib.request
import urllib.parse
import json
import soundfile as sf

# Add parent path to import voice generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train_voice_model import generate_synthetic_wave

def main():
    print("=" * 60)
    print("Verification Script: Testing Server Endpoints")
    print("=" * 60)
    
    # 1. Start app.py in background
    print("Starting flask server app.py...")
    # Run with current python executable
    proc = subprocess.Popen([sys.executable, "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(4.0)  # Wait for flask to start up
    
    # Ensure it's running
    if proc.poll() is not None:
        print("Error: Flask failed to start.")
        stdout, stderr = proc.communicate()
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        sys.exit(1)
        
    try:
        url_text = "http://127.0.0.1:5000/predict/text"
        url_voice = "http://127.0.0.1:5000/predict/voice"
        
        # 2. Test Text Endpoint
        print("\nTesting Text Endpoint:")
        req_data = json.dumps({"text": "I am absolutely thrilled and happy about this new task!"}).encode("utf-8")
        req = urllib.request.Request(url_text, data=req_data, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            print("Status Code: 200")
            print("Predicted Emotion:", res_json.get("emotion"))
            print("Confidence: {:.2f}%".format(res_json.get("confidence")))
            print("Probabilities:")
            for emotion, score in res_json.get("probabilities").items():
                print(f"  - {emotion}: {score:.2f}%")
            
            # Assertions - a clearly happy sentence should predict "happy"
            assert res_json.get("emotion") == "happy", "Expected happy prediction for text!"
            print("Text prediction assertion PASSED.")

        # 3. Test Voice Endpoint
        print("\nTesting Voice Endpoint:")
        # Generate a temporary synthetic wav file representing 'angry'
        y = generate_synthetic_wave("angry", duration=2.5, sr=22050)
        temp_wav = "test_angry_voice.wav"
        sf.write(temp_wav, y, 22050)
        print(f"Generated temp wave: {temp_wav}")
        
        # We need a multipart form request to upload the file. We can import and use requests.
        try:
            import requests
            with open(temp_wav, 'rb') as f:
                res = requests.post(url_voice, files={'file': f})
            print("Status Code:", res.status_code)
            res_json = res.json()
            print("Predicted Emotion:", res_json.get("emotion"))
            print("Confidence: {:.2f}%".format(res_json.get("confidence")))
            print("Probabilities:")
            for emotion, score in res_json.get("probabilities").items():
                print(f"  - {emotion}: {score:.2f}%")
            
            assert res.status_code == 200, "Expected status 200"
            assert res_json.get("emotion") == "angry", "Expected angry prediction for synthetic angry voice!"
            print("Voice prediction assertion PASSED.")
            
        except ImportError:
            print("requests library not found. Skipping Multipart HTTP upload assertion (will manually test in browser).")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
    except Exception as e:
        print("\nVerification FAILED with Exception:", e)
        # Terminate server
        proc.terminate()
        sys.exit(1)
        
    print("\nAll automated local test assertions completed successfully.")
    # Terminate server
    proc.terminate()
    print("Flask server terminated.")
    print("=" * 60)

if __name__ == "__main__":
    main()
