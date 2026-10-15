// SensaAI Client Scripts
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');
    
    const textForm = document.getElementById('text-form');
    const textInput = document.getElementById('text-input');
    const textSubmitBtn = document.getElementById('text-submit-btn');
    const textSpinner = document.getElementById('text-spinner');
    
    // Voice Elements
    const recordBtn = document.getElementById('record-btn');
    const recordingStatus = document.getElementById('recording-status');
    const recordingTimer = document.getElementById('recording-timer');
    const playbackContainer = document.getElementById('playback-container');
    const audioPlayback = document.getElementById('audio-playback');
    
    const dropzone = document.getElementById('dropzone');
    const audioFileInput = document.getElementById('audio-file');
    const dropzoneText = document.getElementById('dropzone-text');
    const fileBanner = document.getElementById('file-banner');
    const selectedFileName = document.getElementById('selected-file-name');
    const removeFileBtn = document.getElementById('remove-file');
    
    const voiceSubmitBtn = document.getElementById('voice-submit-btn');
    const voiceSpinner = document.getElementById('voice-spinner');
    
    // Results & Errors
    const resultsContainer = document.getElementById('results-container');
    const resultEmoji = document.getElementById('result-emoji');
    const resultEmotion = document.getElementById('result-emotion');
    const confidenceBars = document.getElementById('confidence-bars');
    const modelNotes = document.getElementById('model-notes');
    const noteContent = document.getElementById('note-content');
    
    const errorCard = document.getElementById('error-card');
    const errorMessage = document.getElementById('error-message');
    
    // Sound recording variables
    let isRecording = false;
    let audioContext;
    let streamRef;
    let processorNode;
    let sourceNode;
    let audioChunks = [];
    const sampleRate = 22050;
    let timerInterval;
    let secondsElapsed = 0;
    
    let recordedBlob = null;
    let selectedFile = null;

    // Emotion settings
    const emotionConfig = {
        happy: { emoji: '😊', color: 'var(--happy-color)' },
        sad: { emoji: '😢', color: 'var(--sad-color)' },
        angry: { emoji: '😡', color: 'var(--angry-color)' },
        fear: { emoji: '😨', color: 'var(--fear-color)' },
        surprise: { emoji: '😲', color: 'var(--surprise-color)' },
        neutral: { emoji: '😐', color: 'var(--neutral-color)' },
        disgust: { emoji: '🤢', color: 'var(--disgust-color)' }
    };

    // --- TAB SWITCHER ---
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Switch tabs
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Clear current outputs
            hideResult();
            hideError();
        });
    });

    // --- TEXT ANALYSIS ---
    textForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const textVal = textInput.value.trim();
        if (!textVal) return;

        showLoader(textSubmitBtn, textSpinner);
        hideResult();
        hideError();

        try {
            const response = await fetch('/predict/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textVal })
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error classification.');
            }

            displayResults(data);
        } catch (err) {
            showError(err.message);
        } finally {
            hideLoader(textSubmitBtn, textSpinner);
        }
    });

    // --- VOICE ANALYSIS ---
    // File inputs & dragover / drop
    dropzone.addEventListener('click', () => {
        audioFileInput.click();
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        // Validation
        if (!file.type.startsWith('audio/') && !file.name.endsWith('.mp3') && !file.name.endsWith('.wav') && !file.name.endsWith('.ogg')) {
            showError("Unsupported format. Please select an audio file (WAV, MP3, OGG).");
            return;
        }

        selectedFile = file;
        recordedBlob = null; // Clear recorded audio
        playbackContainer.classList.add('hidden');
        audioPlayback.src = '';
        
        // Show file selected
        selectedFileName.textContent = file.name;
        dropzoneText.classList.add('hidden');
        fileBanner.classList.remove('hidden');
        
        enableVoiceSubmit();
        hideError();
        hideResult();
    }

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Stop clicking dropzone trigger
        clearSelectedFile();
    });

    function clearSelectedFile() {
        selectedFile = null;
        audioFileInput.value = '';
        fileBanner.classList.add('hidden');
        dropzoneText.classList.remove('hidden');
        checkVoiceInputs();
    }

    // --- MIC VOICE RECORDER ---
    recordBtn.addEventListener('click', async () => {
        if (isRecording) {
            stopRecordingSession();
        } else {
            await startRecordingSession();
        }
    });

    async function startRecordingSession() {
        try {
            audioChunks = [];
            secondsElapsed = 0;
            updateTimerDisplay();
            hideError();
            hideResult();

            // Set up AudioContext (standard 22050 Hz)
            const audioCtxClass = window.AudioContext || window.webkitAudioContext;
            audioContext = new audioCtxClass({ sampleRate: sampleRate });
            
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef = stream;
            
            sourceNode = audioContext.createMediaStreamSource(stream);
            
            // ScriptProcessorNode handles collecting PCM buffers
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            
            processorNode.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                audioChunks.push(new Float32Array(inputData));
            };
            
            sourceNode.connect(processorNode);
            processorNode.connect(audioContext.destination);

            // Update UI
            isRecording = true;
            recordBtn.classList.add('recording');
            recordingStatus.textContent = "Recording... Press button again to stop.";
            clearSelectedFile(); // Clear file upload in favor of microphone
            
            // Start Timer
            timerInterval = setInterval(() => {
                secondsElapsed++;
                updateTimerDisplay();
                if (secondsElapsed >= 10) {
                    stopRecordingSession();
                }
            }, 1000);

        } catch (err) {
            console.error("Microphone hardware error:", err);
            showError("Could not access microphone. Make sure permission is allowed and microphone is connected.");
        }
    }

    function stopRecordingSession() {
        if (!isRecording) return;
        
        // Stop timer
        clearInterval(timerInterval);
        
        // Stop nodes
        if (processorNode) {
            processorNode.disconnect();
        }
        if (sourceNode) {
            sourceNode.disconnect();
        }
        if (streamRef) {
            streamRef.getTracks().forEach(track => track.stop());
        }
        if (audioContext) {
            audioContext.close();
        }

        isRecording = false;
        recordBtn.classList.remove('recording');
        recordingStatus.textContent = "Recording captured. Ready to analyze.";

        const totalLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
        if (totalLength === 0) {
            showError("No audio was capture. Check microphone volume levels.");
            return;
        }

        // Rescale into final wav blob
        const finalCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sampleRate });
        const audioBuffer = finalCtx.createBuffer(1, totalLength, sampleRate);
        
        // Merge Float32Array chunks
        const mergedArray = new Float32Array(totalLength);
        let offset = 0;
        for (let chunk of audioChunks) {
            mergedArray.set(chunk, offset);
            offset += chunk.length;
        }
        audioBuffer.copyToChannel(mergedArray, 0);
        
        recordedBlob = bufferToWav(audioBuffer);
        
        // Enable local playback audio element
        const audioUrl = URL.createObjectURL(recordedBlob);
        audioPlayback.src = audioUrl;
        playbackContainer.classList.remove('hidden');

        enableVoiceSubmit();
    }

    function updateTimerDisplay() {
        const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
        const secs = String(secondsElapsed % 60).padStart(2, '0');
        recordingTimer.textContent = `${mins}:${secs}`;
    }

    function checkVoiceInputs() {
        if (recordedBlob || selectedFile) {
            enableVoiceSubmit();
        } else {
            disableVoiceSubmit();
        }
    }

    function enableVoiceSubmit() {
        voiceSubmitBtn.disabled = false;
        voiceSubmitBtn.classList.remove('disabled');
    }

    function disableVoiceSubmit() {
        voiceSubmitBtn.disabled = true;
        voiceSubmitBtn.classList.add('disabled');
    }

    // Submit Voice Form
    voiceSubmitBtn.addEventListener('click', async () => {
        const fileToUpload = selectedFile || recordedBlob;
        if (!fileToUpload) return;

        showLoader(voiceSubmitBtn, voiceSpinner);
        hideResult();
        hideError();

        const formData = new FormData();
        // Send under key "file"
        const filename = selectedFile ? selectedFile.name : "voice_micro_recording.wav";
        formData.append('file', fileToUpload, filename);

        try {
            const response = await fetch('/predict/voice', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server failed to analyze voice recording.');
            }

            displayResults(data);
        } catch (err) {
            showError(err.message);
        } finally {
            hideLoader(voiceSubmitBtn, voiceSpinner);
        }
    });


    // --- WAV COMPILER HELPER FUNCTIONS ---
    // Standard WAV container packing logic for raw Float32 data structure
    function bufferToWav(buffer) {
        let numOfChan = buffer.numberOfChannels,
            length = buffer.length * numOfChan * 2 + 44,
            bufferArr = new ArrayBuffer(length),
            view = new DataView(bufferArr),
            channels = [], i, sample,
            offset = 0,
            pos = 0;

        // write WAV headers
        setUint32(0x46464952);                         // "RIFF"
        setUint32(length - 8);                         // file length - 8
        setUint32(0x45564157);                         // "WAVE"

        setUint32(0x20746d66);                         // "fmt " chunk
        setUint32(16);                                 // chunk length = 16
        setUint16(1);                                  // sample format (raw PCM)
        setUint16(numOfChan);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * numOfChan); // byte rate (sampleRate * blockAlign)
        setUint16(numOfChan * 2);                      // block align (channel count * bytes per sample)
        setUint16(16);                                 // bits per sample

        setUint32(0x61746164);                         // "data" chunk
        setUint32(length - pos - 4);                   // chunk length

        for(i=0; i<buffer.numberOfChannels; i++)
            channels.push(buffer.getChannelData(i));

        while(pos < length - 4) {
            for(i=0; i<numOfChan; i++) {             // interleave channels
                sample = Math.max(-1, Math.min(1, channels[i][offset])); // clip
                sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF); // scale to 16-bit
                view.setInt16(pos, sample, true);    // write 16-bit signed INT
                pos += 2;
            }
            offset++;
        }

        return new Blob([view], { type: 'audio/wav' });

        function setUint16(data) {
            view.setUint16(pos, data, true);
            pos += 2;
        }

        function setUint32(data) {
            view.setUint32(pos, data, true);
            pos += 4;
        }
    }


    // --- HELPERS & RENDER ---
    function showLoader(btn, spinner) {
        btn.disabled = true;
        btn.classList.add('disabled');
        spinner.style.display = 'inline-block';
    }

    function hideLoader(btn, spinner) {
        btn.disabled = false;
        btn.classList.remove('disabled');
        spinner.style.display = 'none';
        checkVoiceInputs(); // Keep voice button inline with selections
    }

    function displayResults(data) {
        const topEmotion = data.emotion;
        const config = emotionConfig[topEmotion] || { emoji: '❓', color: 'var(--primary)' };
        
        // Update general badges
        resultEmoji.textContent = config.emoji;
        resultEmotion.textContent = topEmotion;
        
        // Set up probability bars list
        confidenceBars.innerHTML = '';
        
        // Sort emotions so the most likely is displayed at the top
        const sortedEmotions = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1]);

        sortedEmotions.forEach(([em, prob]) => {
            const conf = emotionConfig[em] || { color: '#ffffff' };
            const percentage = prob.toFixed(1);
            
            const barRow = document.createElement('div');
            barRow.className = 'bar-row';
            barRow.innerHTML = `
                <div class="bar-labels">
                    <span class="bar-name">${em}</span>
                    <span class="bar-val">${percentage}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="background-color: ${conf.color}; width: 0%;"></div>
                </div>
            `;
            confidenceBars.appendChild(barRow);
            
            // Trigger animation in next repaint frame
            setTimeout(() => {
                const fill = barRow.querySelector('.bar-fill');
                if (fill) fill.style.width = `${percentage}%`;
            }, 50);
        });

        // Handle Disclaimer Notes
        if (data.note) {
            noteContent.textContent = data.note;
            modelNotes.classList.remove('hidden');
        } else {
            modelNotes.classList.add('hidden');
        }

        resultsContainer.classList.remove('hidden');
        
        // Smooth scroll to container
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideResult() {
        resultsContainer.classList.add('hidden');
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorCard.classList.remove('hidden');
        errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideError() {
        errorCard.classList.add('hidden');
    }
});
