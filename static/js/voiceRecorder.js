// static/js/voiceRecorder.js

/**
 * Voice recording with optional Speech-to-Text transcription.
 *
 * STT providers:
 *   "disabled"       — record audio as file attachment (original behavior)
 *   "browser"        — use Web Speech API for real-time transcription
 *   "local"          — send recording to server /api/stt/transcribe (Whisper)
 *   "endpoint:<id>"  — send recording to server /api/stt/transcribe (API)
 */

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordingStartTime = null;
let recordingInterval = null;

// Browser STT state
let _recognition = null;

// Cached STT provider — refreshed on settings change
let _sttProvider = 'disabled';

// ── Live (interim) transcription state (browser provider) ──
let _initialInputValue = '';   // text already in #message before recording started
let _finalTranscript = '';     // finalized browser STT text
let _interimTranscript = '';   // in-progress browser STT text

// ── Voice Activity Detection (VAD) auto-stop ──
// Uniform across all providers: meter mic RMS and auto-stop after sustained silence.
const VAD_INTERVAL_MS = 100;   // metering tick
const SILENCE_RMS = 0.02;      // amplitude below this = silence (0..1)
const SILENCE_MS = 1800;       // continuous silence before auto-stop
const MIN_RECORD_MS = 500;     // ignore silence until at least this much has been recorded
let _audioCtx = null;
let _analyser = null;
let _vadTimer = null;
let _silenceStart = null;
let _hasSpoken = false;
let _cancelled = false;
let _onSilence = null;         // optional UI callback(remainingMs)

/**
 * Fetch current STT provider from server settings
 */
async function refreshSttProvider() {
  try {
    const res = await fetch('/api/stt/stats', { credentials: 'same-origin' });
    if (res.ok) {
      const stats = await res.json();
      _sttProvider = stats.provider || 'disabled';
      // Notify the send button to update its icon
      if (window._updateSendBtnIcon) window._updateSendBtnIcon();
    }
  } catch (e) {
    console.warn('Failed to fetch STT stats:', e);
  }
}

/**
 * Format seconds as MM:SS
 */
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
  const secs = (seconds % 60).toString().padStart(2, '0');
  return `${mins}:${secs}`;
}

/**
 * Reset UI state after recording ends
 */
function _resetRecordingUI() {
  isRecording = false;
  _stopVad();
  if (recordingInterval) {
    clearInterval(recordingInterval);
    recordingInterval = null;
  }
  // Restore the Record Voice (mic) button — that's where the recording
  // animation lives. Restore its icon/title from the shared icon set in app.js.
  const micBtn = document.getElementById('mic-btn');
  if (micBtn) {
    micBtn.classList.remove('recording', 'silence-near');
    micBtn.title = 'Record voice';
    if (window._odysseusBtnIcons && window._odysseusBtnIcons.mic) {
      micBtn.innerHTML = window._odysseusBtnIcons.mic;
    }
  }
  if (window._updateSendBtnIcon) {
    setTimeout(window._updateSendBtnIcon, 50);
  }
}

/**
 * Render the live (final + interim) browser transcript into the chat input,
 * preserving any text that was already present before recording started.
 */
function _renderLiveTranscript() {
  const input = document.getElementById('message');
  if (!input) return;
  const live = (_finalTranscript + _interimTranscript).trim();
  input.value = _initialInputValue ? _initialInputValue + ' ' + live : live;
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

/**
 * Visual feedback for the approaching auto-stop. Defaults to toggling a class
 * on the send button; can be overridden via setOnSilence().
 */
function _emitSilence(remaining) {
  const micBtn = document.getElementById('mic-btn');
  if (micBtn) {
    if (remaining < SILENCE_MS) micBtn.classList.add('silence-near');
    else micBtn.classList.remove('silence-near');
  }
  if (_onSilence) _onSilence(remaining);
}

/**
 * Stop the VAD metering and release the AudioContext.
 */
function _stopVad() {
  if (_vadTimer) { clearInterval(_vadTimer); _vadTimer = null; }
  if (_audioCtx) { try { _audioCtx.close(); } catch (e) { /* ignore */ } _audioCtx = null; }
  _analyser = null;
  _silenceStart = null;
}

/**
 * Start VAD metering on the live mic stream. Auto-stops recording after
 * SILENCE_MS of silence once speech has been detected.
 */
function _startVad(stream) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return;
  try {
    _audioCtx = new Ctx();
    const src = _audioCtx.createMediaStreamSource(stream);
    _analyser = _audioCtx.createAnalyser();
    _analyser.fftSize = 1024;
    src.connect(_analyser);
  } catch (e) {
    console.warn('VAD init failed:', e);
    return;
  }
  const buf = new Float32Array(_analyser.fftSize);
  _silenceStart = null;
  _hasSpoken = false;
  _vadTimer = setInterval(() => {
    if (!isRecording || !_analyser) return;
    _analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);
    const now = Date.now();
    const elapsed = now - recordingStartTime.getTime();
    if (rms >= SILENCE_RMS) {
      _hasSpoken = true;
      _silenceStart = null;
      _emitSilence(SILENCE_MS);
      return;
    }
    // silence
    if (!_hasSpoken) return; // wait for first speech before arming auto-stop
    if (_silenceStart === null) _silenceStart = now;
    const silentFor = now - _silenceStart;
    _emitSilence(Math.max(0, SILENCE_MS - silentFor));
    if (elapsed > MIN_RECORD_MS && silentFor >= SILENCE_MS) {
      stopRecording();
    }
  }, VAD_INTERVAL_MS);
}

/**
 * Start browser speech recognition alongside recording
 */
function startBrowserSTT() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;

  _finalTranscript = '';
  _interimTranscript = '';
  _recognition = new SpeechRecognition();
  _recognition.continuous = true;
  _recognition.interimResults = true; // stream words live into the input
  _recognition.lang = '';

  _recognition.onresult = (event) => {
    _interimTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const text = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        _finalTranscript += text + ' ';
      } else {
        _interimTranscript += text;
      }
    }
    _renderLiveTranscript();
  };

  _recognition.onerror = (e) => {
    console.warn('Browser STT error:', e.error);
  };

  _recognition.start();
}

function stopBrowserSTT() {
  if (_recognition) {
    try { _recognition.stop(); } catch (e) { /* ignore */ }
    _recognition = null;
  }
  const final = _finalTranscript.trim();
  const interim = _interimTranscript.trim();
  // Prefer finalized text; if the engine only emitted interim results
  // (e.g. Chrome hadn't finalized before stop), keep the live text so we
  // don't regress to the audio-attachment fallback.
  const text = final || interim;
  const input = document.getElementById('message');
  if (input) {
    input.value = _initialInputValue ? (_initialInputValue + (text ? ' ' + text : '')) : text;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }
  return text;
}

/**
 * Send audio to server for transcription
 */
async function transcribeOnServer(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'audio.webm');

  const res = await fetch('/api/stt/transcribe', {
    method: 'POST',
    credentials: 'same-origin',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail?.message || 'Transcription failed');
  }

  const data = await res.json();
  return data.text || '';
}

/**
 * Insert transcribed text into the chat input
 */
function insertTranscription(text, showToast) {
  if (!text) return;
  const input = document.getElementById('message');
  if (!input) return;

  const existing = input.value.trim();
  input.value = existing ? existing + ' ' + text : text;

  // Trigger auto-resize and icon update
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.focus();

  if (showToast) showToast('Transcribed');
}

/**
 * Start voice recording
 */
export function startRecording(onFileCreated, showToast, showError) {
  // Check for secure context (getUserMedia requires HTTPS or localhost)
  if (!window.isSecureContext) {
    if (showError) showError('Microphone requires HTTPS. Use a reverse proxy with SSL or access via localhost.');
    _resetRecordingUI();
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (showError) showError('Microphone not supported in this browser.');
    _resetRecordingUI();
    return;
  }

  audioChunks = [];

  const input = document.getElementById('message');
  _initialInputValue = input ? input.value.trim() : '';
  _cancelled = false;

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

      mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());
        _stopVad();

        // Cancelled — discard the clip, leave the input untouched.
        if (_cancelled) {
          _cancelled = false;
          _resetRecordingUI();
          return;
        }

        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const provider = _sttProvider;

        if (provider === 'browser') {
          const transcript = stopBrowserSTT(); // finalizes #message directly
          if (transcript) {
            if (showToast) showToast('Transcribed');
          } else if (!_initialInputValue) {
            if (showToast) showToast('No speech detected');
            const audioFile = new File([audioBlob], `voice-message-${Date.now()}.webm`, { type: 'audio/webm' });
            if (onFileCreated) onFileCreated(audioFile);
          }
        } else if (provider === 'local' || provider.startsWith('endpoint:')) {
          // Show "Transcribing..." feedback
          if (showToast) showToast('Transcribing...', 5000);
          try {
            const transcript = await transcribeOnServer(audioBlob);
            if (transcript) {
              insertTranscription(transcript, showToast);
            } else {
              if (showToast) showToast('No speech detected');
            }
          } catch (e) {
            console.error('STT transcription error:', e);
            if (showError) showError('Transcription failed: ' + e.message);
            // Fallback: attach as file
            const audioFile = new File([audioBlob], `voice-message-${Date.now()}.webm`, { type: 'audio/webm' });
            if (onFileCreated) onFileCreated(audioFile);
          }
        } else {
          // STT disabled — attach audio file
          const audioFile = new File([audioBlob], `voice-message-${Date.now()}.webm`, { type: 'audio/webm' });
          if (onFileCreated) onFileCreated(audioFile);
        }

        _resetRecordingUI();
      };

      mediaRecorder.start();
      isRecording = true;
      recordingStartTime = new Date();

      // Voice Activity Detection — auto-stop on sustained silence (all providers)
      _startVad(stream);

      // Start browser STT if that's the provider
      if (_sttProvider === 'browser') {
        startBrowserSTT();
      }

      if (showToast) {
        showToast('Recording...');
      }
    })
    .catch(error => {
      console.error('Microphone access error:', error);
      if (showError) {
        if (error.name === 'NotAllowedError') {
          showError('Microphone access denied. Check browser permissions.');
        } else if (error.name === 'NotFoundError') {
          showError('No microphone found.');
        } else {
          showError('Microphone error: ' + error.message);
        }
      }
      _resetRecordingUI();
    });
}

/**
 * Stop voice recording
 */
export function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    // isRecording will be set to false in _resetRecordingUI called from onstop
  } else {
    _resetRecordingUI();
  }
}

/**
 * Check if currently recording
 */
export function getIsRecording() {
  return isRecording;
}

/**
 * Cancel the current recording — discard the clip, do not insert any text.
 */
export function cancelRecording() {
  if (!isRecording) return;
  _cancelled = true;
  if (_recognition) {
    try { _recognition.abort(); } catch (e) { /* ignore */ }
    _recognition = null;
  }
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  } else {
    _resetRecordingUI();
  }
}

/**
 * Register a UI callback invoked with the remaining silence-ms before auto-stop.
 */
export function setOnSilence(cb) {
  _onSilence = cb;
}

/**
 * Initialize recording state
 */
export function init() {
  isRecording = false;
  refreshSttProvider();
}

const voiceRecorderModule = {
  startRecording,
  stopRecording,
  getIsRecording,
  cancelRecording,
  setOnSilence,
  init,
  refreshSttProvider,
  get _sttProvider() { return _sttProvider; },
  set _sttProvider(v) { _sttProvider = v; },
};

export default voiceRecorderModule;
