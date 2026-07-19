// static/js/vad.js
// Shared Voice Activity Detection — used by both voice-to-text (voiceRecorder.js)
// and the Call feature (callController.js). Watches mic RMS to detect the start
// and end of an utterance.
//
// Two absolute thresholds (kept as sane fallbacks / minimums):
//   silenceRms — absolute floor for "true silence" (used when the room is quiet)
//   speechRms  — absolute minimum for "real speech" (barge-in floor)
// Anything between the two counts as "sound" (resets the silence timer) but is
// NOT treated as a speech-start, so speaker leakage during TTS playback doesn't
// falsely trigger barge-in.
//
// Adaptive noise floor:
//   In a real room the mic also hears fans, A/C, keyboard, etc. That noise usually
//   lands in the band [silenceRms, speechRms). With fixed thresholds it keeps
//   resetting the silence timer, so the utterance-ending timer never reaches
//   silenceMs and the call never answers (it just keeps "listening"). To avoid
//   that, VAD tracks an adaptive noise floor (EMA of the ambient RMS) and treats
//   "silence" relative to that floor. The floor is only updated while we believe
//   we're hearing ambient noise (below the speech threshold) and is paused while
//   the AI is speaking, so TTS echo can't drag the floor up.

export class VAD {
  constructor(opts = {}) {
    this.silenceRms   = opts.silenceRms   ?? 0.02;
    this.silenceMs    = opts.silenceMs    ?? 1500;
    this.minRecordMs  = opts.minRecordMs  ?? 500;
    this.speechRms    = opts.speechRms    ?? 0.06; // absolute minimum for speech start

    // Adaptive noise floor — tracks the room's ambient RMS so background sound
    // is treated as silence instead of holding the utterance open forever.
    this.adapt        = opts.adapt        ?? true;
    this.noiseFloor   = opts.noiseFloor   ?? 0.008; // seeded below typical ambient
    this.noiseAlpha   = opts.noiseAlpha   ?? 0.015; // EMA rate (lower = slower, stabler)
    this.speechDelta  = opts.speechDelta  ?? 0.035; // extra RMS above floor to count as speech
    this.silenceDelta = opts.silenceDelta ?? 0.010; // hysteresis: silence if rms < floor + this

    this.audioCtx = null;
    this.analyser = null;
    this.timer = null;
    this.buf = null;
    this.running = false;

    this._silenceStart = null;
    this._hasSpoken = false;
    this._speechStartTs = null;
    this._startTs = 0;
    this._floorPaused = false; // when true, don't adapt the floor (AI is speaking)

    // Callbacks (set by the consumer)
    this.onSilence = null;     // (remainingMs) => void — called each tick before utterance end
    this.onSpeechEnd = null;   // () => void — fired after sustained silence (end of utterance)
    this.onSpeechStart = null; // () => void — fired when real speech begins (barge-in candidate)
  }

  // Pause/resume adaptation of the noise floor. Call with `true` while the AI is
  // speaking so its own voice (leaking back into the mic) can't raise the floor
  // and make the call deaf to the user afterward.
  setFloorPaused(paused) { this._floorPaused = !!paused; }

  start(stream) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return false;
    try {
      this.audioCtx = new Ctx();
      const src = this.audioCtx.createMediaStreamSource(stream);
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 1024;
      src.connect(this.analyser);
    } catch (e) {
      console.warn('VAD init failed:', e);
      return false;
    }
    this.buf = new Float32Array(this.analyser.fftSize);
    this._silenceStart = null;
    this._hasSpoken = false;
    this._speechStartTs = null;
    this._floorPaused = false;
    this._startTs = Date.now();
    this.running = true;
    this.timer = setInterval(() => this._tick(), 100);
    return true;
  }

  _rms() {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this.buf);
    let sum = 0;
    for (let i = 0; i < this.buf.length; i++) sum += this.buf[i] * this.buf[i];
    return Math.sqrt(sum / this.buf.length);
  }

  _tick() {
    if (!this.running || !this.analyser) return;
    const rms = this._rms();
    const now = Date.now();
    const elapsed = now - this._startTs;

    // Adaptive floor: ease toward the observed RMS only when it looks like
    // ambient noise (below the speech threshold) and adaptation isn't paused.
    const floor = this.noiseFloor;
    const speechThresh = Math.max(this.speechRms, floor + this.speechDelta);
    if (this.adapt && !this._floorPaused && rms < speechThresh) {
      this.noiseFloor = floor * (1 - this.noiseAlpha) + rms * this.noiseAlpha;
    }

    const silenceThresh = Math.max(this.silenceRms, this.noiseFloor + this.silenceDelta);

    if (rms >= speechThresh) {
      // Real speech — candidate for barge-in.
      if (!this._hasSpoken) {
        this._hasSpoken = true;
        this._speechStartTs = now;
        if (this.onSpeechStart) this.onSpeechStart();
      }
      this._silenceStart = null;
      this._speechStartTs = null;
      if (this.onSilence) this.onSilence(this.silenceMs);
      return;
    }

    if (rms >= silenceThresh) {
      // Ambient / leakage: not silence, not speech — hold the utterance open but
      // don't treat it as a speech start (prevents barge-in on speaker leakage).
      this._silenceStart = null;
      if (this.onSilence) this.onSilence(this.silenceMs);
      return;
    }

    // True silence (relative to the adaptive ambient floor).
    if (!this._hasSpoken) return; // wait for first speech before arming
    if (elapsed < this.minRecordMs) return;
    if (this._silenceStart === null) this._silenceStart = now;
    const silentFor = now - this._silenceStart;
    if (this.onSilence) this.onSilence(Math.max(0, this.silenceMs - silentFor));
    if (silentFor >= this.silenceMs) {
      this._hasSpoken = false; // re-arm for the next utterance
      this._silenceStart = null;
      if (this.onSpeechEnd) this.onSpeechEnd();
    }
  }

  stop() {
    this.running = false;
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    if (this.audioCtx) { try { this.audioCtx.close(); } catch (e) { /* ignore */ } this.audioCtx = null; }
    this.analyser = null;
    this._silenceStart = null;
    this._hasSpoken = false;
    this._floorPaused = false;
  }
}

export default VAD;
