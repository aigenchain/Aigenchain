// static/js/callController.js
// Real-time "Voice Call" mode for Aigenchain — turn-based, hands-free conversation.
//
// Loop:  LISTENING  ⇄  SPEAKING   (THINKING in between)
//   LISTENING : mic open, browser STT transcribes live; VAD detects end of
//               utterance → send text to the chat API → THINKING.
//   THINKING  : stream AI tokens, render them live into the overlay.
//   SPEAKING  : feed the full reply to TTS; barge-in (VAD speech-start) aborts
//               playback and returns to LISTENING.
//
// Reuses: VAD (vad.js), browser SpeechRecognition (STT), aiTTSManager (tts-ai.js),
// and the same /api/chat_stream endpoint the Send button uses.

import { VAD } from './vad.js';
import sessionModule from './sessions.js';
import uiModule from './ui.js';

const STATE = { IDLE: 'IDLE', LISTENING: 'LISTENING', THINKING: 'THINKING', SPEAKING: 'SPEAKING' };

// ── DOM (created on init) ──
let overlay = null;
let transcriptEl = null;
let statusEl = null;
let timerEl = null;
let endBtn = null;

// ── Runtime state ──
let state = STATE.IDLE;
let micStream = null;
let vad = null;
let recognition = null;
let abortCtrl = null;

let finalTranscript = '';   // all finalized STT text since recognition (re)start
let interimTranscript = ''; // in-progress STT text
let turnPrefixLen = 0;      // length of finalTranscript at the start of the current turn

let aiFullText = '';        // accumulated AI reply (reasoning + answer) for the current turn
let aiLineEl = null;        // live AI transcript line in the overlay

let callStartTime = 0;
let timerInterval = null;
let callActive = false;

// While the AI is speaking (THINKING/SPEAKING) the mic is "muted": STT
// transcription and VAD barge-in are ignored. This stops the AI's own
// speaker output from being heard back through the mic and re-sent to the
// model (the "too sensitive / repeats itself" loop).
let _micMuted = false;
const POST_SPEECH_COOLDOWN_MS = 700; // swallow trailing TTS echo before re-arming

// Call TTS overrides — saved/restored so the global TTS setting is untouched.
let _callTTSOn = false;
let _ttsPrev = null;
let _ttsWaitIv = null;

// The Call icon (idle + animated-active) now lives inside the Send button and is
// owned by app.js (_callIcon / _callIconActive). This module only drives the call
// logic + overlay and tells app.js when the call is active via window._aigenchainCallUI.

// ── UI construction ──
function buildUI() {
  // Note: the Call trigger is now the Send button itself (app.js swaps the
  // empty-state icon to the call icon and routes the click to callController).
  // We only build the overlay panel here.
  overlay = document.createElement('div');
  overlay.id = 'call-overlay';
  overlay.className = 'call-overlay hidden';
  overlay.innerHTML = `
    <div class="call-overlay-header">
      <div class="call-overlay-title">
        <span class="call-dot"></span> Voice Call
        <span class="call-timer" id="call-timer">00:00</span>
      </div>
      <button type="button" class="call-end-btn" id="call-end-btn" title="End call" aria-label="End call">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
        End
      </button>
    </div>
    <div class="call-transcript" id="call-transcript"></div>
    <div class="call-status" id="call-status">Ready</div>
  `;
  document.body.appendChild(overlay);

  transcriptEl = overlay.querySelector('#call-transcript');
  statusEl = overlay.querySelector('#call-status');
  timerEl = overlay.querySelector('#call-timer');
  endBtn = overlay.querySelector('#call-end-btn');
  endBtn.addEventListener('click', stop);
}

// ── Transcript helpers ──
function addLine(role) {
  const line = document.createElement('div');
  line.className = 'call-line call-' + role;
  const who = document.createElement('span');
  who.className = 'call-who';
  who.textContent = role === 'user' ? 'You' : 'AI';
  const body = document.createElement('span');
  body.className = 'call-text';
  line.appendChild(who);
  line.appendChild(body);
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return body;
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

// Remove reasoning from a block of text so ONLY the answer remains.
// Mirror of the server's src/text_helpers.strip_think(prose=True), tuned for
// the voice call: the reply is LLM-only output, so the "reasoning prose"
// heuristic is safe to apply (it would false-positive on user text elsewhere,
// but never here). This guarantees thinking — wrapped in <think> tags OR
// emitted as untagged chain-of-thought — is stripped from what the call
// shows and speaks. Decoupled from the backend `thinking` flag entirely:
// even if the model flags its whole stream (reasoning + answer) as
// thinking:true, this recovers the clean answer from the full text.
function stripForCall(text) {
  let t = text || '';
  if (!t) return '';
  // Normalize <thought>…</thought> and Gemma <|channel>thought…<channel|>
  // wrappers into canonical <think>…</think> markup (matches the server).
  t = t.replace(/<thought(\s[^<>]*)?>/gi, '<think$1>');
  t = t.replace(/<\/thought>/gi, '</think>');
  t = t.replace(/<\|channel>thought\s*\n?([\s\S]*?)<channel\|>/gi,
    (m, inner) => (inner && inner.trim()) ? `<think>${inner.trim()}</think>\n` : '');
  t = t.replace(/<\|channel>response\s*\n?/gi, '');
  t = t.replace(/<channel\|>/gi, '');
  // Normalize attribute-bearing tags (<think time="0.42">) to plain <think>.
  t = t.replace(/<think\s[^<>]*>/gi, '<think>');
  t = t.replace(/<\/think\s[^<>]*>/gi, '</think>');
  // Remove closed <think>…</think> blocks, then any dangling opener to EOF,
  // then stray tags. Forward-only-ish regexes; input is short LLM output.
  t = t.replace(/<think(?:ing)?(?:\s[^<>]*)?>[\s\S]*?<\/think(?:ing)?>/gi, '');
  t = t.replace(/<think(?:ing)?(?:\s[^<>]*)?>[\s\S]*$/gi, '');
  t = t.replace(/<\/?think(?:ing)?[^<>]*>\s*/gi, '');
  // Qwen "Thinking Process:" block and leaked prompt echoes.
  t = t.replace(/^Thinking Process:[\s\S]*?(?=\n\n#|\n\n\*\*|\Z)/i, '');
  t = t.replace(/^The user asks:[\s\S]*?(?=\n\n#|\n\n\*\*[A-Z]|\Z)/, '');
  t = t.replace(/^We need to[\s\S]*?(?=\n\n#|\n\n\*\*[A-Z]|\Z)/, '');
  // Untagged chain-of-thought: strip a leading run of reasoning paragraphs.
  t = _stripReasoningProse(t);
  return t.replace(/^\s*\n\s*\n/, '').trim();
}

// Strip a leading contiguous run of "reasoning prose" paragraphs (mirrors the
// server's _strip_reasoning_prose). Only removes from the TOP until the first
// non-reasoning paragraph, so a trailing reasoning-style sentence never eats
// the real answer above it.
function _stripReasoningProse(text) {
  const trimmed = (text || '').trim();
  if (!trimmed) return text;
  const paras = trimmed.split(/\n\s*\n/);
  if (paras.length <= 1) return text;
  const RE = /^\s*(?:the user (?:wants|is|asks|needs|wrote|said|told|messaged|requested)|i (?:need|should|have|'ll|will|am going)(?: to)? (?:write|draft|reply|respond|read|check|look|review|consider|think|provide|generate|produce|craft|compose|acknowledge|summarize|answer|give|keep|aim|make|address|focus|use|just|simply|analyze|format|create|build|note|decide)|let me (?:think|look|see|check|read|review|consider|draft|write|analyze|format|summarize|create|produce|craft|note|extract|identify|figure)|looking at (?:the|this|that)|(?:okay|alright|hmm|right|so|well|first|next|now)[,.]?\s+(?:the|i|let|so|now|this|here)|based on (?:the|this|what|context)|to (?:draft|write|reply|respond|summarize|answer))\b/i;
  let firstKeep = 0;
  for (let i = 0; i < paras.length; i++) {
    if (RE.test(paras[i])) firstKeep = i + 1;
    else break;
  }
  if (firstKeep === 0) return text;
  const keep = paras.slice(firstKeep);
  return keep.length ? keep.join('\n\n').trim() : text;
}


// ── Speech recognition (browser STT) ──
function startRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return false;
  try {
    recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = '';
    recognition.onresult = (event) => {
      // While muted (AI is speaking) discard everything — the mic is hearing
      // the speaker, not the user, and that must never become a transcript.
      if (_micMuted) { finalTranscript = ''; interimTranscript = ''; return; }
      interimTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalTranscript += t + ' ';
        else interimTranscript += t;
      }
      // Live preview of what the user is saying.
      if (state === STATE.LISTENING) {
        const live = finalTranscript.slice(turnPrefixLen) + interimTranscript;
        if (live.trim()) setStatus('Listening… "' + live.trim().slice(0, 60) + '"');
      }
    };
    recognition.onerror = (e) => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        setStatus('Microphone blocked for speech recognition');
      }
      // 'no-speech' / 'aborted' are benign — onend handles restart.
    };
    recognition.onend = () => {
      // The engine stopped on its own. If the call is still live, restart it so
      // the next utterance is captured.
      if (callActive && recognition && recognition._dead !== true) {
        try { recognition.start(); } catch (e) { /* will retry on next tick */ }
      }
    };
    recognition._dead = false;
    recognition.start();
    return true;
  } catch (e) {
    console.warn('SpeechRecognition start failed:', e);
    return false;
  }
}

function stopRecognition() {
  if (recognition) {
    recognition._dead = true;
    try { recognition.stop(); } catch (e) { /* ignore */ }
    recognition = null;
  }
}

// ── State transitions ──
function setMicMuted(muted) {
  _micMuted = muted;
  // While the AI speaks, its voice can leak into the mic. Pause VAD noise-floor
  // adaptation so that echo doesn't drag the floor up and make the call deaf to
  // the user's next turn.
  if (vad) vad.setFloorPaused(muted);
}

function setState(next, statusText) {
  state = next;
  if (statusText) setStatus(statusText);
  // Let app.js reflect the call state on the Send button (icon + active style).
  if (window._aigenchainCallUI) window._aigenchainCallUI(next !== STATE.IDLE);
}

function beginListening() {
  if (!callActive) return;
  setMicMuted(false); // arm the mic again for the user's next turn
  finalTranscript = '';
  interimTranscript = '';
  turnPrefixLen = 0;
  setState(STATE.LISTENING, 'Listening…');
}

function _currentUserText() {
  const t = (finalTranscript.slice(turnPrefixLen) + interimTranscript).trim();
  return t;
}

// VAD: end of the user's utterance → send to the model.
function onUserSilence() {
  if (state !== STATE.LISTENING) return;
  if (_micMuted) return; // AI still speaking — ignore any captured audio
  const text = _currentUserText();
  if (!text) {
    // Heard silence but no speech captured — keep listening.
    return;
  }
  sendToChat(text);
}

// VAD: real speech started (barge-in) during THINKING or SPEAKING.
function onSpeechStart() {
  if (_micMuted) return; // AI is speaking — ignore its own echo as barge-in
  if (state === STATE.LISTENING) return; // already listening
  if (state === STATE.IDLE) return;
  interrupt();
}

function interrupt() {
  if (abortCtrl) { try { abortCtrl.abort(); } catch (e) {} abortCtrl = null; }
  if (_ttsWaitIv) { clearInterval(_ttsWaitIv); _ttsWaitIv = null; }
  if (window.aiTTSManager) window.aiTTSManager.stop();
  beginListening();
}

// ── Chat streaming (with sentence-by-sentence TTS) ──
let _aiSpeakingStarted = false;

async function sendToChat(text) {
  if (!callActive) return;
  setMicMuted(true); // AI's turn now — ignore the mic until it finishes
  setState(STATE.THINKING, 'Thinking…');
  _aiSpeakingStarted = false;

  // Render the user's line in the transcript.
  const userBody = addLine('user');
  userBody.textContent = text;

  const sessionId = sessionModule.getCurrentSessionId && sessionModule.getCurrentSessionId();
  if (!sessionId) {
    setStatus('No active chat session — start a chat first');
    beginListening();
    return;
  }

  const fd = new FormData();
  fd.append('message', text);
  fd.append('session', sessionId);
  fd.append('mode', 'chat');

  abortCtrl = new AbortController();
  aiFullText = '';
  aiLineEl = null; // created lazily on the first clean answer token

  const _tzOffsetMin = -new Date().getTimezoneOffset();
  let _tzName = '';
  try { _tzName = Intl.DateTimeFormat().resolvedOptions().timeZone || ''; } catch (e) {}

  let _nextErr = false;
  try {
    const res = await fetch('/api/chat_stream', {
      method: 'POST',
      body: fd,
      headers: { 'X-Tz-Offset': String(_tzOffsetMin), 'X-Tz-Name': _tzName },
      signal: abortCtrl.signal,
    });

    if (!res.ok) {
      _streamError('Error ' + res.status);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          if (line.slice(7).trim() === 'error') _nextErr = true;
          continue;
        }
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]' || data === '') continue;
        let json;
        try { json = JSON.parse(data); } catch (e) { continue; }
        if (_nextErr || (json.status && json.status >= 400)) {
          _nextErr = false;
          _streamError(json.text || json.error?.message || 'Error');
          return;
        }
        if (json.delta) {
          // Accumulate ONLY the answer tokens. Reasoning/thinking-flagged
          // deltas (thinking:true) are skipped outright — the call must never
          // show or read the thinking process aloud, whether the model wraps
          // it in <think> tags OR streams it flag-only. Any residual wrapped
          // thinking is still stripped by stripForCall() at the end as a backstop.
          // We still deliberately do NOT render/speak mid-stream: the answer is
          // revealed and spoken ONLY at stream end, where separation is final.
          if (!json.thinking) aiFullText += json.delta;
        }
        if (json.type === 'done' || json.done) break;
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') return; // interrupted by barge-in / end call
    _streamError('Connection error');
    return;
  }

  const reply = stripForCall(aiFullText);
  if (!reply) {
    beginListening();
    return;
  }

  // Reveal the final (clean) answer in the transcript. In the normal case the
  // line was already created live; in the no-flag case (model tagged the whole
  // stream as thinking) it's created here — reasoning is never shown live.
  if (!aiLineEl) {
    aiLineEl = addLine('ai');
    setStatus('AI is responding…');
  }
  aiLineEl.textContent = reply;

  if (_callTTSOn && window.aiTTSManager) {
    // If we never streamed live there was no confirmed answer to speak, so the
    // speech engine was never started. Start it now and speak the full clean
    // reply at once. (If it WAS started live, streamingEnd only speaks the
    // remainder, so we must NOT start again or the answer repeats.)
    if (!_aiSpeakingStarted) {
      _aiSpeakingStarted = true;
      window.aiTTSManager.streamingStart();
      setState(STATE.THINKING, 'AI is speaking…');
    }
    window.aiTTSManager.streamingEnd(reply);
    _onAiSpoken();
  } else {
    // No browser TTS available — pause so the user can read, then listen again.
    setTimeout(() => { if (callActive) beginListening(); }, 1200);
  }
}

function _streamError(msg) {
  if (aiLineEl) aiLineEl.textContent = '⚠ ' + msg;
  // Stay in the call — go back to listening for the next turn.
  beginListening();
}

// Wait for the streamed TTS queue to finish, then return to listening.
function _onAiSpoken() {
  const mgr = window.aiTTSManager;
  if (!mgr) { setTimeout(() => { if (callActive) beginListening(); }, 1200); return; }
  _ttsWaitIv = setInterval(() => {
    if (!mgr.isPlaying && !mgr._processing) {
      clearInterval(_ttsWaitIv); _ttsWaitIv = null;
      // Cooldown before re-arming the mic: the tail of the AI's last spoken
      // words can still be ringing in the room, and unmuting instantly would
      // capture it as the user's next turn (the repeat loop).
      setTimeout(() => { if (callActive) beginListening(); }, POST_SPEECH_COOLDOWN_MS);
    }
  }, 150);
  // Safety valve so the call never hangs on a stuck utterance.
  setTimeout(() => {
    if (_ttsWaitIv) { clearInterval(_ttsWaitIv); _ttsWaitIv = null; if (callActive) beginListening(); }
  }, 25000);
}

// ── Call TTS: reuse the existing streaming TTS manager (browser TTS) ──
// We temporarily force browser TTS + autoPlay so the reply streams
// sentence-by-sentence, then restore the user's global TTS setting on hang-up.
function _enableCallTTS() {
  const mgr = window.aiTTSManager;
  if (!mgr) return false;
  _ttsPrev = {
    available: mgr.available,
    useBrowserTTS: mgr.useBrowserTTS,
    autoPlay: mgr.autoPlay,
    browserVoice: mgr.browserVoice,
    playbackSpeed: mgr.playbackSpeed,
  };
  mgr.available = true;
  // The call works with any TTS provider. Only force the in-browser Web Speech
  // engine when the configured provider is "browser". For server-side providers
  // (piper / endpoint / local) we keep the manager's server-synthesis flag so
  // the call streams audio fetched from /api/tts/synthesize. Browser TTS is the
  // lowest-latency option, but it has no offline Indonesian voice, so server
  // Piper is the better choice on hardware like an Intel MacBook.
  if (mgr._provider === 'browser') {
    if (!('speechSynthesis' in window)) return false; // no browser TTS engine
    mgr.useBrowserTTS = true;
    // Honor the user's Voice Call settings — a different speaker than read-aloud.
    if (typeof window._callVoice === 'string' && window._callVoice) mgr.browserVoice = window._callVoice;
    if (typeof window._callSpeed === 'string' && window._callSpeed) mgr.playbackSpeed = parseFloat(window._callSpeed) || 1;
  }
  mgr.autoPlay = true;
  _callTTSOn = true;
  return true;
}

function _disableCallTTS() {
  const mgr = window.aiTTSManager;
  if (mgr) mgr.stop();
  if (mgr && _ttsPrev) {
    mgr.available = _ttsPrev.available;
    mgr.useBrowserTTS = _ttsPrev.useBrowserTTS;
    mgr.autoPlay = _ttsPrev.autoPlay;
    mgr.browserVoice = _ttsPrev.browserVoice;
    mgr.playbackSpeed = _ttsPrev.playbackSpeed;
  }
  _ttsPrev = null;
  _callTTSOn = false;
}

// ── Start / stop the whole call ──
async function start() {
  if (state !== STATE.IDLE) return;

  // Browser STT availability check.
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    if (uiModule && uiModule.showToast) {
      uiModule.showToast('Voice call needs Chrome or Edge (Web Speech API)', 4000);
    } else {
      alert('Voice call needs Chrome or Edge (Web Speech API)');
    }
    return;
  }

  if (!window.isSecureContext) {
    if (uiModule && uiModule.showError) {
      uiModule.showError('Microphone requires HTTPS or localhost.');
    }
    return;
  }

  callActive = true;
  // Refresh the TTS manager's view of the configured provider so the call uses
  // whatever is set in Settings (browser, piper, endpoint, …) rather than a
  // stale value from page load.
  if (window.aiTTSManager) { try { await window.aiTTSManager.checkAvailability(); } catch (e) {} }
  _enableCallTTS(); // force browser TTS streaming for the duration of the call

  // Open the mic for VAD metering.
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    callActive = false;
    if (uiModule && uiModule.showError) {
      uiModule.showError('Microphone access denied.');
    }
    return;
  }

  vad = new VAD();
  vad.onSilence = (remaining) => { /* reserved for future "about to stop" UI */ };
  vad.onSpeechEnd = onUserSilence;
  vad.onSpeechStart = onSpeechStart;
  vad.start(micStream);

  // Show overlay.
  transcriptEl.innerHTML = '';
  overlay.classList.remove('hidden');
  callStartTime = Date.now();
  timerInterval = setInterval(updateTimer, 1000);
  updateTimer();

  startRecognition();
  beginListening();
}

function stop() {
  if (state === STATE.IDLE && !callActive) return;
  callActive = false;

  if (abortCtrl) { try { abortCtrl.abort(); } catch (e) {} abortCtrl = null; }
  if (_ttsWaitIv) { clearInterval(_ttsWaitIv); _ttsWaitIv = null; }
  stopRecognition();
  if (vad) { vad.stop(); vad = null; }
  _disableCallTTS();
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }

  setState(STATE.IDLE, 'Call ended');

  // Best-effort: refresh the session list so the call transcript shows up in history.
  try { if (sessionModule.loadSessions) sessionModule.loadSessions(); } catch (e) {}

  // Hide the overlay shortly after, so the user can read the final transcript.
  setTimeout(() => {
    if (!callActive && overlay) overlay.classList.add('hidden');
  }, 600);
}

function updateTimer() {
  if (!timerEl) return;
  const secs = Math.floor((Date.now() - callStartTime) / 1000);
  const m = String(Math.floor(secs / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  timerEl.textContent = `${m}:${s}`;
}

// ── Init ──
function init() {
  buildUI();
  window.callController = { start, stop, isActive: () => state !== STATE.IDLE };
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

export default { start, stop, init };
