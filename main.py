"""Thor Voice Satellite — Raspberry Pi Client (Streaming)"""

import asyncio
import base64
import collections
import json
import logging
import os
import struct
import threading
import time
import wave

import ctypes
import ctypes.util

# Suppress ALSA warnings during PyAudio init
_ERROR_HANDLER_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
_error_handler = _ERROR_HANDLER_TYPE(lambda *_: None)  # prevent GC
_alsa = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound"))
_alsa.snd_lib_error_set_handler(_error_handler)

import numpy as np
import onnxruntime as ort
import pyaudio
import usb.core
import websockets
from openwakeword.model import Model as OWWModel
from openwakeword.utils import AudioFeatures

log = logging.getLogger("satellite")

# ---------------------------------------------------------------------------
# Config — anpassen pro Raum
# ---------------------------------------------------------------------------
ROOM = "kueche"
THOR_HTTP_URL = "http://192.168.50.163:9000/api/voice"
THOR_WS_URL = "ws://192.168.50.163:9000/api/voice/stream"
WAKEWORD_ACK_URL = "http://192.168.50.163:9000/api/assets/wakeword-ack.wav"
WAKEWORD_ACK_PATH = os.path.join(os.path.dirname(__file__), "wakeword-ack.wav")
WAKEWORD_MODEL = "hey_bernd"
WAKEWORD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hey_bernd.onnx")
WAKEWORD_THRESHOLD = 0.65
WAKEWORD_N_FRAMES = 13  # frames the model expects
SAMPLE_RATE = 16000
CHANNELS = 2  # XVF3800 liefert 2ch: CH0=AEC ref, CH1=ASR beam
ASR_CHANNEL = 1  # Beamformed + noise-suppressed output
CHUNK_MS = 80  # OpenWakeWord optimal window
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
RINGBUFFER_SECONDS = 2
SILENCE_TIMEOUT_MS = 900  # Sprechpause = Ende des Befehls
INITIAL_SILENCE_TIMEOUT_MS = 2000  # Warten auf ersten Sprechbeginn nach Wakeword
SILENCE_DROP_FACTOR = 0.8  # Silence = RMS < 80% of wakeword baseline
MAX_RECORD_MS = 15000  # Hard limit for recording
# Mikrofon-Device-Index (None = dynamisch per find_respeaker)
MIC_DEVICE_INDEX = None
# Speaker: reSpeaker output = 16kHz stereo
SPEAKER_RATE = 16000
SPEAKER_CHANNELS = 2
SPEAKER_GAIN = 1.0  # No software boost, HW volume at max

# ---------------------------------------------------------------------------
# LED control (XVF3800 USB control transfers, resource_id=20)
# ---------------------------------------------------------------------------
LED_RESID = 20
LED_TIMEOUT = 100000

class LedRing:
    """Control the XVF3800 WS2812 LED ring."""

    def __init__(self):
        self._dev = usb.core.find(idVendor=0x2886, idProduct=0x001a)
        if self._dev is None:
            log.warning("XVF3800 USB device not found, LEDs disabled")
        else:
            log.info("LED ring connected")

    def _set(self, cmdid, data):
        if self._dev is None:
            return
        try:
            self._dev.ctrl_transfer(0x40, 0, cmdid, LED_RESID, data, LED_TIMEOUT)
        except Exception as e:
            log.warning("LED write failed: %s", e)

    def solid(self, rgb):
        self._set(16, struct.pack('<I', rgb))
        self._set(12, [3])

    def breath(self, rgb, speed=4, brightness=200):
        self._set(16, struct.pack('<I', rgb))
        self._set(13, [brightness])
        self._set(15, [speed])
        self._set(12, [1])

    def flash(self, rgb, times=2, on_ms=200, off_ms=150):
        for _ in range(times):
            self.solid(rgb)
            time.sleep(on_ms / 1000)
            self.off()
            time.sleep(off_ms / 1000)

    def off(self):
        self._set(12, [0])


# ---------------------------------------------------------------------------
# Custom Wakeword Detector
# ---------------------------------------------------------------------------
class WakewordDetector:
    """Wakeword detection using AudioFeatures embeddings + custom ONNX classifier."""

    # embed_clips needs >= 76 mel frames = ~1.2s of audio at 16kHz
    MIN_AUDIO_SAMPLES = int(1.3 * SAMPLE_RATE)

    PREDICT_EVERY_N = 5  # Only run embedding every N chunks (~400ms at 80ms chunks)

    def __init__(self, model_path: str, n_frames: int = WAKEWORD_N_FRAMES):
        self._af = AudioFeatures()
        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self._n_frames = n_frames
        self._audio_buffer = collections.deque(maxlen=int(2.0 * SAMPLE_RATE / CHUNK_SIZE))
        self._input_name = self._session.get_inputs()[0].name
        self._chunk_count = 0
        self._last_score = 0.0
        log.info("WakewordDetector loaded: %s (%d frames, predict every %dms)",
                 model_path, n_frames, self.PREDICT_EVERY_N * CHUNK_MS)

    def predict(self, audio_chunk: np.ndarray) -> float:
        """Feed a mono int16 audio chunk, return wakeword score (0-1)."""
        self._audio_buffer.append(audio_chunk.copy())
        self._chunk_count += 1

        # Only run expensive embedding every N chunks
        if self._chunk_count % self.PREDICT_EVERY_N != 0:
            return self._last_score

        # Need enough audio for embedding extraction
        total_samples = sum(len(c) for c in self._audio_buffer)
        if total_samples < self.MIN_AUDIO_SAMPLES:
            return 0.0

        # Concatenate sliding window of audio
        audio = np.concatenate(list(self._audio_buffer))
        emb = self._af.embed_clips(audio.reshape(1, -1))

        if emb is None or emb.shape[1] < self._n_frames:
            return 0.0

        # Take last n_frames
        frames = emb[0, -self._n_frames:, :].astype(np.float32)
        inp = frames.reshape(1, self._n_frames, 96)
        logit = self._session.run(None, {self._input_name: inp})[0]
        self._last_score = float(1.0 / (1.0 + np.exp(-logit.flatten()[0])))
        return self._last_score

    def reset(self):
        """Clear internal buffers after a detection."""
        self._audio_buffer.clear()
        self._chunk_count = 0
        self._last_score = 0.0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def find_respeaker(pa: pyaudio.PyAudio) -> int:
    """Find ReSpeaker XVF3800 device index dynamically."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "xvf3800" in info["name"].lower() or "respeaker" in info["name"].lower():
            return i
    raise RuntimeError("ReSpeaker XVF3800 not found")


def extract_channel(stereo_chunk: bytes, channel: int = ASR_CHANNEL) -> bytes:
    """Extract a single channel from interleaved stereo int16 audio."""
    samples = np.frombuffer(stereo_chunk, dtype=np.int16)
    return samples[channel::CHANNELS].tobytes()


def rms(audio_chunk: bytes) -> float:
    """Calculate RMS volume of mono audio chunk."""
    samples = np.frombuffer(audio_chunk, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def resample_to_speaker(samples: np.ndarray, src_rate: int) -> bytes:
    """Resample mono int16 samples to SPEAKER_RATE stereo int16 bytes."""
    if src_rate != SPEAKER_RATE:
        ratio = SPEAKER_RATE / src_rate
        indices = np.arange(0, len(samples), 1 / ratio).astype(int)
        indices = indices[indices < len(samples)]
        samples = samples[indices]
    # Boost + clip to int16 range
    boosted = np.clip(samples.astype(np.float32) * SPEAKER_GAIN, -32768, 32767).astype(np.int16)
    # Mono to stereo interleaved
    stereo = np.column_stack([boosted, boosted]).flatten()
    return stereo.tobytes()


def load_wakeword_ack() -> bytes:
    """Load wakeword-ack.wav, resample to speaker format, return raw PCM bytes."""
    # Download if missing
    if not os.path.exists(WAKEWORD_ACK_PATH):
        log.info("Downloading wakeword-ack.wav...")
        import urllib.request
        urllib.request.urlretrieve(WAKEWORD_ACK_URL, WAKEWORD_ACK_PATH)

    with wave.open(WAKEWORD_ACK_PATH, 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
        sr = wf.getframerate()

    samples = np.frombuffer(raw, dtype=np.int16)
    pcm = resample_to_speaker(samples, sr)
    # Prepend 300ms silence so ACK doesn't fire too fast
    silence = b'\x00' * int(SPEAKER_RATE * SPEAKER_CHANNELS * 2 * 0.3)
    pcm = silence + pcm
    log.info("Wakeword ACK loaded: %.2fs (incl 300ms delay)", len(samples) / sr + 0.3)
    return pcm


# ---------------------------------------------------------------------------
# Speaker: non-blocking playback with interrupt support
# ---------------------------------------------------------------------------
class Speaker:
    """Persistent audio output on reSpeaker. Supports interrupt for new wakeword."""

    def __init__(self, pa: pyaudio.PyAudio, device_index: int):
        self._pa = pa
        self._device_index = device_index
        self._stream = None
        self._stop_event = threading.Event()
        self._play_thread = None
        log.info("Speaker ready (device=%d, %dHz, %dch)",
                 device_index, SPEAKER_RATE, SPEAKER_CHANNELS)

    def _ensure_stream(self):
        if self._stream is None:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=SPEAKER_CHANNELS,
                rate=SPEAKER_RATE,
                output=True,
                output_device_index=self._device_index,
            )

    def play_async(self, pcm_bytes: bytes):
        """Play PCM bytes in background thread. Non-blocking. Interrupts previous."""
        self.stop()
        self._stop_event.clear()
        self._started = threading.Event()
        self._play_thread = threading.Thread(
            target=self._play, args=(pcm_bytes,), daemon=True
        )
        self._play_thread.start()
        self._started.wait()  # ensure thread is running before returning

    def _play(self, pcm_bytes: bytes):
        try:
            self._ensure_stream()
            self._started.set()
            chunk_bytes = SPEAKER_RATE * SPEAKER_CHANNELS * 2 // 10  # 100ms
            offset = 0
            while offset < len(pcm_bytes) and not self._stop_event.is_set():
                end = min(offset + chunk_bytes, len(pcm_bytes))
                self._stream.write(pcm_bytes[offset:end])
                offset = end
        except Exception as e:
            log.warning("Speaker error: %s", e)

    def write(self, pcm_bytes: bytes):
        """Write PCM directly to output stream. Blocks until played."""
        self._ensure_stream()
        self._stream.write(pcm_bytes)

    def stop(self):
        """Stop any active playback."""
        self._stop_event.set()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=1)

    @property
    def playing(self) -> bool:
        return self._play_thread is not None and self._play_thread.is_alive()

    def close(self):
        self.stop()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()


# ---------------------------------------------------------------------------
# Persistent WebSocket connection to Thor
# ---------------------------------------------------------------------------
class ThorConnection:
    """Persistent WebSocket connection to Thor with auto-reconnect."""

    def __init__(self, loop: asyncio.AbstractEventLoop, speaker: Speaker):
        self._loop = loop
        self._ws = None
        self._speaker = speaker

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._ws.state == websockets.protocol.State.OPEN

    def ensure_connected(self):
        """Connect if not already connected."""
        if not self.connected:
            self._loop.run_until_complete(self._connect())

    async def _connect(self):
        try:
            self._ws = await websockets.connect(
                THOR_WS_URL,
                open_timeout=5,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
            )
            log.info("Thor WS connected")
        except (OSError, websockets.exceptions.WebSocketException) as e:
            log.warning("Thor WS connect failed: %s", e)
            self._ws = None

    def stream(self, pre_trigger: list[bytes], audio_stream,
               led: LedRing, silence_threshold: float) -> tuple[dict | None, bool]:
        """Stream a voice command to Thor. Returns (result_dict, success)."""
        try:
            return self._loop.run_until_complete(
                self._stream(pre_trigger, audio_stream, led, silence_threshold)
            )
        except Exception as e:
            log.warning("WS stream error: %s", e)
            self._ws = None
            return None, False

    async def _stream(self, pre_trigger, audio_stream, led, silence_threshold):
        if not self.connected:
            await self._connect()
        if not self.connected:
            return None, False

        # Quick health check
        try:
            pong = await asyncio.wait_for(self._ws.ping(), timeout=2)
            await pong
        except Exception:
            log.debug("WS stale, reconnecting")
            await self._connect()
            if not self.connected:
                return None, False

        # Signal start of new utterance
        await self._ws.send(json.dumps({
            "type": "start",
            "room": ROOM,
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "channels": 1,
            "chunk_ms": CHUNK_MS,
        }))

        # Send pre-trigger audio
        for chunk in pre_trigger:
            await self._ws.send(chunk)
        log.debug("Sent %d pre-trigger chunks", len(pre_trigger))

        # Stream live audio until silence
        silence_ms = 0
        speech_detected = False
        timeout = INITIAL_SILENCE_TIMEOUT_MS
        total_ms = len(pre_trigger) * CHUNK_MS
        while silence_ms < timeout and total_ms < MAX_RECORD_MS:
            raw = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            mono = extract_channel(raw)
            await self._ws.send(mono)
            total_ms += CHUNK_MS

            vol = rms(mono)
            if vol < silence_threshold:
                silence_ms += CHUNK_MS
            else:
                silence_ms = 0
                if not speech_detected:
                    speech_detected = True
                    timeout = SILENCE_TIMEOUT_MS
                    log.debug("Speech detected, timeout -> %dms", timeout)

        # Signal end of audio
        await self._ws.send(json.dumps({"type": "end"}))
        log.info("Stream complete: %.1fs sent", total_ms / 1000)

        led.breath(0x0000FF)  # blue breath: waiting for response

        # Receive response: text result, then streamed TTS audio
        result = None
        audio_sr = None
        leftover = b''  # buffer for incomplete int16 samples
        pending_write = None  # track last speaker.write future
        total_audio_bytes = 0
        total_playback_bytes = 0

        while True:
            msg = await asyncio.wait_for(self._ws.recv(), timeout=30)

            if isinstance(msg, str):
                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "audio_start":
                    audio_sr = data.get("sample_rate", 44100)
                    leftover = b''
                    # Prime speaker with 50ms silence to avoid cold-start underrun
                    prime = b'\x00' * int(SPEAKER_RATE * SPEAKER_CHANNELS * 2 * 0.05)
                    self._speaker.write(prime)
                    log.info("TTS audio stream starting (sr=%d)", audio_sr)

                elif msg_type == "audio_end":
                    # Wait for last speaker write to finish
                    if pending_write is not None:
                        await pending_write
                    audio_dur = total_audio_bytes / (2 * (audio_sr or 44100))
                    playback_dur = total_playback_bytes / (SPEAKER_RATE * SPEAKER_CHANNELS * 2)
                    log.info("TTS playback complete (recv=%.1fs @%dHz, playback=%.1fs @%dHz)",
                             audio_dur, audio_sr or 44100, playback_dur, SPEAKER_RATE)
                    break

                elif "error" in data:
                    log.error("Thor error: %s", data["error"])
                    break

                else:
                    # Text result
                    result = data
                    log.info("Transkript: %s", result.get("transcript", ""))
                    log.info("Antwort: %s", result.get("response_text", ""))
                    log.info("Zeiten: %s", result.get("timings", ""))

            elif isinstance(msg, bytes) and audio_sr:
                # Prepend any leftover byte from previous chunk
                raw = leftover + msg
                # Ensure even number of bytes for int16
                usable = len(raw) - (len(raw) % 2)
                leftover = raw[usable:]
                if usable > 0:
                    total_audio_bytes += usable
                    samples = np.frombuffer(raw[:usable], dtype=np.int16).copy()
                    pcm = resample_to_speaker(samples, audio_sr)
                    total_playback_bytes += len(pcm)
                    # Run blocking write in thread to not block ws.recv()
                    if pending_write is not None:
                        await pending_write
                    pending_write = asyncio.get_event_loop().run_in_executor(
                        None, self._speaker.write, pcm
                    )

        return result, result is not None

    def close(self):
        if self._ws:
            self._loop.run_until_complete(self._ws.close())
            self._ws = None


def send_to_thor_http(audio_bytes: bytes) -> tuple[dict | None, int]:
    """Fallback: send audio to Thor via HTTP. Returns (response_dict, http_status)."""
    import urllib.request as urlreq
    payload = {
        "room": ROOM,
        "audio_b64": base64.b64encode(audio_bytes).decode(),
        "sample_rate": SAMPLE_RATE,
    }
    req = urlreq.Request(
        THOR_HTTP_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urlreq.urlopen(req, timeout=30)
        return json.loads(resp.read()), resp.status
    except Exception as e:
        log.error("Thor HTTP unreachable: %s", e)
        return None, 0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=== Thor Voice Satellite [%s] (Streaming) ===", ROOM)
    log.info("Thor WS: %s", THOR_WS_URL)
    log.info("Wakeword: %s (threshold %.1f)", WAKEWORD_MODEL, WAKEWORD_THRESHOLD)

    # Init LED ring
    led = LedRing()

    # Init wakeword detector
    oww = WakewordDetector(WAKEWORD_MODEL_PATH)
    log.info("Wakeword model: %s", WAKEWORD_MODEL_PATH)

    # Load wakeword ACK sound
    ack_pcm = load_wakeword_ack()

    # Init audio
    pa = pyaudio.PyAudio()
    mic_index = MIC_DEVICE_INDEX if MIC_DEVICE_INDEX is not None else find_respeaker(pa)
    mic_name = pa.get_device_info_by_index(mic_index)["name"]
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=CHUNK_SIZE,
    )
    log.info("Mikrofon: %s (index=%d, %dHz, %dms chunks)", mic_name, mic_index, SAMPLE_RATE, CHUNK_MS)

    # Speaker (share PyAudio instance)
    speaker = Speaker(pa, device_index=mic_index)

    # Ringbuffer
    max_chunks = int(RINGBUFFER_SECONDS * 1000 / CHUNK_MS)
    ringbuffer = collections.deque(maxlen=max_chunks)

    # Persistent Thor connection
    loop = asyncio.new_event_loop()
    thor = ThorConnection(loop, speaker=speaker)
    thor.ensure_connected()

    log.info("Lausche...")

    try:
      while True:
        raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        mono = extract_channel(raw)
        ringbuffer.append(mono)

        # Feed to wakeword
        audio_np = np.frombuffer(mono, dtype=np.int16)
        score = oww.predict(audio_np)

        if score > 0.1:
            log.debug("%s: %.3f", WAKEWORD_MODEL, score)
        if score > WAKEWORD_THRESHOLD:
                # Stop any active playback (interrupt for new command)
                speaker.stop()

                # Dynamic silence threshold from ringbuffer
                rms_values = [rms(c) for c in ringbuffer]
                baseline_rms = sum(rms_values) / len(rms_values) if rms_values else 500
                silence_threshold = baseline_rms * SILENCE_DROP_FACTOR
                log.info("Wakeword '%s' erkannt (score=%.2f, baseline=%.0f, thr=%.0f)",
                         WAKEWORD_MODEL, score, baseline_rms, silence_threshold)

                # Play ACK sound and drain mic while it plays
                speaker.play_async(ack_pcm)
                led.solid(0x00FF00)  # green: listening
                oww.reset()
                while speaker.playing:
                    stream.read(CHUNK_SIZE, exception_on_overflow=False)

                pre_trigger = list(ringbuffer)
                t0 = time.time()

                # Stream via persistent WS, fall back to HTTP
                result, ok = thor.stream(pre_trigger, stream, led, silence_threshold)

                if not ok:
                    log.info("WS failed, falling back to HTTP batch")
                    command_chunks = []
                    silence_ms = 0
                    speech_detected = False
                    fb_timeout = INITIAL_SILENCE_TIMEOUT_MS
                    total_ms = 0
                    while silence_ms < fb_timeout and total_ms < MAX_RECORD_MS:
                        raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        mono = extract_channel(raw)
                        command_chunks.append(mono)
                        total_ms += CHUNK_MS
                        vol = rms(mono)
                        if vol < silence_threshold:
                            silence_ms += CHUNK_MS
                        else:
                            silence_ms = 0
                            if not speech_detected:
                                speech_detected = True
                                fb_timeout = SILENCE_TIMEOUT_MS

                    audio_bytes = b"".join(pre_trigger + command_chunks)
                    duration = len(audio_bytes) / (SAMPLE_RATE * 2)
                    log.info("HTTP fallback: %.1fs audio", duration)
                    led.breath(0x0000FF)

                    result, status = send_to_thor_http(audio_bytes)
                    ok = status == 200 and result is not None

                elapsed = time.time() - t0

                if ok and result:
                    led.flash(0x00FF00, times=2)
                    log.info("E2E: %.2fs", elapsed)
                else:
                    led.flash(0xFF0000, times=2)
                    log.warning("Keine Antwort von Thor")

                # Flush
                ringbuffer.clear()
                stream.stop_stream()
                stream.start_stream()
                oww.reset()
                log.debug("flushed")

                # Reconnect WS if dropped
                if not thor.connected:
                    thor.ensure_connected()

                led.off()
                log.info("Lausche...")
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down...")
        speaker.close()
        thor.close()
        loop.close()
        led.off()
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
