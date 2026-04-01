"""Thor Voice Satellite — Raspberry Pi Client"""

import base64
import collections
import json
import struct
import time
import urllib.request

import ctypes
import ctypes.util

# Suppress ALSA warnings during PyAudio init
_ERROR_HANDLER_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
_error_handler = _ERROR_HANDLER_TYPE(lambda *_: None)  # prevent GC
_alsa = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound"))
_alsa.snd_lib_error_set_handler(_error_handler)

import numpy as np
import pyaudio
from openwakeword.model import Model as WakewordModel

# ---------------------------------------------------------------------------
# Config — anpassen pro Raum
# ---------------------------------------------------------------------------
ROOM = "wohnzimmer"  # ÄNDERN: küche, schlafzimmer, büro, etc.
THOR_URL = "http://192.168.50.165:9000/api/voice"
WAKEWORD_MODEL = "hey_jarvis_v0.1"  # Oder custom "hey_thor" Modell
WAKEWORD_MODEL_PATH = None  # Auto-resolved from openwakeword package
WAKEWORD_THRESHOLD = 0.7
SAMPLE_RATE = 16000
CHANNELS = 2  # XVF3800 liefert 2ch: CH0=AEC ref, CH1=ASR beam
ASR_CHANNEL = 1  # Beamformed + noise-suppressed output
CHUNK_MS = 80  # OpenWakeWord optimal window
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
RINGBUFFER_SECONDS = 5
SILENCE_TIMEOUT_MS = 1500  # Sprechpause = Ende des Befehls
SILENCE_THRESHOLD = 500  # RMS-Schwellwert für Stille
# Mikrofon-Device-Index (None = dynamisch per find_respeaker)
MIC_DEVICE_INDEX = None

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


def send_to_thor(audio_bytes: bytes) -> dict | None:
    """Send audio to Thor server and return response."""
    payload = {
        "room": ROOM,
        "audio_b64": base64.b64encode(audio_bytes).decode(),
        "sample_rate": SAMPLE_RATE,
    }
    req = urllib.request.Request(
        THOR_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except Exception as e:
        print(f"[ERROR] Thor unreachable: {e}")
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    print(f"=== Thor Voice Satellite [{ROOM}] ===")
    print(f"Thor: {THOR_URL}")
    print(f"Wakeword: {WAKEWORD_MODEL} (threshold {WAKEWORD_THRESHOLD})")

    # Resolve wakeword model path
    if WAKEWORD_MODEL_PATH:
        model_path = WAKEWORD_MODEL_PATH
    else:
        import openwakeword
        for p in openwakeword.get_pretrained_model_paths():
            if WAKEWORD_MODEL in p:
                model_path = p
                break
        else:
            raise FileNotFoundError(f"Wakeword model '{WAKEWORD_MODEL}' not found in pretrained models")

    # Init wakeword model
    oww = WakewordModel(wakeword_model_paths=[model_path])
    print(f"OpenWakeWord loaded: {model_path}")

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
    print(f"Mikrofon: {mic_name} (index={mic_index}, {SAMPLE_RATE}Hz, {CHUNK_MS}ms chunks)")

    # Ringbuffer: hält die letzten N Sekunden
    max_chunks = int(RINGBUFFER_SECONDS * 1000 / CHUNK_MS)
    ringbuffer = collections.deque(maxlen=max_chunks)

    print("Lausche...")

    try:
      while True:
        raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        mono = extract_channel(raw)
        ringbuffer.append(mono)

        # Feed to wakeword
        audio_np = np.frombuffer(mono, dtype=np.int16)
        scores = oww.predict(audio_np)

        # Check all wakeword models
        for name, current_score in scores.items():
            if current_score > WAKEWORD_THRESHOLD:
                print(f"\n>>> Wakeword '{name}' erkannt! (score={current_score:.2f})")
                oww.reset()

                # Weiter aufnehmen bis Sprechpause
                silence_ms = 0
                while silence_ms < SILENCE_TIMEOUT_MS:
                    raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    mono = extract_channel(raw)
                    ringbuffer.append(mono)
                    vol = rms(mono)
                    if vol < SILENCE_THRESHOLD:
                        silence_ms += CHUNK_MS
                    else:
                        silence_ms = 0

                # Ringbuffer → Audio bytes
                audio_bytes = b"".join(ringbuffer)
                duration = len(audio_bytes) / (SAMPLE_RATE * 2)
                print(f"    Aufnahme: {duration:.1f}s, sende an Thor...")

                # Send to Thor
                t0 = time.time()
                result = send_to_thor(audio_bytes)
                elapsed = time.time() - t0

                if result:
                    print(f"    Transkript: {result['transcript']}")
                    print(f"    Antwort:    {result['response_text']}")
                    print(f"    Zeiten:     {result['timings']}")
                    print(f"    E2E:        {elapsed:.2f}s")

                    # TODO: Audio-Antwort abspielen wenn verfügbar
                    if result.get("audio_b64"):
                        print("    [Audio-Antwort empfangen, Playback TODO]")
                else:
                    print("    [Keine Antwort von Thor]")

                print("\nLausche...")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
