# Thor Voice Satellite

Voice satellite client for the Thor smart home system. Runs on a Raspberry Pi 5 with a reSpeaker XVF3800 4-Mic Array, listens for a wakeword, streams audio to the Thor server via WebSocket, and plays back TTS responses.

## Hardware

- **Raspberry Pi 5** (admin@raspberry, 192.168.50.x, WiFi SSID: TMC)
- **Mikrofon:** Seeed reSpeaker XVF3800 4-Mic Array (USB, VID 0x2886 / PID 0x001a)
  - 2 Channels: CH0 = AEC reference, **CH1 = ASR-optimized beam** (beamformed + noise-suppressed)
  - Capture: 16kHz, 16-bit, stereo — we extract CH1 mono
  - Playback: 16kHz, 16-bit, stereo
  - 12x WS2812 LED ring, controlled via USB control transfers (resource_id=20)
  - udev rule in `/etc/udev/rules.d/99-respeaker.rules` for non-root USB access
- **Speaker:** reSpeaker onboard output (same USB device, 16kHz stereo)

## Thor Server

- **IP:** 192.168.50.163
- **WebSocket (primary):** `ws://192.168.50.163:9000/api/voice/stream` — persistent connection
- **HTTP (fallback):** `http://192.168.50.163:9000/api/voice`
- **Assets:** `http://192.168.50.163:9000/api/assets/wakeword-ack.wav`

## Architecture

```
Mikrofon (CH1 beam) → Ringbuffer (2s) → OpenWakeWord → Trigger
  → ACK sound abspielen (non-blocking)
  → Pre-trigger + live audio streamen via WebSocket
  → Silence detection (dynamisch, RMS-basiert) → end signal
  → Thor antwortet: Text JSON + TTS Audio stream
  → TTS resample 44100→16000 → reSpeaker abspielen
  → Zurück zu Lauschen
```

## WebSocket Protocol

Connection is persistent (opened once at startup, auto-reconnect on failure).

```
Satellite → Thor:  TEXT   {"type":"start", "room":"wohnzimmer", "sample_rate":16000,
                           "encoding":"pcm_s16le", "channels":1, "chunk_ms":80}
Satellite → Thor:  BINARY [2560 bytes per chunk, 80ms × 16kHz × 16bit]
Satellite → Thor:  BINARY ...
Satellite → Thor:  TEXT   {"type":"end"}

Thor → Satellite:  TEXT   {"transcript":"...", "response_text":"...", "domain":"...", "timings":{...}}
Thor → Satellite:  TEXT   {"type":"audio_start", "sample_rate":44100, "encoding":"pcm_s16le"}
Thor → Satellite:  BINARY [TTS PCM chunks, streamed as they arrive, may have gaps between sentences]
Thor → Satellite:  TEXT   {"type":"audio_end"}
```

## Key Components

### `main.py` — single-file satellite application

- **LedRing:** XVF3800 LED control via USB control transfers (pyusb). Effects: solid, breath, flash, off.
- **Speaker:** Persistent PyAudio output stream. Background playback thread with interrupt support (`stop()` kills active playback immediately for new wakeword).
- **ThorConnection:** Persistent WebSocket with auto-reconnect. Ping health check before each utterance. Falls back to HTTP batch if WS unavailable.
- **Wakeword ACK:** `wakeword-ack.wav` downloaded from Thor at startup, pre-resampled to 16kHz stereo PCM, stored in memory. 300ms silence prepended for natural timing.

### Audio Pipeline

1. PyAudio captures 2ch stereo at 16kHz in 80ms chunks (1280 samples = 2560 bytes per channel)
2. `extract_channel()` pulls CH1 (ASR beam) as mono
3. Mono chunks feed into OpenWakeWord and ringbuffer simultaneously
4. On wakeword trigger: ringbuffer snapshot = pre-trigger audio (2s)

### Silence Detection

- **Dynamic threshold:** RMS average of ringbuffer at trigger time × 0.8 = silence threshold. Adapts to room noise automatically.
- **Two-phase timeout:** Initial 2000ms to wait for speech start, then 900ms after first speech detected.
- **Hard limit:** 15s max recording.

### TTS Playback

- Thor streams 44100Hz mono PCM s16le chunks
- Satellite resamples to 16kHz (linear interpolation) and duplicates to stereo
- Chunks are buffered (≥100ms) then played via persistent Speaker stream
- Wakeword during playback: speaker.stop() interrupts immediately

### LED Feedback

| State | LED Effect | Color |
|-------|-----------|-------|
| Wakeword erkannt | Solid | Green |
| Warten auf Thor | Breath | Blue |
| Erfolg (200) | Flash ×2 | Green |
| Fehler | Flash ×2 | Red |
| Idle | Off | — |

LED registers (resource_id=20): EFFECT=12, BRIGHTNESS=13, SPEED=15, COLOR=16.
Modes: 0=off, 1=breath, 2=rainbow, 3=solid, 4=DOA, 5=ring.

## Config (top of main.py)

| Variable | Wert | Beschreibung |
|----------|------|-------------|
| `ROOM` | `"wohnzimmer"` | Raum-ID, an Thor gesendet |
| `WAKEWORD_MODEL` | `"hey_jarvis_v0.1"` | OpenWakeWord built-in model |
| `WAKEWORD_THRESHOLD` | `0.7` | Trigger-Schwelle |
| `SAMPLE_RATE` | `16000` | Mic capture rate |
| `CHUNK_MS` | `80` | OpenWakeWord optimal window |
| `RINGBUFFER_SECONDS` | `2` | Pre-trigger audio buffer |
| `SILENCE_TIMEOUT_MS` | `900` | Stille nach Sprache → Stop |
| `INITIAL_SILENCE_TIMEOUT_MS` | `2000` | Warten auf Sprechbeginn |
| `SILENCE_DROP_FACTOR` | `0.8` | Threshold = Baseline × 0.8 |
| `MAX_RECORD_MS` | `15000` | Hard limit Aufnahme |

## Systemd Service

```bash
# Service file: /etc/systemd/system/voice-satellite.service
sudo systemctl start voice-satellite
sudo systemctl stop voice-satellite
sudo systemctl restart voice-satellite
sudo journalctl -u voice-satellite -f   # live logs
```

Runs as user `admin`, auto-restarts on failure, starts on boot.

## Python Environment

```
~/satellite/              # venv + application
~/satellite/main.py       # satellite code
~/satellite/wakeword-ack.wav  # ACK sound (auto-downloaded)
```

Key dependencies: `openwakeword`, `pyaudio`, `numpy`, `websockets`, `pyusb`, `torch` (CPU, for Silero VAD — currently unused, kept as dependency).

## Konventionen

- **Sprache:** Deutsch für Doku, Englisch für Code
- **Netzwerk:** Alle Geräte im TMC WiFi (192.168.50.x)
- **Logging:** Python `logging` module, level DEBUG, format `HH:MM:SS [LEVEL] name: message`
- **ALSA-Warnungen:** Unterdrückt via ctypes libasound error handler
