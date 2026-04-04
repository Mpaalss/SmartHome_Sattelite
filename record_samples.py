"""Record wakeword training samples. Press Enter to start, Enter to stop."""

import os
import sys
import threading
import wave

import ctypes
import ctypes.util

# Suppress ALSA warnings
_EHT = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
_eh = _EHT(lambda *_: None)
_alsa = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound"))
_alsa.snd_lib_error_set_handler(_eh)

import numpy as np
import pyaudio

SAMPLE_RATE = 16000
CHANNELS = 2
ASR_CHANNEL = 1
CHUNK_SIZE = 1280  # 80ms
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "samples")


def find_respeaker(pa):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "xvf3800" in info["name"].lower() or "respeaker" in info["name"].lower():
            return i
    raise RuntimeError("ReSpeaker not found")


def extract_channel(stereo_chunk, channel=ASR_CHANNEL):
    samples = np.frombuffer(stereo_chunk, dtype=np.int16)
    return samples[channel::CHANNELS]


def save_wav(filename, samples, sr=SAMPLE_RATE):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def next_filename():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    n = len(existing) + 1
    return os.path.join(OUTPUT_DIR, f"sample_{n:03d}.wav")


def count_samples():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])


def main():
    pa = pyaudio.PyAudio()
    mic_index = find_respeaker(pa)
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=CHUNK_SIZE,
    )

    print(f"=== Wakeword Sample Recorder ===")
    print(f"Mic: {pa.get_device_info_by_index(mic_index)['name']}")
    print(f"Output: {OUTPUT_DIR}/ ({count_samples()} existing)")
    print()
    print("Enter = Start Aufnahme")
    print("Enter = Stop Aufnahme + Speichern")
    print("Ctrl+C = Beenden")
    print()

    try:
        while True:
            input(f"  [{count_samples()} samples] Enter zum Aufnehmen...")

            # Drain buffered audio
            for _ in range(3):
                stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # Record in background, wait for Enter to stop
            frames = []
            stop_event = threading.Event()

            def record():
                while not stop_event.is_set():
                    raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    mono = extract_channel(raw)
                    frames.append(mono)

            rec_thread = threading.Thread(target=record, daemon=True)
            rec_thread.start()

            print("  \033[91m● REC\033[0m — Sag 'Hey Thor' — Enter zum Stoppen")
            input()

            stop_event.set()
            rec_thread.join(timeout=1)

            # Save
            if not frames:
                print("  Keine Aufnahme.")
                continue

            audio = np.concatenate(frames)
            duration = len(audio) / SAMPLE_RATE

            if duration < 0.3:
                print(f"  Zu kurz ({duration:.1f}s), verworfen.")
                continue

            filepath = next_filename()
            save_wav(filepath, audio)
            name = os.path.basename(filepath)
            print(f"  \033[92m✓\033[0m {name}: {duration:.1f}s  ({count_samples()} total)")
            print()

    except KeyboardInterrupt:
        print(f"\n\n{count_samples()} Samples in {OUTPUT_DIR}/")
    finally:
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
