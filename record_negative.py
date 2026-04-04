"""Record negative samples (speech without the wakeword). Just records continuously."""

import os
import sys
import wave

import ctypes
import ctypes.util

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
CHUNK_SIZE = 1280
RECORD_SECONDS = 120  # 2 minutes
OUTPUT = os.path.join(os.path.dirname(__file__), "negative_speech.wav")


def find_respeaker(pa):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "xvf3800" in info["name"].lower() or "respeaker" in info["name"].lower():
            return i
    raise RuntimeError("ReSpeaker not found")


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

    print(f"=== Negative Speech Recorder ===")
    print(f"Recording {RECORD_SECONDS}s to {OUTPUT}")
    print(f"Sprich normal, OHNE 'Hey Bernd'!")
    print(f"Ctrl+C zum vorzeitigen Stoppen.\n")

    frames = []
    try:
        total_chunks = int(RECORD_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
        for i in range(total_chunks):
            raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)
            mono = samples[ASR_CHANNEL::CHANNELS]
            frames.append(mono)
            elapsed = (i + 1) * CHUNK_SIZE / SAMPLE_RATE
            if i % 25 == 0:  # every 2s
                sys.stdout.write(f"\r  {elapsed:.0f}s / {RECORD_SECONDS}s")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass

    stream.close()
    pa.terminate()

    audio = np.concatenate(frames)
    duration = len(audio) / SAMPLE_RATE
    print(f"\n\nSaving {duration:.0f}s to {OUTPUT}")

    with wave.open(OUTPUT, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print("Done!")


if __name__ == "__main__":
    main()
