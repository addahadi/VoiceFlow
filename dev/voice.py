import time
import threading

import numpy as np
import pyperclip
import sounddevice as sd
import pyautogui

from faster_whisper import WhisperModel
from pynput import keyboard


SAMPLE_RATE = 16000
HOTKEY = keyboard.KeyCode.from_char("q")


print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)

print("Whisper loaded.")
print()
print("Hold Q, speak, then release Q.")
print("Press ESC to quit.")
print()


recording = False
audio_chunks = []
stream = None
lock = threading.Lock()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    with lock:
        if recording:
            audio_chunks.append(indata.copy())


def start_recording():
    global recording, audio_chunks, stream

    with lock:
        if recording:
            return

        audio_chunks = []
        recording = True

    print("\nRecording...")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )

    stream.start()


def stop_recording():
    global recording, stream

    with lock:
        if not recording:
            return

        recording = False
        chunks = audio_chunks.copy()

    if stream:
        stream.stop()
        stream.close()
        stream = None

    print("Recording stopped.")
    print("Transcribing...")

    if not chunks:
        print("No audio recorded.")
        return

    audio = np.concatenate(chunks, axis=0).flatten()

    segments, info = model.transcribe(
        audio,
        vad_filter=True,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    if not text:
        print("No speech detected.")
        return

    print()
    print("Transcription:")
    print(text)
    print()

    pyperclip.copy(text)

    time.sleep(0.2)

    pyautogui.hotkey("ctrl", "v")

    print("Sent to focused application.")


def on_press(key):
    if key == HOTKEY:
        start_recording()

    elif key == keyboard.Key.esc:
        print("\nExiting...")
        return False


def on_release(key):
    if key == HOTKEY:
        stop_recording()


with keyboard.Listener(
    on_press=on_press,
    on_release=on_release,
) as listener:
    listener.join()