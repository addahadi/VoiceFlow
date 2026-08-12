"""Live microphone level meter — for diagnosing 'it can't hear me'.

    python dev/mic_meter.py            # default input device
    python dev/mic_meter.py 13         # a specific device index (see the list it prints)

Speak normally. If the bar stays flat/empty, that device isn't capturing you.
Ctrl-C to stop.
"""
import sys
import numpy as np
import sounddevice as sd

dev = int(sys.argv[1]) if len(sys.argv) > 1 else None

print("Input devices:")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        mark = "  <- default" if (dev is None and i == sd.default.device[0]) else ""
        print(f"  {i:2d} | {d['name']}{mark}")
print(f"\nUsing device: {dev if dev is not None else 'system default'}")
print("Speak now — Ctrl-C to stop.\n")

peak_seen = 0.0


def cb(indata, frames, t, status):
    global peak_seen
    rms = float(np.sqrt(np.mean(indata ** 2)) + 1e-12)
    peak = float(np.max(np.abs(indata)))
    peak_seen = max(peak_seen, peak)
    bars = int(min(1.0, rms * 25) * 40)
    print("\r[" + "#" * bars + " " * (40 - bars) + f"] rms={rms:.4f} peak_seen={peak_seen:.4f}", end="")


try:
    with sd.InputStream(device=dev, samplerate=16000, channels=1,
                        dtype="float32", blocksize=512, callback=cb):
        while True:
            sd.sleep(200)
except KeyboardInterrupt:
    print("\nstopped.")
