"""Scan every input device and report which one actually hears you.

    python dev/mic_scan.py

Keep talking continuously while it runs (~2s per device). The device(s) with
a high 'peak' are capturing your voice; near-zero means silence.
"""
import numpy as np
import sounddevice as sd

SECONDS = 2.0

devs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
default_in = sd.default.device[0]

print(f"Default input device index: {default_in}\n")
print("Talk continuously now...\n")
print(f"{'idx':>3} | {'peak':>8} | {'rms':>8} | device")
print("-" * 60)

results = []
for i, d in devs:
    got = None
    for sr in (16000, int(d["default_samplerate"]) or 44100):
        try:
            rec = sd.rec(int(SECONDS * sr), samplerate=sr, channels=1,
                         dtype="float32", blocksize=512, device=i)
            sd.wait()
            got = (float(np.max(np.abs(rec))), float(np.sqrt(np.mean(rec ** 2))))
            break
        except Exception:
            continue
    if got is None:
        print(f"{i:>3} | {'--':>8} | {'--':>8} | {d['name']}  (could not open)")
        continue
    peak, rms = got
    flag = "  <== HEARS YOU" if peak > 0.02 else ""
    star = " *" if i == default_in else "  "
    print(f"{i:>3} | {peak:8.4f} | {rms:8.5f} |{star}{d['name']}{flag}")
    results.append((peak, i, d["name"]))

print("-" * 60)
if results:
    results.sort(reverse=True)
    p, i, name = results[0]
    if p > 0.02:
        print(f"\nBest device: index {i}  ->  {name}  (peak {p:.3f})")
    else:
        print("\nNo device heard anything. Mic is likely muted or blocked by "
              "Windows privacy settings.")
