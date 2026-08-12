from faster_whisper import WhisperModel

print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)

print("Transcribing...")

segments, info = model.transcribe("test.wav")

print("\nTranscription:")
print("----------------")

for segment in segments:
    print(segment.text)

print("----------------")
