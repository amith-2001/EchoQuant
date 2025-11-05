from faster_whisper import WhisperModel
import time

# Path to an audio file (16 kHz mono WAV)
AUDIO_PATH = "backend/test scripts/Loka_input.wav"

# Load Whisper model (use small for 3050)
print("🔄 Loading model...")
model = WhisperModel("C:/Users/amith/OneDrive/Desktop/EchoQuant/models/whisper-small", device="cuda", compute_type="float16")


print("🎙️ Transcribing:", AUDIO_PATH)
start = time.time()

segments, info = model.transcribe(AUDIO_PATH, beam_size=5, vad_filter=True)

print(f"\nDetected language: {info.language}")
print("Segments:")
for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")

print(f"\n⏱️  Transcription done in {time.time() - start:.2f}s")
