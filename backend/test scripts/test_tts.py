from TTS.api import TTS
import soundfile as sf
import torch
import time

# ================================
# 1️⃣ Configuration
# ================================
TEXT = "Hello! This is EchoQuant speaking from your RTX 3050 GPU."
OUTPUT_PATH = "samples/tts_output.wav"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# ================================
# 2️⃣ Load the model
# ================================
print("🔄 Loading XTTS-v2 model...")
start_load = time.time()

tts = TTS(MODEL_NAME, progress_bar=False, gpu=torch.cuda.is_available())

print(f"✅ Model loaded in {time.time() - start_load:.2f}s (GPU={torch.cuda.is_available()})")

# ================================
# 3️⃣ Generate speech
# ================================
print(f"🗣️ Synthesizing:\n\"{TEXT}\"\n")
start_gen = time.time()

# # Generate speech as waveform array (float32)
# wav = tts.tts(text=TEXT, speaker="en", language="en")




# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="backend/test scripts/Loka_input.wav",
                language="en")


# Save audio to file
# sf.write(OUTPUT_PATH, wav, 24000)

# print(f"✅ Speech saved to {OUTPUT_PATH}")
# print(f"⏱️ Generation time: {time.time() - start_gen:.2f}s")

# # ================================
# # 4️⃣ Optional playback (local only)
# # ================================
# try:
#     import sounddevice as sd
#     print("🎧 Playing audio...")
#     sd.play(wav, 24000)
#     sd.wait()
# except Exception:
#     print("⚠️ Install `sounddevice` for direct playback: pip install sounddevice")
