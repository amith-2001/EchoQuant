from faster_whisper import WhisperModel

class STTEngine:
    def __init__(self, size="small"):
        self.model = WhisperModel(size, device="cuda", compute_type="float16")

    def transcribe(self, wav_path):
        segments, _ = self.model.transcribe(wav_path, vad_filter=True)
        return " ".join(s.text.strip() for s in segments if s.text)
