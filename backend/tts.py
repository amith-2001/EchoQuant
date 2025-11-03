from TTS.api import TTS
import soundfile as sf
import numpy as np

class TTSEngine:
    def __init__(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    def synthesize(self, text, outfile):
        audio = self.tts.tts(text=text, speaker="en", language="en")
        sf.write(outfile, np.array(audio), 22050)
        return outfile
