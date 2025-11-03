from fastapi import FastAPI, WebSocket
import asyncio, numpy as np, soundfile as sf, io
from stt import STTEngine
from llm import LLMEngine
from tts import TTSEngine
from config import LLM_MODEL_PATH

app = FastAPI()

stt = STTEngine()
llm = LLMEngine(model_path=LLM_MODEL_PATH)
tts = TTSEngine()

@app.websocket("/stream")
async def voice_socket(ws: WebSocket):
    await ws.accept()
    audio_buffer = bytearray()
    text_context = ""

    while True:
        msg = await ws.receive_bytes()
        audio_buffer.extend(msg)

        # 1️⃣ when buffer ~0.25 s of PCM (≈8 k samples @32 kHz)
        if len(audio_buffer) > 32000 * 0.25 * 2:
            pcm = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_buffer.clear()

            # 2️⃣ partial STT
            partial = stt.model.transcribe(pcm, vad_filter=True)
            partial_text = " ".join(s.text for s, _ in partial)
            await ws.send_json({"partial_text": partial_text})

            # 3️⃣ LLM streaming
            async for chunk in llm.llm.create_completion(
                prompt=text_context + partial_text, stream=True, max_tokens=64
            ):
                token = chunk["choices"][0]["text"]
                await ws.send_json({"token": token})

                # 4️⃣ stream TTS PCM
                for pcm_chunk in tts.tts.stream(token):
                    await ws.send_bytes(pcm_chunk)
