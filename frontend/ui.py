import gradio as gr
import requests, soundfile as sf, numpy as np, tempfile

BACKEND = "http://127.0.0.1:8000"

def talk(audio):
    sr, data = audio
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, np.array(data), sr)
    with open(tmp.name, "rb") as f:
        r = requests.post(f"{BACKEND}/process", files={"file": f})
    res = r.json()
    reply = res["reply"]

    r2 = requests.get(f"{BACKEND}/audio", params={"q": reply})
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.write(r2.content)
    out.close()

    return out.name, res

gr.Interface(
    fn=talk,
    inputs=gr.Audio(sources=["microphone"], type="numpy", label="Speak"),
    outputs=[gr.Audio(label="Assistant Reply"), gr.JSON(label="Details")],
    title="EchoQuant – Base Prototype"
).launch(server_port=7860)
