from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/stream")
async def stream_latency(ws: WebSocket):
    await ws.accept()
    print("✅ Connection accepted from", ws.headers.get("origin"))

    total_bytes = 0
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            pcm = data["audio"]
            packet_id = data["id"]

            total_bytes += len(pcm) * 2  # bytes

            await ws.send_json({
                "packet_id": packet_id,
                "echo_bytes": len(pcm) * 2,
                "total_bytes": total_bytes
            })
    except Exception as e:
        print("❌ WebSocket closed:", e)
