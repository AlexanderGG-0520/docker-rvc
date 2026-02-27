from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
def health():
    return JSONResponse({"ok": True})

# =========================================================
# ここをRVC推論に置き換える
# 入力/出力: 48kHz / mono / s16le (int16 PCM) / 20ms(960 samples)推奨
# =========================================================
def infer_pcm_s16le_48k_mono(frame: bytes) -> bytes:
    # TODO: RVC推論を実装
    # まずは疎通確認用にパススルー
    return frame

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        # 最初にhello JSONが来ても来なくてもOKにする
        first = await ws.receive()
        if first.get("bytes"):
            out = infer_pcm_s16le_48k_mono(first["bytes"])
            await ws.send_bytes(out)

        while True:
            data = await ws.receive_bytes()
            out = infer_pcm_s16le_48k_mono(data)
            await ws.send_bytes(out)

    except WebSocketDisconnect:
        return