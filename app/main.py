from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import os
import tempfile

from app.rvc_engine import rvc_infer

app = FastAPI(title="docker-rvc", version="0.1.0")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model: str = Form(...),               # 例: "my_model" -> /models/my_model.pth
    transpose: int = Form(0),             # semitone
    f0_method: str = Form("rmvpe"),       # "rmvpe" / "harvest" / "dio" など
    index_rate: float = Form(0.7),
):
    filename = file.filename or ""
    if not filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
        raise HTTPException(400, "audio file required")

    models_dir = os.environ.get("RVC_MODELS_DIR", "/models")
    pth = os.path.join(models_dir, f"{model}.pth")
    index = os.path.join(models_dir, f"{model}.index")

    if not os.path.exists(pth):
        raise HTTPException(404, f"model pth not found: {pth}")

    # indexは無くても動かす（RVCの設定次第）
    if not os.path.exists(index):
        index = None

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in_audio")
        raw = await file.read()
        with open(in_path, "wb") as f:
            f.write(raw)

        out_wav_bytes = rvc_infer(
            in_audio_path=in_path,
            model_pth=pth,
            index_path=index,
            transpose=transpose,
            f0_method=f0_method,
            index_rate=index_rate,
        )

    return Response(content=out_wav_bytes, media_type="audio/wav")