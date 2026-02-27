import os
import time
from typing import Optional, Literal

import numpy as np
import torch
from fastapi import FastAPI, Request, Response, HTTPException, Query

app = FastAPI(title="RVC Infer API", version="0.1.0")

MODELS_DIR = os.getenv("RVC_MODELS_DIR", "/models")
DEFAULT_SR = int(os.getenv("RVC_DEFAULT_SR", "48000"))
DEFAULT_CH = int(os.getenv("RVC_DEFAULT_CH", "1"))

# ====== GPU / Device ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== ここに “モデルロード” のキャッシュを置く ======
# 例: model_cache["my_model.pth"] = loaded_model_object
model_cache = {}

def parse_int_header(req: Request, name: str, fallback: int) -> int:
    v = req.headers.get(name)
    if v is None:
        return fallback
    try:
        return int(v)
    except ValueError:
        return fallback

def pcm_s16le_to_f32mono(pcm_bytes: bytes, channels: int) -> np.ndarray:
    """s16le raw PCM -> float32 mono [-1, 1]"""
    if len(pcm_bytes) % 2 != 0:
        raise ValueError("PCM length must be even (s16le).")
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 1:
        return x
    if channels <= 0:
        raise ValueError("channels must be >= 1")
    # interleaved -> mono (average)
    if x.size % channels != 0:
        raise ValueError("PCM samples not aligned to channels.")
    x = x.reshape(-1, channels).mean(axis=1)
    return x

def f32mono_to_pcm_s16le(x: np.ndarray) -> bytes:
    """float32 mono [-1,1] -> s16le raw PCM"""
    x = np.clip(x, -1.0, 1.0)
    y = (x * 32767.0).astype(np.int16)
    return y.tobytes()

def load_model(model_id: str):
    """
    model_id:
      - filename like "foo.pth" (resolved under MODELS_DIR)
      - or full path under MODELS_DIR
    """
    # 絶対に /models 外へ出さない
    safe_path = os.path.normpath(os.path.join(MODELS_DIR, model_id))
    if not safe_path.startswith(os.path.normpath(MODELS_DIR) + os.sep) and safe_path != os.path.normpath(MODELS_DIR):
        raise HTTPException(status_code=400, detail="invalid model path")

    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail=f"model not found: {model_id}")

    if safe_path in model_cache:
        return model_cache[safe_path]

    # ====== TODO: ここでRVCモデルをロード ======
    # 例:
    # model = torch.load(safe_path, map_location=DEVICE)
    # model.eval()
    # model_cache[safe_path] = model
    # return model
    model = {"_dummy": True, "path": safe_path}
    model_cache[safe_path] = model
    return model

@app.get("/healthz")
def healthz():
    return {"ok": True, "device": DEVICE}

@app.post("/infer")
async def infer(
    request: Request,
    # クエリでも指定可能（ヘッダが付けられないクライアント対策）
    sr: int = Query(DEFAULT_SR, ge=8000, le=192000),
    ch: int = Query(DEFAULT_CH, ge=1, le=2),
    model: Optional[str] = Query(None),
    f0: Literal["rmvpe", "harvest", "dio"] = Query("rmvpe"),
    transpose: int = Query(0, ge=-36, le=36),
    index_rate: float = Query(0.75, ge=0.0, le=1.0),
    filter_radius: int = Query(3, ge=0, le=7),
):
    """
    Input:  Content-Type: application/octet-stream
            Body: raw PCM s16le (interleaved)
            Header (optional): X-Sample-Rate, X-Channels, X-Model, X-F0, X-Transpose, X-Index-Rate, X-Filter-Radius
    Output: raw PCM s16le (mono)
    """
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/octet-stream":
        raise HTTPException(status_code=415, detail="Content-Type must be application/octet-stream")

    # ヘッダ優先（存在すれば上書き）
    sr_h = parse_int_header(request, "X-Sample-Rate", sr)
    ch_h = parse_int_header(request, "X-Channels", ch)
    model_h = request.headers.get("X-Model")
    f0_h = request.headers.get("X-F0")
    transpose_h = request.headers.get("X-Transpose")
    index_rate_h = request.headers.get("X-Index-Rate")
    filter_radius_h = request.headers.get("X-Filter-Radius")

    sr = sr_h
    ch = ch_h
    if model_h:
        model = model_h
    if f0_h in {"rmvpe", "harvest", "dio"}:
        f0 = f0_h  # type: ignore
    if transpose_h is not None:
        try: transpose = int(transpose_h)
        except ValueError: pass
    if index_rate_h is not None:
        try: index_rate = float(index_rate_h)
        except ValueError: pass
    if filter_radius_h is not None:
        try: filter_radius = int(filter_radius_h)
        except ValueError: pass

    pcm = await request.body()
    if not pcm:
        raise HTTPException(status_code=400, detail="empty body")

    t0 = time.time()

    # decode PCM
    try:
        x = pcm_s16le_to_f32mono(pcm, channels=ch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # model select
    if model is None:
        # ここは運用で決めて。とりあえずデフォルトモデル名を環境変数で許可
        model = os.getenv("RVC_DEFAULT_MODEL", "")
        if not model:
            raise HTTPException(status_code=400, detail="model is required (query model= or header X-Model)")

    m = load_model(model)

    # ====== TODO: ここが“本体”。RVC推論に置き換える ======
    # 入力: x (float32 mono), sr, f0, transpose, index_rate, filter_radius, DEVICE
    # 出力: y (float32 mono), out_sr(=sr推奨)
    #
    # 例のダミー: 音量を少し下げるだけ
    y = (x * 0.9).astype(np.float32)
    out_sr = sr

    # encode PCM
    out_pcm = f32mono_to_pcm_s16le(y)

    dt = time.time() - t0
    # 10秒音声をどれくらいで処理したかが重要（<1.0秒とかが目標）
    headers = {
        "X-Device": DEVICE,
        "X-Elapsed": f"{dt:.4f}",
        "X-In-SR": str(sr),
        "X-Out-SR": str(out_sr),
        "X-Model-Path": m.get("path", "") if isinstance(m, dict) else "",
    }

    return Response(content=out_pcm, media_type="application/octet-stream", headers=headers)