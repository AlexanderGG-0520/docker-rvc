import io
import os
import subprocess
from typing import Optional

# 方針:
# - RVC repo内部の推論APIは変わりやすいので、
#   まず「RVCの既存推論スクリプトを呼ぶ」方式で安定させる。
# - 速度最適化は、その後にimport呼び出しへ移行してもいい。

RVC_ROOT = "/opt/rvc"

def rvc_infer(
    in_audio_path: str,
    model_pth: str,
    index_path: Optional[str],
    transpose: int,
    f0_method: str,
    index_rate: float,
) -> bytes:
    """
    RVC公式repoにある推論エントリをCLIで呼び出し、wav bytesを返す。
    ここは「公式repoの推論スクリプトのパス」に合わせて調整が必要。
    """

    # 出力先
    out_path = "/tmp/out.wav"

    # ★ここが要調整ポイント★
    # RVC WebUIは "infer-web.py" が中心だが、API用途に使いにくいことが多い。
    # 多くのfork/版では "infer/modules/vc" 配下に推論ロジックがある。
    #
    # いったん「このコンテナ内で動く推論CLIを自作」するのが確実なので、
    # /app/app/rvc_cli.py を用意してそれを呼ぶ。
    cmd = [
        "python3",
        "-m",
        "app.rvc_cli",
        "--input", in_audio_path,
        "--output", out_path,
        "--pth", model_pth,
        "--transpose", str(transpose),
        "--f0", f0_method,
        "--index_rate", str(index_rate),
    ]
    if index_path:
        cmd += ["--index", index_path]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{RVC_ROOT}:/app"

    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"RVC infer failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")

    with open(out_path, "rb") as f:
        return f.read()