import argparse
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--index", default=None)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--f0", default="rmvpe")
    ap.add_argument("--index_rate", type=float, default=0.7)
    args = ap.parse_args()

    # ===== ここが要調整ポイント =====
    # RVC WebUIの内部モジュールがどこにあるかでimportが変わる。
    # エラーに出たパスに合わせて書き換えろ。
    try:
        # 例: 多くのRVC構造でこういう雰囲気
        from infer.modules.vc.modules import VC  # type: ignore
    except Exception as e:
        print("Failed to import RVC VC module. You need to adjust import path.", file=sys.stderr)
        raise

    # VCの初期化も版で違う。ここもログ見て合わせる。
    vc = VC()

    # 下も版で違うので「動く形に合わせて」直す前提。
    # ここでは“それっぽい引数名”で置く。
    # 実際には vc.pipeline(...) や vc.vc_inference(...) みたいな関数になることが多い。
    wav = vc.infer(
        audio_path=args.input,
        pth_path=args.pth,
        index_path=args.index,
        transpose=args.transpose,
        f0_method=args.f0,
        index_rate=args.index_rate,
    )

    # wavをファイル出力する処理も、返り値形式で変わる（bytes/np.array/path）。
    # ここは最初に動かして形を確定させる。
    if isinstance(wav, (bytes, bytearray)):
        with open(args.output, "wb") as f:
            f.write(wav)
        return

    # numpy配列だった場合
    try:
        import numpy as np
        import soundfile as sf
        if isinstance(wav, np.ndarray):
            sf.write(args.output, wav, 48000)
            return
    except Exception:
        pass

    raise RuntimeError(f"Unexpected output type from vc.infer(): {type(wav)}")

if __name__ == "__main__":
    main()