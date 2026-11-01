"""单独补测锁定模型的分阶段时间。不调参、不改变模型和预测表。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    import joblib
    import pandas as pd
    import sklearn
    from har_study.data import prepare_dataset, load_split, file_hashes
    from har_study.verify import verify_run
    from har_study.stage_timing import load_timing_config, benchmark_stages, verify_stage_timing, draw_stage_figures
    verify_run(args.run)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["sklearn"] != sklearn.__version__:
        raise RuntimeError("模型保存与当前scikit-learn版本不同；请使用相同环境。")
    root = prepare_dataset(args.data, ROOT / "data")
    protocol = json.loads((args.run / "protocol_frozen.json").read_text(encoding="utf-8"))
    if file_hashes(root) != protocol["source_data_hashes"]:
        raise ValueError("输入数据哈希与原训练记录不同。")
    config_file = args.run / "timing_config_frozen.json"
    if not config_file.is_file():
        config_file.write_text((ROOT / "timing_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    config = load_timing_config(config_file)
    selected = pd.read_csv(args.run / "tables/selected_configs.csv")
    # 只加载用户自己产生或已确认可信的joblib，不接受陌生来源二进制模型。
    fitted = {sid: joblib.load(args.run / "models" / f"{sid}.joblib") for sid in selected.selection_id}
    test = load_split(root, "test")
    benchmark_stages(args.run, test.X, fitted, config)
    print(verify_stage_timing(args.run))
    draw_stage_figures(args.run, args.font)
