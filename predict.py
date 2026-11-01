"""用可信的已保存Pipeline预测561维特征；不支持直接输入手机原始信号。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预测输入必须与官方561维特征的顺序和定义一致。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="无表头、空格分隔的561列特征文本")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    metadata = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["sklearn_version"] != sklearn.__version__:
        raise RuntimeError(f"模型使用scikit-learn {metadata['sklearn_version']}；请使用相同版本。")
    X = np.loadtxt(args.input, dtype=np.float64, ndmin=2)
    if X.shape[1] != metadata["input_features"] or not np.isfinite(X).all():
        raise ValueError("输入特征数或数值有效性不符合模型要求。")
    # joblib反序列化可能执行代码，仅加载自己训练或来源可信的模型文件。
    model = joblib.load(args.model)
    with threadpool_limits(limits=1): pred = model.predict(X)
    activities = metadata["activities"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": np.arange(len(X)), "predicted_id": pred,
                  "predicted_activity": [activities[str(int(p))] for p in pred]}).to_csv(args.output, index=False)
    print(f"[Predict] {len(X)} rows saved to {args.output}")
