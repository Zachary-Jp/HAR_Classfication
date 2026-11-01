"""数值对照和成本汇总。不生成课程报告，不按测试成绩重新选模型。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


FIXED_PARAMS = {
    "RBF_SVM": {"model__C": 10.0, "model__gamma": 0.001},
    "RandomForest": {"model__max_depth": None, "model__min_samples_leaf": 1},
    "LogisticRegression": {"model__C": 1.0},
    "KNN": {"model__n_neighbors": 5, "model__weights": "uniform"},
    "FisherLDA": {"model__shrinkage": None},
}


def fixed_parameter_curves(candidates: pd.DataFrame, config: dict, out: Path):
    """从已实际拟合网格中提取固定配置对照，无须重复训练。

    分清固定配置的压缩表现与每个维数重新调参后的管线表现。
    RF的max_features=sqrt随输入维数改变；此表不是所有内部机制完全固定的因果消融。
    调优减固定分数属于开发期描述，不是无偏的调参收益估计。
    """
    from .study import save_table
    rows = []
    for (model, method, k), sub in candidates.groupby(["model", "method", "k"], sort=False):
        fixed = FIXED_PARAMS[model]
        match = sub[sub.params_json.map(lambda value: json.loads(value) == fixed)]
        if len(match) != 1:
            # 允许用户缩小参数网格；没有实际评估该固定设置时不伪造分数。
            continue
        row = match.iloc[0]
        tuned = sub.sort_values(["cv_macro_f1", "candidate_id"], ascending=[False, True]).iloc[0]
        rows.append({"model": model, "method": method, "k": int(k),
                     "fixed_candidate_id": row.candidate_id,
                     "fixed_params_json": json.dumps(fixed, sort_keys=True),
                     "fixed_cv_macro_f1": row.cv_macro_f1,
                     "fixed_cv_std": row.cv_macro_f1_std,
                     "tuned_candidate_id": tuned.candidate_id,
                     "tuned_cv_macro_f1": tuned.cv_macro_f1,
                     "development_tuning_difference_pp": 100 * (tuned.cv_macro_f1 - row.cv_macro_f1)})
    save_table(rows, Path(out) / "tables/fixed_parameter_comparison.csv")
    return pd.DataFrame(rows)


def derive_decisions(out: Path):
    """不删除失败方案；以同一个分类器的full作为压缩代价的参照。"""
    from .audit import decision_summary
    from .study import save_table
    out = Path(out)
    frame = decision_summary(out)
    save_table(frame, out / "tables/decision_summary.csv")
    records = []
    # 帕累托表是测试后描述性汇总，不参与测试前配置选择。
    for row in frame.itertuples():
        no_worse = ((frame.test_macro_f1_percent >= row.test_macro_f1_percent)
                    & (frame.batch_ms_per_window <= row.batch_ms_per_window)
                    & (frame.pipeline_mib <= row.pipeline_mib))
        better_one = ((frame.test_macro_f1_percent > row.test_macro_f1_percent)
                      | (frame.batch_ms_per_window < row.batch_ms_per_window)
                      | (frame.pipeline_mib < row.pipeline_mib))
        dominates = frame.loc[no_worse & better_one, "selection_id"].tolist()
        records.append({"selection_id": row.selection_id, "test_macro_f1_percent": row.test_macro_f1_percent,
                        "batch_ms_per_window": row.batch_ms_per_window, "pipeline_mib": row.pipeline_mib,
                        "descriptive_pareto_efficient": not bool(dominates),
                        "dominated_by_json": json.dumps(dominates),
                        "note": "仅为固定测试结果及本机计时的描述性比较，不是独立选模证据。"})
    save_table(records, out / "tables/resource_tradeoff.csv")
    return frame
