"""独立复算已有预测；检查人员隔离、选择规则与执行顺序。不是重训全网格。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import sha256
from .study import metrics, select_configs


def verify_run(run: Path) -> dict:
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("synthetic", True):
        raise ValueError("正式验证拒绝未完成结果和模拟数据。")
    audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
    if set(audit["train"]["subject_ids"]) & set(audit["test"]["subject_ids"]):
        raise AssertionError("训练/测试人员交叉。")
    if audit["train"]["samples"] != 7352 or audit["test"]["samples"] != 2947:
        raise AssertionError("不是官方完整规模。")
    config = json.loads((run / "config_frozen.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(run / "splits/training_cv.csv")
    if len(assignments) != 7352 or not assignments.row_id.is_unique:
        raise AssertionError("CV验证索引不完整或重复。")
    if not np.array_equal(np.sort(assignments.row_id), np.arange(7352)):
        raise AssertionError("CV索引超出官方训练范围。")
    if (assignments.groupby("subject").validation_fold.nunique() != 1).any():
        raise AssertionError("同一人员被拆到不同验证折。")
    if assignments.validation_fold.nunique() != config["cv_folds"]:
        raise AssertionError("交叉验证折数不符。")
    candidates = pd.read_csv(run / "tables/cv_candidates.csv")
    fold_metrics = pd.read_csv(run / "tables/cv_folds.csv")
    if fold_metrics.groupby("candidate_id").size().ne(config["cv_folds"]).any():
        raise AssertionError("部分候选参数缺少某折结果。")
    for candidate in candidates.itertuples():
        values = fold_metrics[fold_metrics.candidate_id == candidate.candidate_id]
        if not np.isclose(values.macro_f1.mean(), candidate.cv_macro_f1, atol=1e-9):
            raise AssertionError("CV平均分与逐折分数不一致。")
        if not np.isclose(values.macro_f1.std(ddof=0), candidate.cv_macro_f1_std, atol=1e-9):
            raise AssertionError("CV标准差不一致。")
    expected, _ = select_configs(candidates, config["allowed_macro_f1_drop"])
    selected = pd.read_csv(run / "tables/selected_configs.csv")
    if expected.set_index("selection_id").candidate_id.to_dict() != selected.set_index("selection_id").candidate_id.to_dict():
        raise AssertionError("实际选择不符合预先设定的规则。")
    lock = json.loads((run / "selection_lock.json").read_text(encoding="utf-8"))
    if lock["cv_sha256"] != sha256(run / "tables/cv_candidates.csv"):
        raise AssertionError("锁定后CV表发生变化。")
    if lock["config_sha256"] != sha256(run / "config_frozen.json"):
        raise AssertionError("锁定后配置发生变化。")
    if manifest["selection_lock_sha256"] != sha256(run / "selection_lock.json"):
        raise AssertionError("选择锁文件发生变化。")
    locked_ids = {r["selection_id"]: r["candidate_id"] for r in lock["selected"]}
    if locked_ids != selected.set_index("selection_id").candidate_id.to_dict():
        raise AssertionError("测试配置与锁定配置不符。")
    events = [json.loads(line)["event"] for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    if not (events.index("selection_locked") < events.index("test_loaded") < events.index("test_evaluated")):
        raise AssertionError("测试集读取与选择顺序不符。")
    preds = pd.read_csv(run / "tables/test_predictions.csv")
    scores = pd.read_csv(run / "tables/test_metrics.csv")
    people = pd.read_csv(run / "tables/test_subjects.csv")
    test_split = pd.read_csv(run / "splits/official_test.csv").sort_values("row_id")
    for row in scores.itertuples():
        sub = preds[preds.selection_id == row.selection_id].sort_values("row_id")
        if not np.array_equal(sub.row_id, np.arange(2947)): raise AssertionError("测试预测不完整。")
        if not np.array_equal(sub.actual, test_split.activity) or not np.array_equal(sub.subject, test_split.subject):
            raise AssertionError("测试真值或人员信息不匹配。")
        actual = metrics(sub.actual.to_numpy(), sub.predicted.to_numpy())
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            if not np.isclose(actual[key], getattr(row, key), atol=1e-9):
                raise AssertionError(f"{row.selection_id} 的 {key} 无法从预测复算。")
        for person in people[people.selection_id == row.selection_id].itertuples():
            pp = sub[sub.subject == person.subject]
            if not np.isclose(metrics(pp.actual, pp.predicted)["macro_f1"], person.macro_f1, atol=1e-9):
                raise AssertionError("逐人员F1无法复算。")
    fit = pd.read_csv(run / "tables/final_fit_and_size.csv")
    for row in fit.itertuples():
        model = run / "models" / f"{row.selection_id}.joblib"
        if sha256(model) != row.model_sha256 or model.stat().st_size != row.pipeline_bytes:
            raise AssertionError("模型文件与保存时记录不一致。")
    return {"status": "PASS", "n_test_models": len(scores), "n_cv_candidates": len(candidates),
            "checks": ["人员隔离", "折分配完整性", "CV均值/标准差复算", "维数选择规则", "选择锁与事件顺序",
                       "测试总体/逐人员指标复算", "模型文件哈希和体积"],
            "scope": "文件与计算一致性检查；并非从头重新训练每个候选，也不能防止人为篡改全部日志。"}
