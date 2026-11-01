#!/usr/bin/env python3
"""完整五模型实验入口：适度调参、断点恢复、验证和中文绘图。

python run_experiment.py --plan
python run_experiment.py
python run_experiment.py --resume outputs/实际目录
python run_experiment.py --verify-only outputs/实际目录
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

# 网格在进程层面并行；底层数值库及各分类器不再重复启动线程。
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

try:
    from har_study import study as core
    from har_study.data import ACTIVITIES, audit_splits, file_hashes, load_split, prepare_dataset, sha256
    from har_study.search import prepare_fold, score_fold
    from har_study.verify import verify_run
except ImportError as exc:
    raise SystemExit(
        "请把 run_experiment.py 放到最新五模型整合项目的根目录，"
        "与 run_experiment.py、config.json、src/ 同级。不要放进src/，也不要单独放在空目录。\n"
        f"导入错误：{exc}"
    ) from exc

VERSION = "integrated-five-models-stage-timing-1.2"
MODEL_ORDER = ("RBF_SVM", "RandomForest", "LogisticRegression", "KNN", "FisherLDA")
METHODS = ("full", "pca", "anova")
GRID_KEYS = {"RBF_SVM": "svm_grid", "RandomForest": "rf_grid",
             "LogisticRegression": "lr_grid", "KNN": "knn_grid", "FisherLDA": "lda_grid"}
METHOD_CN = {"full": "全特征", "pca": "PCA", "anova": "ANOVA筛选"}
ACTIVITY_CN = ["行走", "上楼", "下楼", "坐", "站立", "躺卧"]
FIXED_PARAMS = {
    "RBF_SVM": {"model__C": 10.0, "model__gamma": 0.001},
    "RandomForest": {"model__max_depth": None, "model__min_samples_leaf": 1},
    "LogisticRegression": {"model__C": 1.0},
    "KNN": {"model__n_neighbors": 5, "model__weights": "uniform"},
    "FisherLDA": {"model__shrinkage": None},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    """稳定序列化，用于配置去重和断点匹配，不以测试成绩生成配置。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    temporary = path.with_name(path.name + ".tmp")
    # 17位有效数字便于保留锁定选择时的浮点值；显示时再四舍五入。
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.17g")
    temporary.replace(path)


def build_config() -> dict:
    """只有一个配置源；恢复时与冻结配置一致，禁止中途改变网格。"""
    config = read_json(ROOT / "config.json")
    core.validate_config(config)
    return config


def previous_grids() -> dict:
    """原四模型的网格；新增Fisher没有历史基线，不伪造旧成绩。"""
    return {
        "RBF_SVM": {"model__C": [1.0, 10.0, 100.0], "model__gamma": ["scale", 0.001]},
        "RandomForest": {"model__max_depth": [None, 20], "model__min_samples_leaf": [1, 3]},
        "LogisticRegression": {"model__C": [0.1, 1.0, 10.0]},
        "KNN": {"model__n_neighbors": [3, 5, 9], "model__weights": ["uniform", "distance"]},
    }


def build_search_plan(config: dict) -> dict:
    """完整遍历适度扩展后的有限网格，不再使用RF大范围随机抽样。"""
    parameters = {m: list(ParameterGrid(config[GRID_KEYS[m]])) for m in config["models"]}
    old = previous_grids()
    for model in config["models"]:
        if model in old and not {canonical(p) for p in ParameterGrid(old[model])} <= {canonical(p) for p in parameters[model]}:
            raise ValueError(f"{model}网格缺少原版本候选。为保持旧/新搜索对照，请保留原候选。")
        if canonical(FIXED_PARAMS[model]) not in {canonical(p) for p in parameters[model]}:
            raise ValueError(f"{model}缺少固定参数对照。")
    nrep = 1 + 2 * len(config["dimensions"])
    n = nrep * sum(len(values) for values in parameters.values())
    return {
        "version": VERSION, "seed": config["seed"], "parameter_sets": parameters,
        "parameter_counts": {m: len(values) for m, values in parameters.items()},
        "representations_per_model": nrep, "n_candidates": n,
        "n_cv_fits": n * config["cv_folds"],
        "n_protocol_fits": 2 * len(config["models"]) * config["cv_folds"],
        "n_final_models": 3 * len(config["models"]),
        "scope": "五模型均完整遍历给定有限网格；不是全局最优搜索。",
    }


def candidate_specs(config: dict, plan: dict, n_features: int = 561) -> list[dict]:
    specs = []
    for model in config["models"]:
        for method in METHODS:
            for k in ([n_features] if method == "full" else config["dimensions"]):
                for i, params in enumerate(plan["parameter_sets"][model]):
                    specs.append({"candidate_id": f"{model}__{method}_{k}__p{i:04d}",
                                  "model": model, "method": method, "k": int(k),
                                  "params_json": canonical(params)})
    if len(specs) != plan["n_candidates"]:
        raise AssertionError("候选数量与搜索计划不一致。")
    return specs


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__}


def source_hashes() -> dict:
    paths = [Path(__file__).resolve(), *sorted((ROOT / "src" / "har_study").glob("*.py"))]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def fingerprint(context: dict) -> str:
    return hashlib.sha256(canonical(context).encode("utf-8")).hexdigest()


def check_checkpoint(record: dict, expected: set[tuple], token: str) -> None:
    if record.get("context_sha256") != token:
        raise ValueError("断点记录不属于当前配置、数据和代码，禁止混用。")
    key = (record.get("candidate_id"), record.get("fold"))
    if key not in expected:
        raise ValueError(f"存在不属于搜索计划的断点：{key}")
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        value = record.get(metric)
        if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"断点中的{metric}不合法。")
    for metric in ("classifier_fit_seconds", "classifier_predict_seconds"):
        if not np.isfinite(record.get(metric, np.nan)) or record[metric] < 0:
            raise ValueError("断点中的运行时间不合法。")


def run_search(train, config: dict, plan: dict, folds: list, out: Path,
               workers: int, token: str) -> pd.DataFrame:
    """共享折内变换，候选分类器逐个实际拟合；一折完成就原子保存断点。"""
    cache = out / ".extended_cache"
    cache.mkdir(exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    specs = candidate_specs(config, plan, train.X.shape[1])
    expected = {(s["candidate_id"], f) for s in specs for f in range(len(folds))}
    found = {}
    for file in sorted(checkpoints.glob("*.json")):
        item = read_json(file)
        check_checkpoint(item, expected, token)
        key = (item["candidate_id"], item["fold"])
        if key in found:
            raise ValueError(f"断点包含重复任务：{key}")
        found[key] = item
    pending = [(s, f) for s in specs for f in range(len(folds)) if (s["candidate_id"], f) not in found]
    total = len(expected)
    print(f"[Search] {len(specs)} candidates, {total} fold fits; reused checkpoints={len(found)}", flush=True)
    if pending:
        prep = []
        # 恢复时重新拟合5份变换。代价小，避免读取损坏或过期的临时PCA缓存。
        for f, (tr, va) in enumerate(folds):
            with threadpool_limits(limits=1):
                item = prepare_fold(train, tr, va, max(config["dimensions"]), cache / f"fold_{f}")
            prep.append({"fold": f, **item})
            print(f"[Prepare] fold={f + 1}/{len(folds)}; subject-disjoint transform ready", flush=True)
        atomic_json(out / "fold_transform_audit.json", prep)
        try:
            with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
                tasks = (joblib.delayed(score_fold)(s, f, cache / f"fold_{f}", config) for s, f in pending)
                stream = joblib.Parallel(n_jobs=workers, pre_dispatch=workers,
                                        return_as="generator_unordered")(tasks)
                for item in stream:
                    item["context_sha256"] = token
                    check_checkpoint(item, expected, token)
                    key = (item["candidate_id"], item["fold"])
                    if key in found:
                        raise AssertionError("重复完成同一搜索任务。")
                    atomic_json(checkpoints / f"{key[0]}__fold{key[1]}.json", item)
                    found[key] = item
                    if len(found) % 50 == 0 or len(found) == total:
                        atomic_csv(pd.DataFrame(found.values()).sort_values(["candidate_id", "fold"]),
                                   out / "tables/cv_folds.csv")
                        print(f"[Search] {len(found)}/{total} completed", flush=True)
        finally:
            if found:
                atomic_csv(pd.DataFrame(found.values()).sort_values(["candidate_id", "fold"]),
                           out / "tables/cv_folds.csv")
    if set(found) != expected:
        raise AssertionError("搜索记录不完整。")
    scores = pd.DataFrame(found.values()).sort_values(["candidate_id", "fold"])
    summaries = []
    for spec in specs:
        part = scores[scores.candidate_id == spec["candidate_id"]]
        summaries.append({**spec, "cv_macro_f1": float(part.macro_f1.mean()),
            "cv_macro_f1_std": float(part.macro_f1.std(ddof=0)),
            "cv_accuracy": float(part.accuracy.mean()),
            "mean_classifier_fit_seconds": float(part.classifier_fit_seconds.mean()),
            "cv_warning_count": int(part.warning_count.sum())})
    frame = pd.DataFrame(summaries)
    atomic_csv(frame, out / "tables/cv_candidates.csv")
    atomic_csv(scores, out / "tables/cv_folds.csv")
    atomic_json(out / "search_engine.json", {
        "engine": VERSION, "n_candidates": len(specs), "actual_classifier_fits": len(scores),
        "all_configured_combinations_evaluated": True,
        "preprocessing_fitted_within_each_training_fold": True,
        "context_sha256": token, "complete_at_utc": utc_now(),
        "timing_scope": "CV fit时间仅为分类器，不含共享变换；最终另测完整Pipeline预测。",
    })
    return frame


def comparison_tables(candidates: pd.DataFrame, config: dict, out: Path) -> None:
    """原网格对照也用本次真实拟合成绩，不读取历史测试分数。"""
    fixed_rows, old_rows, boundaries = [], [], []
    old_keys = {m: {canonical(p) for p in ParameterGrid(g)} for m, g in previous_grids().items()}
    for (model, method, k), part in candidates.groupby(["model", "method", "k"], sort=False):
        keys = part.params_json.map(lambda x: canonical(json.loads(x)))
        fixed = part[keys == canonical(FIXED_PARAMS[model])]
        old = part[keys.isin(old_keys.get(model, set()))]
        tuned = part.sort_values(["cv_macro_f1", "candidate_id"], ascending=[False, True]).iloc[0]
        if len(fixed) != 1 or len(old) != len(old_keys.get(model, set())):
            raise AssertionError("固定参数或旧网格候选缺失，不能公平比较扩展范围。")
        base = fixed.iloc[0]
        old_best = old.sort_values(["cv_macro_f1", "candidate_id"], ascending=[False, True]).iloc[0] if len(old) else None
        fixed_rows.append({"model": model, "method": method, "k": int(k),
            "fixed_candidate_id": base.candidate_id, "fixed_params_json": base.params_json,
            "fixed_cv_macro_f1": base.cv_macro_f1, "fixed_cv_std": base.cv_macro_f1_std,
            "tuned_candidate_id": tuned.candidate_id, "tuned_cv_macro_f1": tuned.cv_macro_f1,
            "development_tuning_difference_pp": 100 * (tuned.cv_macro_f1 - base.cv_macro_f1)})
        if old_best is not None:
            old_rows.append({"model": model, "method": method, "k": int(k),
                "old_grid_best_id": old_best.candidate_id, "old_grid_cv_macro_f1": old_best.cv_macro_f1,
                "extended_best_id": tuned.candidate_id, "extended_cv_macro_f1": tuned.cv_macro_f1,
                "cv_difference_pp": 100 * (tuned.cv_macro_f1 - old_best.cv_macro_f1),
                "interpretation": "同一训练验证记录中的开发期比较，扩展取最大值有选参偏差，不是独立测试收益。"})
        params = json.loads(tuned.params_json)
        for name, value in params.items():
            pool = config[GRID_KEYS[model]][name]
            numeric = [v for v in pool if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if isinstance(value, (int, float)) and numeric and value in (min(numeric), max(numeric)):
                boundaries.append({"model": model, "method": method, "k": int(k),
                    "parameter": name, "selected_value": value,
                    "numeric_min": min(numeric), "numeric_max": max(numeric),
                    "note": "仅提示有限范围边界；不自动扩大搜索，不使用测试分数作决定。"})
    atomic_csv(fixed_rows, out / "tables/fixed_parameter_comparison.csv")
    atomic_csv(old_rows, out / "tables/old_vs_extended_cv.csv")
    cols = ["model", "method", "k", "parameter", "selected_value", "numeric_min", "numeric_max", "note"]
    atomic_csv(pd.DataFrame(boundaries, columns=cols), out / "tables/boundary_diagnostics.csv")


def lock_selection(out: Path, selected: pd.DataFrame) -> dict:
    payload = {"selected": json.loads(selected.to_json(orient="records", double_precision=15)),
               "cv_sha256": sha256(out / "tables/cv_candidates.csv"),
               "config_sha256": sha256(out / "config_frozen.json"),
               "search_plan_sha256": sha256(out / "search_plan.json")}
    path = out / "selection_lock.json"
    if path.is_file():
        old = read_json(path)
        if any(canonical(old[key]) != canonical(value) for key, value in payload.items()):
            raise ValueError("锁定配置与当前计算不一致；停止恢复，不能在已查看测试后悄悄改参数。")
        return old
    payload["locked_at_utc"] = utc_now()
    atomic_json(path, payload)
    return payload


def run_extended(root: Path, config: dict, plan: dict, out: Path,
                 workers: int = 2, resume: bool = False) -> Path:
    """搜索使用train数组；选择锁定并拟合后才加载test数组。"""
    if workers < 1:
        raise ValueError("--workers必须为正整数。")
    out = Path(out).resolve()
    context = {"version": VERSION, "config": config, "plan": plan,
               "data_hashes": file_hashes(root), "code_hashes": source_hashes(),
               "versions": versions(), "platform": platform.platform()}
    from har_study.stage_timing import load_timing_config
    stage_config = load_timing_config(ROOT / "timing_config.json")
    context["stage_timing_config"] = stage_config
    token = fingerprint(context)
    if resume:
        if not out.is_dir() or not (out / "resume_context.json").is_file():
            raise FileNotFoundError("恢复目录没有适度扩展搜索的断点上下文。")
        if read_json(out / "resume_context.json")["context_sha256"] != token:
            raise ValueError("数据、配置、脚本、软件版本或平台发生变化。不能复用这批断点，请新建实验。")
        if read_json(out / "manifest.json").get("status") == "completed":
            print("[Resume] Run already complete; validating and redrawing only.", flush=True)
            return out
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"输出目录非空，禁止覆盖：{out}")
        out.mkdir(parents=True, exist_ok=True)
        for name in ("tables", "models", "figures", "splits", "checkpoints"):
            (out / name).mkdir(exist_ok=True)
        atomic_json(out / "config_frozen.json", config)
        atomic_json(out / "timing_config_frozen.json", stage_config)
        atomic_json(out / "search_plan.json", plan)
        atomic_json(out / "source_hashes.json", context["code_hashes"])
        atomic_json(out / "resume_context.json", {"context_sha256": token, "context": context})
        atomic_json(out / "protocol_frozen.json", {
            "created_at_utc": utc_now(), "config_sha256": sha256(out / "config_frozen.json"),
            "source_data_hashes": context["data_hashes"],
            "source_code_hashes_sha256": sha256(out / "source_hashes.json"),
            "search_plan_sha256": sha256(out / "search_plan.json"),
            "selection": "全特征、各维数同预算调参；选验证F1下降不超过0.01的最小维数；未达标保留诊断。",
            "prior_test_exposure": True,
            "interpretation": "扩展探索性实验，不是从未接触测试集的独立确认；不得用测试成绩回头改网格。",
            "not_preregistered": "本地配置及事件记录不等于第三方预注册。",
        })
    def event(name, **values):
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": utc_now(), "event": name, **values}, ensure_ascii=False) + "\n")
    manifest = read_json(out / "manifest.json") if (out / "manifest.json").is_file() else {
        "started_at_utc": utc_now(), "dataset": "UCI HAR", "synthetic": False,
        **versions(), "seed": config["seed"], "models": config["models"],
        "prior_test_exposure": True, "search_engine": VERSION,
    }
    manifest.update(status="running", workers=workers, resumed=resume)
    manifest.pop("error", None)
    atomic_json(out / "manifest.json", manifest)
    event("resumed" if resume else "protocol_frozen")
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            train = load_split(root, "train")
            folds, assignments = core.grouped_folds(train, config)
            atomic_csv(assignments, out / "splits/training_cv.csv")
            event("training_loaded", n=len(train.y), n_subjects=len(np.unique(train.subjects)))
            candidates = run_search(train, config, plan, folds, out, workers, token)
            comparison_tables(candidates, config, out)
            selected, curve = core.select_configs(candidates, config["allowed_macro_f1_drop"])
            atomic_csv(curve, out / "tables/compression_curve.csv")
            atomic_csv(selected, out / "tables/selected_configs.csv")
            lock_selection(out, selected)
            event("selection_locked", sha256=sha256(out / "selection_lock.json"))
            # 参数锁定后不会根据以下协议对照/测试结果更改配置。
            core.protocol_comparison(train, config, folds, out)
            fitted = core.fit_selected(train, selected, config, out)
            event("final_models_fitted")
            test = load_split(root, "test")
            atomic_json(out / "data_audit.json", audit_splits(train, test))
            atomic_csv([{"row_id": i, "subject": int(s), "activity": int(y)}
                        for i, (s, y) in enumerate(zip(test.subjects, test.y))],
                       out / "splits/official_test.csv")
            event("test_loaded", n=len(test.y), n_subjects=len(np.unique(test.subjects)))
            core.evaluate_test(test, fitted, selected, out)
            event("test_evaluated")
            core.paired_subject_bootstrap(out, config)
            core.benchmark(fitted, test.X, config, out)
            from har_study.diagnostics import derive_decisions
            derive_decisions(out)
            event("benchmark_completed")
            from har_study.stage_timing import benchmark_stages, verify_stage_timing
            benchmark_stages(out, test.X, fitted, stage_config)
            stage_checks = verify_stage_timing(out)
            event("stage_timing_completed", rows=stage_checks["n_raw_rows"],
                  parity_predictions=stage_checks["n_parity_predictions"])
        manifest.update(status="completed", completed_at_utc=utc_now(),
            elapsed_seconds=time.perf_counter() - start,
            selection_lock_sha256=sha256(out / "selection_lock.json"),
            n_candidates=plan["n_candidates"], n_cv_fits=plan["n_cv_fits"],
            n_final_models=plan["n_final_models"], n_protocol_fits=plan["n_protocol_fits"],
            stage_timing_completed=True,
            elapsed_scope="当前调用用时；断点此前耗时见各任务记录，不冒称从头运行总时长。")
        atomic_json(out / "manifest.json", manifest)
        event("completed")
        shutil.rmtree(out / ".extended_cache", ignore_errors=True)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=repr(exc), failed_at_utc=utc_now())
        atomic_json(out / "manifest.json", manifest)
        event("failed", error=repr(exc))
        raise
    return out


def assert_close(a, b, message: str) -> None:
    left, right = np.asarray(a), np.asarray(b)
    if left.shape != right.shape or not np.allclose(left, right, rtol=0, atol=1e-9, equal_nan=False):
        raise AssertionError(message)


def verify_extended(out: Path, data_root: Path) -> dict:
    """核验冻结网格、人员隔离和指标复算，不重训整个搜索。"""
    out = Path(out)
    base = verify_run(out)
    config = read_json(out / "config_frozen.json")
    plan = read_json(out / "search_plan.json")
    protocol = read_json(out / "protocol_frozen.json")
    if canonical(plan) != canonical(build_search_plan(config)):
        raise AssertionError("搜索计划无法由冻结配置复算。")
    if protocol["search_plan_sha256"] != sha256(out / "search_plan.json"):
        raise AssertionError("搜索计划在冻结后变化。")
    if protocol["config_sha256"] != sha256(out / "config_frozen.json"):
        raise AssertionError("配置在冻结后变化。")
    if protocol["source_code_hashes_sha256"] != sha256(out / "source_hashes.json"):
        raise AssertionError("保存的代码哈希清单发生变化。")
    if protocol["source_data_hashes"] != file_hashes(data_root):
        raise AssertionError("原始数据与运行时哈希不同。")
    train, test = load_split(data_root, "train"), load_split(data_root, "test")
    if audit_splits(train, test) != read_json(out / "data_audit.json"):
        raise AssertionError("重新读取官方数据后的人员/样本检查不一致。")
    _, assignment = core.grouped_folds(train, config)
    actual_assignment = pd.read_csv(out / "splits/training_cv.csv").sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.DataFrame(assignment).sort_values("row_id").reset_index(drop=True), actual_assignment)
    split = pd.read_csv(out / "splits/official_test.csv").sort_values("row_id")
    if not np.array_equal(split.activity, test.y) or not np.array_equal(split.subject, test.subjects):
        raise AssertionError("测试真值/人员编号不是官方文件中的顺序。")
    specs = pd.DataFrame(candidate_specs(config, plan, train.X.shape[1])).set_index("candidate_id")
    table = pd.read_csv(out / "tables/cv_candidates.csv")
    if not table.candidate_id.is_unique or set(table.candidate_id) != set(specs.index):
        raise AssertionError("候选缺失、多出或重复。")
    for row in table.itertuples():
        expected = specs.loc[row.candidate_id]
        if row.model != expected.model or row.method != expected.method or row.k != expected.k:
            raise AssertionError("候选的模型、表示或维数被改变。")
        if canonical(json.loads(row.params_json)) != canonical(json.loads(expected.params_json)):
            raise AssertionError("实际候选参数与采样计划不符。")
    folds = pd.read_csv(out / "tables/cv_folds.csv")
    expected_keys = {(cid, f) for cid in specs.index for f in range(config["cv_folds"])}
    if len(folds) != len(expected_keys) or set(zip(folds.candidate_id, folds.fold)) != expected_keys:
        raise AssertionError("验证折记录有重复或缺失。")
    selected = pd.read_csv(out / "tables/selected_configs.csv")
    expected_ids = {f"{m}__{method}" for m in config["models"] for method in METHODS}
    if len(selected) != len(expected_ids) or set(selected.selection_id) != expected_ids:
        raise AssertionError("最终方案数量应等于配置中的模型数乘以3。")
    predictions = pd.read_csv(out / "tables/test_predictions.csv")
    overall = pd.read_csv(out / "tables/test_metrics.csv")
    activities = pd.read_csv(out / "tables/test_activities.csv")
    subjects = pd.read_csv(out / "tables/test_subjects.csv")
    if set(predictions.selection_id) != expected_ids or len(predictions) != len(test.y) * len(expected_ids):
        raise AssertionError("测试预测未覆盖全部锁定方案。")
    labels = sorted(ACTIVITIES)
    for sid in sorted(expected_ids):
        rows = predictions[predictions.selection_id == sid].sort_values("row_id")
        if not set(rows.predicted) <= set(labels):
            raise AssertionError("预测包含未知活动类别。")
        matrix = confusion_matrix(rows.actual, rows.predicted, labels=labels)
        stored = pd.read_csv(out / "tables" / f"confusion_{sid}.csv", index_col=0)
        names = [ACTIVITIES[i] for i in labels]
        if list(stored.index) != names or list(stored.columns) != names:
            raise AssertionError("混淆矩阵类别顺序不一致。")
        assert_close(stored.to_numpy(), matrix, f"{sid}混淆矩阵无法从预测复算。")
        p, r, f, n = precision_recall_fscore_support(rows.actual, rows.predicted, labels=labels, zero_division=0)
        category = activities[activities.selection_id == sid].sort_values("activity")
        for key, expected in (("precision", p), ("recall", r), ("f1", f), ("support", n)):
            assert_close(category[key], expected, f"{sid}逐类别{key}不一致。")
        people = subjects[subjects.selection_id == sid]
        if set(people.subject) != set(test.subjects) or len(people) != len(set(test.subjects)):
            raise AssertionError("人员指标遗漏或重复。")
        score = overall[overall.selection_id == sid]
        if len(score) != 1:
            raise AssertionError("测试指标缺失或重复。")
        for key, expected in (("subject_macro_f1_mean", people.macro_f1.mean()),
                              ("subject_macro_f1_min", people.macro_f1.min()),
                              ("subject_macro_f1_std", people.macro_f1.std(ddof=1))):
            assert_close(score.iloc[0][key], expected, "人员汇总分数不一致。")
        meta = read_json(out / "models" / f"{sid}.json")
        selected_row = selected[selected.selection_id == sid].iloc[0]
        if canonical(json.loads(meta["selection"]["params_json"])) != canonical(json.loads(selected_row.params_json)):
            raise AssertionError("保存模型参数说明与锁定配置不一致。")
    raw = pd.read_csv(out / "tables/timing_raw.csv")
    timing = pd.read_csv(out / "tables/timing_summary.csv")
    for row in timing.itertuples():
        values = raw[(raw.selection_id == row.selection_id) & (raw.kind == row.kind)].ms_per_sample
        if len(values) != config["timing_repeats"] or not np.isfinite(values).all() or (values <= 0).any():
            raise AssertionError("计时轮数或数值异常。")
        assert_close([row.median_ms_per_sample, row.q25_ms_per_sample, row.q75_ms_per_sample],
                     [values.median(), values.quantile(.25), values.quantile(.75)], "计时汇总不一致。")
    extra = {**base, "extended_search_plan": "PASS", "confusion_matrix_recomputed": "PASS",
             "class_metrics_recomputed": "PASS", "original_data_matched": "PASS",
             "timing_quantiles_recomputed": "PASS", "all_configured_combinations_evaluated": True, "checked_at_utc": utc_now(),
             "scope": "复算和文件一致性检查，不是全部参数重训；本地记录不是不可篡改的证据。"}
    atomic_json(out / "verification.json", extra)
    return extra


def verify_and_draw(out: Path, data_root: Path, font_path=None) -> None:
    result = verify_extended(out, data_root)
    if (out / "stage_timing_environment.json").is_file():
        from har_study.stage_timing import verify_stage_timing
        stage_result = verify_stage_timing(out)
        print(f"[Verify] Stage timing PASS: {stage_result['n_raw_rows']} records", flush=True)
    elif read_json(out / "manifest.json").get("stage_timing_completed"):
        raise ValueError("本轮宣称已完成分阶段计时，但没有原始计时记录。")
    print(f"[Verify] PASS: {result['n_cv_candidates']} candidates; predictions, confusion matrices and metrics agree.", flush=True)
    try:
        from har_study.plots import draw_all
        standard = draw_all(out, font_path)
        if (out / "stage_timing_environment.json").is_file():
            from har_study.stage_timing import draw_stage_figures
            standard += draw_stage_figures(out, font_path)
        error_path = out / "extended_figure_error.json"
        error_path.unlink(missing_ok=True)
        print(f"[Figures] {len(standard)} clean charts generated.", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        atomic_json(out / "extended_figure_error.json", {"error": repr(exc), "time_utc": utc_now()})
        print(f"[Figures] {exc}\n[Figures] Numeric results remain valid. Fix the font/layout and use --verify-only to redraw.", file=sys.stderr)


def describe_plan(plan: dict) -> None:
    for model, n in plan["parameter_counts"].items():
        kind = "finite exhaustive grid"
        print(f"[Plan] {model}: {n} combinations per representation; {kind}")
    print(f"[Plan] {plan['representations_per_model']} representations/model; "
          f"{plan['n_candidates']} candidates; {plan['n_cv_fits']} primary CV fits; "
          f"{plan['n_protocol_fits']} fixed-parameter protocol fits; {plan['n_final_models']} final models.")


def main() -> int:
    parser = argparse.ArgumentParser(description="HAR五模型适度扩展搜索：独立入口，保存断点，核验并绘制混淆矩阵/参数热力图")
    parser.add_argument("--data", type=Path, help="官方ZIP或已解压目录；默认使用项目data/")
    parser.add_argument("--out", type=Path, help="新输出目录；默认outputs/compact_时间，不覆盖旧结果")
    parser.add_argument("--workers", type=int, default=2, help="搜索进程数，默认2；计时仍为单线程")
    parser.add_argument("--font", type=Path, help="可选：本机中文字体文件")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true", help="只打印搜索规模，不读取数据或训练")
    actions.add_argument("--resume", type=Path, help="继续中断的适度扩展搜索；配置和版本必须不变")
    actions.add_argument("--verify-only", type=Path, help="复核已完成的扩展结果并重绘图片，不重新训练")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers必须为正整数")
    if args.out is not None and (args.resume or args.verify_only):
        parser.error("--out不能与--resume/--verify-only同时使用")
    if not all(hasattr(core, name) for name in ("make_classifier", "fit_selected", "evaluate_test", "select_configs")):
        parser.error("当前项目不是所需的五模型整合版本。")
    if args.resume:
        if not all((args.resume / name).is_file() for name in ("config_frozen.json", "search_plan.json", "resume_context.json")):
            parser.error("--resume指定的目录缺少适度扩展搜索断点文件；请填写实际outputs/compact_目录。")
        config = read_json(args.resume / "config_frozen.json")
        plan = read_json(args.resume / "search_plan.json")
        if canonical(config) != canonical(build_config()):
            parser.error("恢复配置与当前脚本默认配置不同。请使用原脚本，不要改变已开始实验的配置。")
    else:
        config = build_config()
        try:
            plan = build_search_plan(config)
        except ValueError as exc:
            parser.error(str(exc))
    if args.plan:
        describe_plan(plan)
        return 0
    try:
        root = prepare_dataset(args.data, ROOT / "data")
        if args.verify_only:
            verify_and_draw(args.verify_only, root, args.font)
            return 0
        describe_plan(plan)
        out = args.resume or args.out or ROOT / "outputs" / datetime.now().strftime("compact_%Y%m%d_%H%M%S_%f")
        run_extended(root, config, plan, out, args.workers, resume=bool(args.resume))
        verify_and_draw(out, root, args.font)
        print(f"[Done] {out.resolve()}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\n[Stopped] Completed folds are saved. Continue using --resume <output-directory>.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"[STOP] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
