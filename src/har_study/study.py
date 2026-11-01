"""训练、选模和测试。测试集只在参数及压缩维数锁定后加载。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

from .data import ACTIVITIES, Split, audit_splits, file_hashes, load_split, sha256

LABELS = list(ACTIVITIES)
MODELS = ("RBF_SVM", "RandomForest", "LogisticRegression", "KNN", "FisherLDA")
GRID_KEYS = {"RBF_SVM": "svm_grid", "RandomForest": "rf_grid", "LogisticRegression": "lr_grid", "KNN": "knn_grid", "FisherLDA": "lda_grid"}
METHODS = ("full", "pca", "anova")
SCORING = {
    "macro_f1": make_scorer(f1_score, labels=LABELS, average="macro", zero_division=0),
    "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, obj: dict | list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, Path): return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert,
                                    allow_nan=False), encoding="utf-8")


def save_table(rows, path: Path) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")
    return frame


def validate_config(c: dict, *, n_features: int = 561) -> None:
    required = {"seed", "cv_folds", "dimensions", "allowed_macro_f1_drop", "svm_grid",
                "rf_grid", "rf_trees", "timing_repeats", "timing_warmups",
                "latency_sample_count", "bootstrap_repeats", "models", "lr_grid", "knn_grid", "lda_grid"}
    if set(c) != required:
        raise ValueError(f"配置字段不符：缺少 {required - c.keys()}，多出 {c.keys() - required}")
    if not c["models"] or len(set(c["models"])) != len(c["models"]) or not set(c["models"]) <= set(MODELS):
        raise ValueError("models需要是已支持模型的非空、不重复列表。")
    if not isinstance(c["seed"], int) or c["cv_folds"] < 2:
        raise ValueError("随机种子必须是整数，交叉验证至少2折。")
    if c["dimensions"] != sorted(set(c["dimensions"])) or not c["dimensions"]:
        raise ValueError("dimensions 必须为非空、递增、无重复列表。")
    if any(not isinstance(k, int) or k <= 0 or k >= n_features for k in c["dimensions"]):
        raise ValueError("压缩维数必须小于输入维数。")
    if not 0 <= c["allowed_macro_f1_drop"] < 1:
        raise ValueError("性能容忍范围应以0～1的小数填写；0.01表示1个百分点。")
    for key in ("rf_trees", "timing_repeats", "timing_warmups", "latency_sample_count", "bootstrap_repeats"):
        if not isinstance(c[key], int) or c[key] < 1: raise ValueError(f"{key} 必须为正整数。")
    if set(c["svm_grid"]) != {"model__C", "model__gamma"}:
        raise ValueError("SVM 网格只接受 model__C 和 model__gamma。")
    if set(c["rf_grid"]) != {"model__max_depth", "model__min_samples_leaf"}:
        raise ValueError("随机森林网格字段不符。")
    if set(c["lr_grid"]) != {"model__C"} or set(c["knn_grid"]) != {"model__n_neighbors", "model__weights"}:
        raise ValueError("逻辑回归或KNN网格字段不符。")
    if set(c["lda_grid"]) != {"model__shrinkage"}:
        raise ValueError("FisherLDA只搜索协方差收缩方式，固定lsqr求解器。")
    values = c["lda_grid"]["model__shrinkage"]
    # 这两项始终作为基准，避免扩展数值网格后遗漏原有对照。
    if None not in values or "auto" not in values:
        raise ValueError("FisherLDA参数网格必须保留None和auto两个基准设置。")
    if len(values) != len(set(values)):
        raise ValueError("FisherLDA收缩网格含重复取值。")
    for value in values:
        if value not in (None, "auto"):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
                raise ValueError("LDA shrinkage必须是None、auto或0至1的数值。")
    if any(v <= 0 for v in c["lr_grid"]["model__C"] + c["svm_grid"]["model__C"]):
        raise ValueError("C必须为正数。")
    if any(not isinstance(k, int) or k < 1 for k in c["knn_grid"]["model__n_neighbors"]):
        raise ValueError("KNN邻居数必须为正整数。")
    if not set(c["knn_grid"]["model__weights"]) <= {"uniform", "distance"}:
        raise ValueError("KNN投票方式不符。")
    if any(not v for key in GRID_KEYS.values() for v in c[key].values()):
        raise ValueError("参数候选列表不能为空。")


def make_pipeline(model: str, method: str, k: int, c: dict, memory=None) -> Pipeline:
    """所有数据驱动的变换都在 Pipeline 内；每折只拟合该折训练人员。"""
    if model not in MODELS or method not in METHODS:
        raise ValueError("未知模型或特征方法。")
    reducer = {"full": "passthrough"}.get(method)
    if method == "pca":
        # 固定完整SVD，不使用随机近似；不同维数不依赖随机初始化。
        reducer = PCA(n_components=k, svd_solver="full", whiten=False)
    elif method == "anova":
        # ANOVA F值只用于排序。窗口不独立，不能把F检验p值解释为可靠的显著性证据。
        reducer = SelectKBest(score_func=f_classif, k=k)
    classifier = make_classifier(model, c)
    # 所有路径都用同一标准化策略，避免把压缩收益与额外预处理差异混在一起。
    return Pipeline([("scale", StandardScaler()), ("reduce", reducer), ("model", classifier)], memory=memory)


def make_classifier(model: str, c: dict):
    """仅构造分类器；最终保存和预测仍使用标准sklearn Pipeline。"""
    if model == "RBF_SVM":
        return SVC(kernel="rbf", C=10.0, gamma="scale", probability=False,
                   cache_size=512, random_state=c["seed"])
    if model == "RandomForest":
        return RandomForestClassifier(n_estimators=c["rf_trees"], max_features="sqrt",
                                      random_state=c["seed"], n_jobs=1)
    if model == "LogisticRegression":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=8000, tol=1e-4,
                                  random_state=c["seed"])
    if model == "KNN":
        return KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2,
                                    algorithm="brute", n_jobs=1)
    if model == "FisherLDA":
        # 多类Fisher判别的LDA分类基线，不把LDA当作PCA的替代降维器。
        # lsqr允许无收缩/自动收缩；先验按当前训练折类别比例估计。
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None, priors=None)
    raise ValueError(f"未知分类器：{model}")


def grouped_folds(data: Split, c: dict) -> tuple[list, list]:
    folds = list(StratifiedGroupKFold(n_splits=c["cv_folds"], shuffle=True,
                 random_state=c["seed"]).split(data.X, data.y, groups=data.subjects))
    records = []
    seen = np.zeros(len(data.y), dtype=int)
    for fold_id, (tr, va) in enumerate(folds):
        if set(data.subjects[tr]) & set(data.subjects[va]):
            raise AssertionError("交叉验证发生人员泄漏。")
        if set(data.y[tr]) != set(LABELS) or set(data.y[va]) != set(LABELS):
            raise ValueError("某折缺少活动类别，请检查数据或人员划分。")
        seen[va] += 1
        for row in va:
            records.append({"row_id": int(row), "subject": int(data.subjects[row]),
                            "activity": int(data.y[row]), "validation_fold": fold_id})
    if not np.all(seen == 1): raise AssertionError("每个训练样本必须恰好验证一次。")
    return folds, records


def metrics(y, pred) -> dict:
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "correct": int(np.sum(np.asarray(y) == np.asarray(pred))), "n": len(y)}


def search_candidates(data: Split, c: dict, folds: list, out: Path, workers: int, cache: Path) -> pd.DataFrame:
    """折内共享变换的精确网格搜索；分类器/参数逐个实际拟合，不读取旧成绩。"""
    from .search import cached_grid_search
    return cached_grid_search(data, c, folds, out, workers, cache)


def select_configs(candidates: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每条压缩路线选满足容忍范围的最小k。没有达标时保留最佳诊断方案并明确标失败。"""
    order = candidates.sort_values(["cv_macro_f1", "candidate_id"], ascending=[False, True], kind="stable")
    curve = order.drop_duplicates(["model", "method", "k"]).copy()
    selections = []
    for model in [m for m in MODELS if m in set(curve.model)]:
        baseline = curve[(curve.model == model) & (curve.method == "full")].iloc[0]
        threshold = float(baseline.cv_macro_f1) - tolerance
        for method in METHODS:
            options = curve[(curve.model == model) & (curve.method == method)].copy()
            eligible = options[options.cv_macro_f1 >= threshold - 1e-12]
            passed = not eligible.empty
            if method == "full":
                chosen = baseline
            elif passed:
                chosen = eligible.sort_values(["k", "cv_macro_f1", "candidate_id"], ascending=[True, False, True]).iloc[0]
            else:
                chosen = options.sort_values(["cv_macro_f1", "k", "candidate_id"], ascending=[False, True, True]).iloc[0]
            row = chosen.to_dict()
            row.update({"selection_id": f"{model}__{method}", "cv_threshold": threshold,
                        "meets_cv_tolerance": passed, "cv_delta_from_full_pp":
                        100 * (float(chosen.cv_macro_f1) - float(baseline.cv_macro_f1)),
                        "selection_rule": "baseline" if method == "full" else
                        "smallest_eligible_k" if passed else "best_cv_but_failed_tolerance"})
            selections.append(row)
    return pd.DataFrame(selections), curve.sort_values(["model", "method", "k"])


def protocol_comparison(data: Split, c: dict, grouped: list, out: Path) -> None:
    """固定参数比较两种验证协议，只在官方训练集内进行，不参与选模。"""
    random_folds = list(StratifiedKFold(n_splits=c["cv_folds"], shuffle=True,
                        random_state=c["seed"]).split(data.X, data.y))
    records, predictions = [], []
    for model in c["models"]:
        for protocol, folds in (("subject_grouped", grouped), ("random_windows", random_folds)):
            for fid, (tr, va) in enumerate(folds):
                pipe = make_pipeline(model, "full", data.X.shape[1], c)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    pipe.fit(data.X[tr], data.y[tr])
                pred = pipe.predict(data.X[va])
                score = metrics(data.y[va], pred)
                records.append({"model": model, "protocol": protocol, "fold": fid,
                                "train_n": len(tr), "validation_n": len(va),
                                "overlap_subject_count": len(set(data.subjects[tr]) & set(data.subjects[va])), **score})
                for row, val in zip(va, pred):
                    predictions.append({"model": model, "protocol": protocol, "fold": fid,
                        "row_id": int(row), "subject": int(data.subjects[row]),
                        "actual": int(data.y[row]), "predicted": int(val)})
            print(f"[Protocol] {model} / {protocol} finished (fixed parameters)", flush=True)
    save_table(records, out / "tables/protocol_comparison.csv")
    save_table(predictions, out / "tables/protocol_predictions.csv")


def fit_selected(train: Split, selected: pd.DataFrame, c: dict, out: Path) -> dict:
    fitted, fit_rows, features, pca_rows = {}, [], [], []
    for row in selected.to_dict("records"):
        sid = row["selection_id"]
        pipe = make_pipeline(row["model"], row["method"], int(row["k"]), c)
        pipe.set_params(**json.loads(row["params_json"]))
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipe.fit(train.X, train.y)
        fit_seconds = time.perf_counter() - start
        path = out / "models" / f"{sid}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 保存包含变换器的完整Pipeline；体积比较不能只保存末端分类器。
        joblib.dump(pipe, path, compress=3)
        fitted[sid] = pipe
        fit_rows.append({"selection_id": sid, "final_fit_seconds": fit_seconds,
                         "pipeline_bytes": path.stat().st_size,
                         "model_sha256": sha256(path),
                         "support_vector_count": int(pipe.named_steps['model'].support_vectors_.shape[0])
                         if row["model"] == "RBF_SVM" else None})
        if row["method"] == "anova":
            reducer = pipe.named_steps["reduce"]
            for j in reducer.get_support(indices=True):
                features.append({"selection_id": sid, "feature_index_1based": int(j) + 1,
                    "feature_name": train.feature_names[j], "f_score": float(reducer.scores_[j])})
        elif row["method"] == "pca":
            reducer = pipe.named_steps["reduce"]
            for j, fraction in enumerate(reducer.explained_variance_ratio_):
                pca_rows.append({"selection_id": sid, "component": j + 1,
                    "explained_variance_ratio": float(fraction),
                    "cumulative_ratio": float(reducer.explained_variance_ratio_[:j + 1].sum())})
        dump_json(out / "models" / f"{sid}.json", {
            "selection": row, "input_features": len(train.feature_names),
            "feature_names": train.feature_names, "activities": ACTIVITIES,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "warning": "仅加载可信的joblib文件。此模型输入是561维官方特征，不是原始传感器序列。",
        })
    save_table(fit_rows, out / "tables/final_fit_and_size.csv")
    save_table(features, out / "tables/selected_features.csv")
    save_table(pca_rows, out / "tables/pca_variance.csv")
    return fitted


def evaluate_test(test: Split, fitted: dict, selected: pd.DataFrame, out: Path) -> None:
    overall, per_subject, per_class, predictions = [], [], [], []
    for row in selected.to_dict("records"):
        sid = row["selection_id"]
        pred = fitted[sid].predict(test.X)
        person_scores = []
        for person in sorted(np.unique(test.subjects)):
            mask = test.subjects == person
            score = metrics(test.y[mask], pred[mask])
            person_scores.append(score["macro_f1"])
            per_subject.append({"selection_id": sid, "subject": int(person), **score})
        overall.append({"selection_id": sid, "model": row["model"], "method": row["method"],
            "k": int(row["k"]), "dimension_reduction": 1 - int(row["k"]) / test.X.shape[1],
            **metrics(test.y, pred), "subject_macro_f1_mean": float(np.mean(person_scores)),
            "subject_macro_f1_min": float(np.min(person_scores)),
            "subject_macro_f1_std": float(np.std(person_scores, ddof=1))})
        p, r, f, n = precision_recall_fscore_support(test.y, pred, labels=LABELS, zero_division=0)
        for cls, pp, rr, ff, nn in zip(LABELS, p, r, f, n):
            per_class.append({"selection_id": sid, "activity": cls, "activity_name": ACTIVITIES[cls],
                              "precision": pp, "recall": rr, "f1": ff, "support": int(nn)})
        for idx, (actual, predicted, subject) in enumerate(zip(test.y, pred, test.subjects)):
            predictions.append({"selection_id": sid, "row_id": idx, "subject": int(subject),
                                "actual": int(actual), "predicted": int(predicted)})
        cm = pd.DataFrame(confusion_matrix(test.y, pred, labels=LABELS),
                          index=[ACTIVITIES[i] for i in LABELS], columns=[ACTIVITIES[i] for i in LABELS])
        cm.to_csv(out / "tables" / f"confusion_{sid}.csv", encoding="utf-8-sig", index_label="actual")
        print(f"[Test] {sid} / k={row['k']}: Accuracy={overall[-1]['accuracy']:.4%}, Macro-F1={overall[-1]['macro_f1']:.4%}", flush=True)
    save_table(overall, out / "tables/test_metrics.csv")
    save_table(per_subject, out / "tables/test_subjects.csv")
    save_table(per_class, out / "tables/test_activities.csv")
    save_table(predictions, out / "tables/test_predictions.csv")


def paired_subject_bootstrap(out: Path, c: dict) -> None:
    """重采样单位是人，不把重叠窗口当独立样本；区间仅反映这9人及固定模型的变动。"""
    subjects = pd.read_csv(out / "tables/test_subjects.csv")
    rows = []
    for model in c["models"]:
        baseline = subjects[subjects.selection_id == f"{model}__full"].set_index("subject").sort_index()
        for method in ("pca", "anova"):
            other = subjects[subjects.selection_id == f"{model}__{method}"].set_index("subject").reindex(baseline.index)
            delta = other.macro_f1.to_numpy() - baseline.macro_f1.to_numpy()
            rng = np.random.default_rng(c["seed"])
            sampled = delta[rng.integers(0, len(delta), size=(c["bootstrap_repeats"], len(delta)))].mean(axis=1)
            lo, hi = np.quantile(sampled, [0.025, 0.975])
            rows.append({"model": model, "method": method, "n_subjects": len(delta),
                "mean_subject_delta_pp": float(delta.mean() * 100), "ci95_low_pp": float(lo * 100),
                "ci95_high_pp": float(hi * 100), "bootstrap_repeats": c["bootstrap_repeats"],
                "interpretation": "描述性配对人员重采样区间；不含训练不确定性，不是全人群非劣效性证明。"})
    save_table(rows, out / "tables/subject_bootstrap.csv")


def benchmark(fitted: dict, X: np.ndarray, c: dict, out: Path) -> None:
    """计入标准化、压缩与分类；不含原始信号特征提取、磁盘读取和模型加载。"""
    ids = sorted(fitted)
    indices = np.linspace(0, len(X) - 1, min(len(X), c["latency_sample_count"]), dtype=int)
    singles = [np.ascontiguousarray(X[i:i + 1]) for i in indices]
    batch = np.ascontiguousarray(X)
    rows = []
    rng = np.random.default_rng(c["seed"])
    with threadpool_limits(limits=1):
        for sid in ids:
            for _ in range(c["timing_warmups"]):
                fitted[sid].predict(batch)
                fitted[sid].predict(singles[0])
        # 交错模型顺序，减小CPU温度/机器负载随时间变化的单向偏差。
        for repeat in range(c["timing_repeats"]):
            for sid in rng.permutation(ids):
                start = time.perf_counter_ns()
                fitted[sid].predict(batch)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                rows.append({"selection_id": sid, "repeat": repeat, "kind": "batch",
                             "samples": len(batch), "total_ms": elapsed_ms,
                             "ms_per_sample": elapsed_ms / len(batch)})
                start = time.perf_counter_ns()
                for one in singles: fitted[sid].predict(one)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                rows.append({"selection_id": sid, "repeat": repeat, "kind": "single",
                             "samples": len(singles), "total_ms": elapsed_ms,
                             "ms_per_sample": elapsed_ms / len(singles)})
    raw = save_table(rows, out / "tables/timing_raw.csv")
    summary = []
    for (sid, kind), sub in raw.groupby(["selection_id", "kind"]):
        values = sub.ms_per_sample.to_numpy()
        summary.append({"selection_id": sid, "kind": kind, "repeats": len(values),
            "median_ms_per_sample": float(np.median(values)), "q25_ms_per_sample": float(np.quantile(values, .25)),
            "q75_ms_per_sample": float(np.quantile(values, .75))})
    save_table(summary, out / "tables/timing_summary.csv")
    dump_json(out / "timing_environment.json", {"platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(), "threadpool_info": threadpool_info(),
        "thread_limit_during_measurement": 1, "rf_n_jobs": 1,
        "single_measure": "重复调用单窗口predict，每轮总耗时/窗口数；含Python循环开销。",
        "excluded": "传感器采集、原始特征提取、模型加载、文件IO、手机功耗。"})


