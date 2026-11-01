"""折内共享预处理的网格搜索。

关键边界：先按人员划分，再仅用当前折的训练人员拟合变换。
缓存是计算复用，不是用全体人员预处理后再分折；测试集不会进入本模块。
每个候选的分类器仍独立训练。完整SVD的前k分量等于同一分解的前缀。
最终模型仍由标准Pipeline重新拟合全部训练人员，不依赖这些临时缓存。
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def prepare_fold(data, train, validation, max_k: int, folder: Path):
    if set(data.subjects[train]) & set(data.subjects[validation]):
        raise ValueError("人员交叉；禁止缓存泄漏的验证表示。")
    if max_k > min(len(train), data.X.shape[1]):
        raise ValueError("PCA维数大于本折允许范围。")
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data.X[train])
    X_valid = scaler.transform(data.X[validation])
    pca = PCA(n_components=max_k, svd_solver="full", whiten=False)
    pc_train = pca.fit_transform(X_train)
    pc_valid = pca.transform(X_valid)
    selector = SelectKBest(f_classif, k="all").fit(X_train, data.y[train])
    # 与SelectKBest相同的稳定排序及NaN处理。选择后的列按原始列序返回。
    scores = np.asarray(selector.scores_).copy()
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    ranking = np.argsort(scores, kind="mergesort")
    for name, value in (("train_scaled", X_train), ("valid_scaled", X_valid),
                        ("train_pca", pc_train), ("valid_pca", pc_valid),
                        ("y_train", data.y[train]), ("y_valid", data.y[validation]),
                        ("anova_ranking", ranking)):
        np.save(folder / f"{name}.npy", np.ascontiguousarray(value))
    return {"train_n": len(train), "validation_n": len(validation),
            "train_subjects": sorted(map(int, np.unique(data.subjects[train]))),
            "validation_subjects": sorted(map(int, np.unique(data.subjects[validation]))),
            "max_pca_dimension": max_k, "preprocessing_seconds": time.perf_counter() - t0,
            "scaler_mean": scaler.mean_.tolist(), "fitted_on": "training rows of this fold only"}


def representation(folder: Path, method: str, k: int):
    prefix = "pca" if method == "pca" else "scaled"
    train = np.load(folder / f"train_{prefix}.npy", mmap_mode="r")
    valid = np.load(folder / f"valid_{prefix}.npy", mmap_mode="r")
    if method == "pca":
        train, valid = train[:, :k], valid[:, :k]
    elif method == "anova":
        indices = np.sort(np.load(folder / "anova_ranking.npy")[-k:])
        train, valid = train[:, indices], valid[:, indices]
    elif method != "full":
        raise ValueError(method)
    return (np.ascontiguousarray(train), np.ascontiguousarray(valid),
            np.load(folder / "y_train.npy"), np.load(folder / "y_valid.npy"))


def score_fold(candidate: dict, fold: int, folder: Path, config: dict):
    # 外层进程并行，内层BLAS和森林/KNN均单线程，避免线程数量相乘。
    from .study import make_classifier, metrics
    X, V, y, truth = representation(folder, candidate["method"], candidate["k"])
    model = make_classifier(candidate["model"], config)
    parameters = json.loads(candidate["params_json"])
    model.set_params(**{key.removeprefix("model__"): value for key, value in parameters.items()})
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        predicted = model.predict(V)
        predict_s = time.perf_counter() - t0
    return {"candidate_id": candidate["candidate_id"], "fold": fold,
            **{key: value for key, value in metrics(truth, predicted).items()
               if key in ("accuracy", "macro_f1", "balanced_accuracy")},
            "classifier_fit_seconds": fit_s, "classifier_predict_seconds": predict_s,
            "warning_count": len(caught),
            "warnings_json": json.dumps([str(w.message) for w in caught], ensure_ascii=False)}


def candidate_grid(data, config):
    from .study import METHODS, GRID_KEYS
    rows = []
    for model in config["models"]:
        for method in METHODS:
            for k in ([data.X.shape[1]] if method == "full" else config["dimensions"]):
                for number, params in enumerate(ParameterGrid(config[GRID_KEYS[model]])):
                    rows.append({"candidate_id": f"{model}__{method}_{k}__p{number}",
                                 "model": model, "method": method, "k": k,
                                 "params_json": json.dumps(params, sort_keys=True)})
    return rows


def cached_grid_search(data, config, folds, out, workers, cache):
    from .study import dump_json, save_table
    out, cache = Path(out), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    preparation = []
    for fold_id, (train, valid) in enumerate(folds):
        with threadpool_limits(limits=1):
            record = prepare_fold(data, train, valid, max(config["dimensions"]), cache / f"fold_{fold_id}")
        preparation.append({"fold": fold_id, **record})
        print(f"[Prepare] fold={fold_id+1}/{len(folds)}: shared scale/PCA/ANOVA ready", flush=True)
    dump_json(out / "fold_transform_audit.json", preparation)
    specs = candidate_grid(data, config)
    total = len(specs) * len(folds)
    print(f"[Search] {len(specs)} configurations, {total} actual classifier fits; workers={workers}", flush=True)
    # 逐配置任务可并行，结果按候选编号和折号排序；调度顺序不影响选模平局处理。
    tasks = (joblib.delayed(score_fold)(spec, fold, cache / f"fold_{fold}", config)
             for spec in specs for fold in range(len(folds)))
    results = []
    t0 = time.perf_counter()
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        stream = joblib.Parallel(n_jobs=workers, pre_dispatch=workers * 2,
                                return_as="generator_unordered")(tasks)
        for result in stream:
            results.append(result)
            if len(results) % 50 == 0 or len(results) == total:
                save_table(pd.DataFrame(results).sort_values(["candidate_id", "fold"]),
                           out / "tables/cv_folds.csv")
                print(f"[Search] {len(results)}/{total} fits completed", flush=True)
    scores = pd.DataFrame(results).sort_values(["candidate_id", "fold"])
    if len(scores) != total or scores.duplicated(["candidate_id", "fold"]).any():
        raise AssertionError("网格记录缺失或重复。")
    summaries = []
    for spec in specs:
        part = scores[scores.candidate_id == spec["candidate_id"]]
        summaries.append({**spec, "cv_macro_f1": float(part.macro_f1.mean()),
                          "cv_macro_f1_std": float(part.macro_f1.std(ddof=0)),
                          "cv_accuracy": float(part.accuracy.mean()),
                          "mean_classifier_fit_seconds": float(part.classifier_fit_seconds.mean()),
                          "cv_warning_count": int(part.warning_count.sum())})
    candidates = save_table(summaries, out / "tables/cv_candidates.csv")
    save_table(scores, out / "tables/cv_folds.csv")
    dump_json(out / "search_engine.json", {
        "engine": "shared_fold_transform_exact_grid_v1", "n_candidates": len(specs),
        "actual_classifier_fits": total, "scaler_fits": len(folds), "pca_full_svd_fits": len(folds),
        "anova_score_fits": len(folds), "search_wall_seconds_excluding_transform": time.perf_counter()-t0,
        "warning_count": int(scores.warning_count.sum()),
        "timing_scope": "CV fit秒数仅是分类器拟合，不含共享预处理，不能冒充整条管线训练时间。",
        "complete_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    for (model, method), table in candidates.groupby(["model", "method"], sort=False):
        top = table.loc[table.cv_macro_f1.idxmax()]
        print(f"[CV] {model}/{method}: max candidate Macro-F1={top.cv_macro_f1:.4%}, k={top.k}", flush=True)
    return candidates
