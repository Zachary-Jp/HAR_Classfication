"""从Notebook移入的预测耗时分解：批量1/32/256/2947，逐阶段原始记录与复算。

这些测量只在配置锁定和最终测试后执行，不参与超参数或维数选择。
独立测量的阶段中位数不能严格相加得到完整predict的中位数。
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

STAGES = ("scale", "reduce", "classifier", "whole")


def _json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.17g")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_timing_config(config: dict) -> None:
    if set(config) != {"batch_sizes", "repeats", "warmups", "seed"}:
        raise ValueError("timing_config.json需要batch_sizes、repeats、warmups、seed四个字段。")
    sizes = config["batch_sizes"]
    if not isinstance(sizes, list) or not sizes or any(type(s) is not int or s < 1 for s in sizes):
        raise ValueError("计时批量必须为正整数列表。")
    if sizes != sorted(set(sizes)):
        raise ValueError("计时批量必须递增且不重复。")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("计时种子必须为非负整数。")
    for key in ("repeats", "warmups"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key}必须为正整数。")


def load_timing_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_timing_config(config)
    return config


def _passthrough(reducer) -> bool:
    return isinstance(reducer, str) and reducer == "passthrough"


def summarize_stages(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["selection_id", "batch_size", "stage"], as_index=False, sort=True).agg(
        median_ms=("total_ms", "median"),
        q25_ms=("total_ms", lambda s: s.quantile(.25)),
        q75_ms=("total_ms", lambda s: s.quantile(.75)),
        ms_per_window=("ms_per_window", "median"),
        n_repeats=("repeat", "nunique"),
    )


def benchmark_stages(run_dir: Path, X_test: np.ndarray, models: dict,
                     config: dict) -> pd.DataFrame:
    """只测锁定模型。变换输入在计时前准备；whole总是执行真正完整Pipeline。"""
    validate_timing_config(config)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    if X_test.ndim != 2 or not len(X_test) or not np.isfinite(X_test).all():
        raise ValueError("计时输入必须为非空、有限值二维数组。")
    if not models:
        raise ValueError("没有模型可计时。")
    if max(config["batch_sizes"]) > len(X_test):
        raise ValueError("计时批量不能超过测试样本数；不静默截断，以免错误标注。")
    run_dir = Path(run_dir)
    tables = run_dir / "tables"
    rows, index_rows, parity_rows = [], [], []
    rng = np.random.default_rng(config["seed"])
    started = datetime.now(timezone.utc).isoformat()
    pools = []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for size in config["batch_sizes"]:
            # 与Notebook一致，均匀取位置。B=1只有一个固定窗口，不代表整个流式用户分布。
            indices = np.linspace(0, len(X_test) - 1, size, dtype=int)
            raw_input = np.ascontiguousarray(X_test[indices])
            index_rows.extend({"batch_size": size, "position_in_batch": j, "test_row_id": int(i)}
                              for j, i in enumerate(indices))
            functions = {}
            passthrough_ids = set()
            for sid in sorted(models):
                pipe = models[sid]
                scale, reducer, classifier = (pipe.named_steps[key] for key in ("scale", "reduce", "model"))
                scaled = scale.transform(raw_input)
                no_reduction = _passthrough(reducer)
                reduced = scaled if no_reduction else reducer.transform(scaled)
                # 确认单独拆解不改变分类结果；只比较结果，准确率来自正式完整测试。
                pred_whole = pipe.predict(raw_input)
                pred_parts = classifier.predict(reduced)
                mismatches = int(np.count_nonzero(pred_whole != pred_parts))
                if mismatches:
                    raise AssertionError(f"{sid} batch={size}的分阶段输出与完整Pipeline不同。")
                parity_rows.append({"selection_id": sid, "batch_size": size,
                                    "n_predictions": size, "mismatches": mismatches})
                # 默认实参绑定当前对象，避免循环闭包把不同模型或不同批量混起来。
                funcs: dict[str, Callable] = {
                    "scale": lambda s=scale, x=raw_input: s.transform(x),
                    "classifier": lambda c=classifier, z=reduced: c.predict(z),
                    "whole": lambda p=pipe, x=raw_input: p.predict(x),
                }
                if no_reduction:
                    passthrough_ids.add(sid)
                else:
                    funcs["reduce"] = lambda r=reducer, z=scaled: r.transform(z)
                functions[sid] = funcs
                for fn in funcs.values():
                    for _ in range(config["warmups"]):
                        fn()
            # 对照Notebook的顺序测量，额外随机交错模型与阶段，减轻执行时间漂移。
            # 每个模型和每个阶段仍按同样口径执行15次，不与网格搜索并行。
            for rep in range(config["repeats"]):
                for sid in rng.permutation(sorted(functions)):
                    funcs = functions[sid]
                    for stage in rng.permutation(list(funcs)):
                        t0 = time.perf_counter_ns()
                        funcs[stage]()
                        elapsed_ns = time.perf_counter_ns() - t0
                        rows.append({"selection_id": str(sid), "batch_size": size,
                            "stage": str(stage), "repeat": rep,
                            "elapsed_ns": elapsed_ns, "total_ms": elapsed_ns / 1e6,
                            "ms_per_window": elapsed_ns / 1e6 / size,
                            "measured": True})
                    if sid in passthrough_ids:
                        # 零是没有该操作，而不是计时器测得的零延迟。
                        rows.append({"selection_id": str(sid), "batch_size": size,
                            "stage": "reduce", "repeat": rep, "elapsed_ns": 0,
                            "total_ms": 0., "ms_per_window": 0., "measured": False})
            print(f"[StageTiming] batch={size}; {len(models)} pipelines x {config['repeats']} repeats finished", flush=True)
    raw = pd.DataFrame(rows).sort_values(["selection_id", "batch_size", "stage", "repeat"])
    summary = summarize_stages(raw)
    _csv(tables / "stage_timing_raw.csv", raw)
    _csv(tables / "stage_timing_summary.csv", summary)
    _csv(tables / "stage_timing_inputs.csv", pd.DataFrame(index_rows))
    _csv(tables / "stage_prediction_parity.csv", pd.DataFrame(parity_rows))
    model_hashes = {sid: _sha(run_dir / "models" / f"{sid}.joblib") for sid in sorted(models)
                    if (run_dir / "models" / f"{sid}.joblib").is_file()}
    def cpu_name():
        p = Path("/proc/cpuinfo")
        if p.is_file():
            for line in p.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return platform.processor()
    environment = {
        "scope": "锁定配置后逐阶段计时；不参与模型/参数/维数选择",
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "thread_limit": 1, "threadpools_during_measurement": pools,
        "numpy": np.__version__, "sklearn": sklearn.__version__, "python": platform.python_version(),
        "processor": cpu_name(), "platform": platform.platform(),
        "input_selection": "对每种批量用np.linspace固定取测试行；B=1是test第0行，不是全部单窗口分布。",
        "ordering": "每批量内每轮随机交错模型和阶段；相同种子；全部串行测量。",
        "input_memory_layout": "float64 C-contiguous raw_input; scaled/reduced follow sklearn outputs",
        "passthrough_zero": "full的reduce=0为结构性缺省，measured=False，不是测量值。",
        "timer": "time.perf_counter_ns",
        "raw_sha256": _sha(tables / "stage_timing_raw.csv"),
        "source_model_hashes": model_hashes,
        "excluded": ["传感器采集", "原始信号特征提取", "模型加载", "文件IO", "手机功耗"],
        "caveat": "阶段输入预先准备；缓存、检查和调度不同，独立阶段中位数之和不等于whole；不能用差值当因果分解。",
        "reference": "来自用户Notebook的单元22、24和26；补充交错执行、原始纳秒、输入索引和一致性检查。",
    }
    _json(run_dir / "stage_timing_environment.json", environment)
    build_comparison(run_dir)
    return summary


def build_comparison(run_dir: Path) -> pd.DataFrame:
    """与Notebook单元26对应，按每种批量比较PCA与全特征，避免预置结论。"""
    run_dir = Path(run_dir)
    scores = pd.read_csv(run_dir / "tables/test_metrics.csv").set_index("selection_id")
    selected = pd.read_csv(run_dir / "tables/selected_configs.csv").set_index("selection_id")
    timing = pd.read_csv(run_dir / "tables/stage_timing_summary.csv")
    rows = []
    for model in dict.fromkeys(scores["model"]):
        base, pca = f"{model}__full", f"{model}__pca"
        a, b = scores.loc[base], scores.loc[pca]
        for size in sorted(timing.batch_size.unique()):
            def value(sid, stage):
                return float(timing.loc[(timing.selection_id == sid) &
                    (timing.batch_size == size) & (timing.stage == stage), "ms_per_window"].iloc[0])
            ta, tb = value(base, "whole"), value(pca, "whole")
            rows.append({"model": model, "batch_size": int(size), "pca_k": int(b.k),
                "full_macro_f1": float(a.macro_f1), "pca_macro_f1": float(b.macro_f1),
                "pca_delta_macro_f1_pp": float((b.macro_f1 - a.macro_f1) * 100),
                "meets_cv_tolerance": bool(selected.loc[pca, "meets_cv_tolerance"]),
                "full_whole_ms_per_window": ta, "pca_whole_ms_per_window": tb,
                "pca_speedup_full_div_pca": ta / tb,
                "pca_projection_ms_per_window": value(pca, "reduce"),
                "full_classifier_ms_per_window": value(base, "classifier"),
                "pca_classifier_ms_per_window": value(pca, "classifier")})
    result = pd.DataFrame(rows)
    _csv(run_dir / "tables/PCA_vs_full_with_timing_breakdown.csv", result)
    return result


def verify_stage_timing(run_dir: Path, *, strict: bool = True) -> dict:
    run_dir = Path(run_dir)
    table_dir = run_dir / "tables"
    env = json.loads((run_dir / "stage_timing_environment.json").read_text(encoding="utf-8"))
    config = env["config"]
    validate_timing_config(config)
    frozen = run_dir / "timing_config_frozen.json"
    if frozen.is_file() and json.loads(frozen.read_text(encoding="utf-8")) != config:
        raise AssertionError("计时设置与本轮冻结设置不同。")
    raw = pd.read_csv(table_dir / "stage_timing_raw.csv")
    summary = pd.read_csv(table_dir / "stage_timing_summary.csv")
    selected = pd.read_csv(table_dir / "selected_configs.csv")
    expected = {(sid, b, stage, rep) for sid in selected.selection_id for b in config["batch_sizes"]
                for stage in STAGES for rep in range(config["repeats"])}
    observed = set(raw[["selection_id", "batch_size", "stage", "repeat"]].itertuples(index=False, name=None))
    if observed != expected or len(raw) != len(expected):
        raise AssertionError("分阶段计时记录缺失、重复或有意外配置。")
    if not np.isfinite(raw[["total_ms", "ms_per_window", "elapsed_ns"]].to_numpy()).all():
        raise AssertionError("存在非有限时间。")
    if (raw[["total_ms", "ms_per_window", "elapsed_ns"]] < 0).any().any():
        raise AssertionError("存在负耗时。")
    np.testing.assert_allclose(raw.total_ms, raw.elapsed_ns / 1e6, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(raw.ms_per_window, raw.total_ms / raw.batch_size, rtol=1e-12, atol=1e-12)
    zero = raw.selection_id.str.endswith("__full") & (raw.stage == "reduce")
    if raw.loc[zero, "measured"].any() or raw.loc[zero, "total_ms"].ne(0).any():
        raise AssertionError("全特征的压缩阶段应标为不存在，而不是伪测量。")
    if not raw.loc[~zero, "measured"].all() or raw.loc[~zero, "elapsed_ns"].le(0).any():
        raise AssertionError("实际执行阶段没有有效正耗时。")
    recalculated = summarize_stages(raw)
    keys = ["selection_id", "batch_size", "stage"]
    merged = summary.merge(recalculated, on=keys, suffixes=("_saved", "_calc"), validate="one_to_one")
    if len(merged) != len(summary) or len(merged) != len(recalculated):
        raise AssertionError("汇总分组不完整。")
    for key in ["median_ms", "q25_ms", "q75_ms", "ms_per_window", "n_repeats"]:
        np.testing.assert_allclose(merged[key + "_saved"], merged[key + "_calc"], rtol=1e-10, atol=1e-12)
    parity = pd.read_csv(table_dir / "stage_prediction_parity.csv")
    parity_expected = {(sid, b) for sid in selected.selection_id for b in config["batch_sizes"]}
    if (set(parity[["selection_id", "batch_size"]].itertuples(index=False, name=None)) != parity_expected
        or len(parity) != len(parity_expected) or parity.mismatches.ne(0).any()):
        raise AssertionError("分阶段预测一致性不通过。")
    if not np.array_equal(parity.n_predictions, parity.batch_size):
        raise AssertionError("预测一致性检查的样本数错误。")
    if _sha(table_dir / "stage_timing_raw.csv") != env["raw_sha256"]:
        raise AssertionError("原始计时文件哈希变化。")
    for sid, digest in env["source_model_hashes"].items():
        if _sha(run_dir / "models" / f"{sid}.joblib") != digest:
            raise AssertionError("用于计时的模型文件变化。")
    comparison = pd.read_csv(table_dir / "PCA_vs_full_with_timing_breakdown.csv")
    for row in comparison.itertuples():
        subset = summary[(summary.batch_size == row.batch_size) & (summary.stage == "whole")].set_index("selection_id")
        ratio = subset.loc[row.model + "__full", "ms_per_window"] / subset.loc[row.model + "__pca", "ms_per_window"]
        if not np.isclose(ratio, row.pca_speedup_full_div_pca, rtol=1e-10, atol=1e-12):
            raise AssertionError("PCA速度比无法从计时表复算。")
    result = {"status": "PASS", "n_models": len(selected), "n_raw_rows": len(raw),
              "n_summary_rows": len(summary), "n_parity_predictions": int(parity.n_predictions.sum()),
              "batch_sizes": config["batch_sizes"], "repeats": config["repeats"],
              "scope": "计时记录完整性、统计复算、分阶段预测一致性和模型哈希；不代表跨设备性能保证。"}
    _json(run_dir / "stage_timing_verification.json", result)
    return result


def draw_stage_figures(run_dir: Path, font_path=None) -> list[Path]:
    """Notebook新增图移入项目；五模型各独立出图，不使用多子图挤压文字。"""
    from .plots import configure_font
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    run_dir = Path(run_dir)
    verify_stage_timing(run_dir)
    configure_font(font_path)
    summary = pd.read_csv(run_dir / "tables/stage_timing_summary.csv")
    selected = pd.read_csv(run_dir / "tables/selected_configs.csv")
    folder = run_dir / "figures/stage_timing"
    folder.mkdir(parents=True, exist_ok=True)
    names = {"scale": "标准化", "reduce": "压缩变换", "classifier": "分类预测", "whole": "完整预测"}
    method_names = {"full": "全特征", "pca": "PCA", "anova": "ANOVA"}
    paths = []
    for model in dict.fromkeys(selected.model):
        size = int(summary.batch_size.max())
        sub = summary[(summary.batch_size == size) & summary.selection_id.str.startswith(model + "__")]
        fig, ax = plt.subplots(figsize=(10.5, 5.9))
        fig.subplots_adjust(left=.13, right=.97, top=.88, bottom=.22)
        xx, width = np.arange(3), .18
        for j, stage in enumerate(STAGES):
            vals = [float(sub[(sub.selection_id == model + "__" + method) & (sub.stage == stage)].ms_per_window.iloc[0]) * 1000
                    for method in ("full", "pca", "anova")]
            ax.bar(xx + (j - 1.5) * width, vals, width, label=names[stage])
        ax.set_xticks(xx, [method_names[m] for m in ("full", "pca", "anova")])
        ax.set_xlabel("特征方案")
        ax.set_ylabel("平均每窗口耗时（微秒）")
        ax.set_title(f"{model}：预测耗时分解（批量 {size}）", pad=12)
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
        ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16))
        ax.spines["left"].set_visible(True); ax.spines["bottom"].set_visible(True)
        ax.tick_params(length=4, width=1)
        path = folder / f"{model}_stage_timing.png"
        fig.savefig(path, dpi=300); plt.close(fig); paths.append(path)

        fig, ax = plt.subplots(figsize=(9.5, 5.9))
        fig.subplots_adjust(left=.14, right=.96, top=.88, bottom=.22)
        for method, marker in (("full", "o"), ("pca", "s"), ("anova", "^")):
            vals = summary[(summary.selection_id == model + "__" + method) & (summary.stage == "whole")].sort_values("batch_size")
            ax.plot(vals.batch_size, vals.ms_per_window * 1000, marker=marker, label=method_names[method])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(sorted(summary.batch_size.unique()))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("一次预测的窗口数")
        ax.set_ylabel("平均每窗口耗时（微秒，对数刻度）")
        ax.set_title(f"{model}：预测批量与耗时", pad=12)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.16))
        ax.grid(alpha=.17); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(True); ax.spines["bottom"].set_visible(True)
        ax.tick_params(length=4, width=1)
        path = folder / f"{model}_batch_size_comparison.png"
        fig.savefig(path, dpi=300); plt.close(fig); paths.append(path)
    _json(run_dir / "stage_figure_manifest.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _sha(run_dir / "tables/stage_timing_summary.csv"),
        "files": [str(p.relative_to(run_dir)) for p in paths],
        "note": "独立阶段中位数不可严格求和；full无reduce；B=1仅固定测试窗口。说明放记录而非图下。"})
    return paths
