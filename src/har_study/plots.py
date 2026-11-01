"""结果图：中文标题与坐标，英文模型名，白到深蓝的矩阵；不放图下说明。"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

DPI = 300
ACTIVITY_CN = ["行走", "上楼", "下楼", "坐", "站立", "躺卧"]
MODEL_ORDER = ("RBF_SVM", "RandomForest", "LogisticRegression", "KNN", "FisherLDA")
METHODS = ("full", "pca", "anova")
METHOD_CN = {"full": "全特征", "pca": "PCA", "anova": "ANOVA"}
CM_CMAP = "Blues"

def configure_font(font_path: str | Path | None = None):
    """使用本机中文字体；不附带字体文件，也不静默输出缺字方框。"""
    required = {ord(c) for c in '跨人员分类压缩准确率维数测试'}
    if font_path:
        path = Path(font_path).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(f'找不到字体：{path}')
        if not required.issubset(FT2Font(str(path)).get_charmap()):
            raise ValueError('指定字体缺少需要的中文字形。')
        font_manager.fontManager.addfont(str(path))
        chosen = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        chosen = None
        for name in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC',
                     'Source Han Sans SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei'):
            try:
                path = font_manager.findfont(name, fallback_to_default=False)
                if required.issubset(FT2Font(path).get_charmap()):
                    chosen = name
                    break
            except (ValueError, OSError, RuntimeError):
                continue
        if chosen is None:
            raise RuntimeError('未找到中文字体。安装中文字体，或使用 --font 指定本机字体文件。')
    # 不设置自定义颜色主题：保持默认颜色循环，依靠布局、形状和线型区分系列。
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [chosen, 'DejaVu Sans'],
        'axes.unicode_minus': False, 'font.size': 11, 'axes.labelsize': 11.5,
        'xtick.labelsize': 10.5, 'ytick.labelsize': 11,
        'axes.linewidth': 1.1, 'axes.spines.left': True, 'axes.spines.bottom': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 110, 'savefig.dpi': DPI, 'savefig.bbox': None,
    })
    return chosen



def check_layout(fig) -> list[dict]:
    """保存前检查文字是否超出画布。仍需人工检查图中标注之间是否相互遮挡。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    items = list(fig.texts)
    for ax in fig.axes:
        items.extend([ax.xaxis.label, ax.yaxis.label, ax.title])
        items.extend(ax.texts)
        for label in ax.get_xticklabels():
            x, _ = label.get_position()
            lo, hi = sorted(ax.get_xlim())
            if lo - 1e-9 <= x <= hi + 1e-9: items.append(label)
        for label in ax.get_yticklabels():
            _, y = label.get_position()
            lo, hi = sorted(ax.get_ylim())
            if lo - 1e-9 <= y <= hi + 1e-9: items.append(label)
        if ax.get_legend() is not None: items.append(ax.get_legend())
    items.extend(fig.legends)
    errors = []
    for item in items:
        if not item.get_visible(): continue
        if hasattr(item, 'get_text') and not item.get_text(): continue
        box = item.get_window_extent(renderer=renderer)
        if box.x0 < bounds.x0 - 2 or box.y0 < bounds.y0 - 2 or box.x1 > bounds.x1 + 2 or box.y1 > bounds.y1 + 2:
            errors.append({'text': str(item.get_text()) if hasattr(item, 'get_text') else 'legend',
                           'bbox': [float(v) for v in box.extents]})
    return errors



def canvas(title, size=(10.8, 6.4), *, left=.14, right=.94, bottom=.14, top=.87):
    """只保留标题、坐标、刻度和必要图例，不生成左下角脚注。"""
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig.suptitle(title, x=(left + right) / 2, y=.96, fontsize=16, fontweight="bold", va="top")
    ax.tick_params(axis="both", direction="out", length=4, width=1.1, pad=7)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(1.2)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, path):
    """固定比例输出；画布外文字检查失败时不生成残缺图片。"""
    problems = check_layout(fig)
    if problems:
        plt.close(fig)
        raise ValueError(f"{Path(path).name}有超出画布的标注：{problems}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def _label(row):
    return f"{row.model} / {METHOD_CN[row.method]} / {int(row.k)}维"


def _sorted(frame, models):
    copy = frame.copy()
    copy["_order"] = [models.index(m) * 3 + METHODS.index(t) for m, t in zip(copy.model, copy.method)]
    return copy.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def confusion_figure(counts, title, *, vmax=None):
    """色块和格内数字都表示样本数；各方案使用同一个颜色上限。"""
    counts = np.asarray(counts)
    if counts.shape != (6, 6) or (counts < 0).any():
        raise ValueError("混淆矩阵应为6×6非负计数。")
    vmax = max(1, float(counts.max())) if vmax is None else max(1, float(vmax))
    fig, ax = canvas(title, size=(8.5, 7.5), left=.14, right=.90, bottom=.14, top=.87)
    image = ax.imshow(counts, cmap=CM_CMAP, vmin=0, vmax=vmax,
                      interpolation="nearest", aspect="equal")
    for i in range(6):
        for j in range(6):
            # 深色格配白字，浅色格配黑字；不再用透明度把色带冲淡。
            ax.text(j, i, str(int(counts[i, j])), ha="center", va="center",
                    fontsize=12, color="white" if counts[i, j] > .57 * vmax else "black")
    ax.set_xticks(np.arange(6), ACTIVITY_CN)
    ax.set_yticks(np.arange(6), ACTIVITY_CN)
    ax.set_xlabel("预测活动", labelpad=12)
    ax.set_ylabel("真实活动", labelpad=12)
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, fraction=.045, pad=.045)
    bar.set_label("样本数", labelpad=12)
    return fig



def confusion_comparison_figure(matrices, model, dimensions, *, vmax=None):
    """同一分类器的全特征/PCA/ANOVA按固定次序横排，共享计数色标。"""
    if set(matrices) != set(METHODS) or set(dimensions) != set(METHODS):
        raise ValueError("每个模型必须同时包含full、pca、anova三种特征方案。")
    values = {method: np.asarray(matrices[method]) for method in METHODS}
    for method, counts in values.items():
        if counts.shape != (6, 6) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(f"{method}混淆矩阵必须是6×6非负计数。")
        if not np.all(counts == np.floor(counts)):
            raise ValueError(f"{method}混淆矩阵必须是整数计数。")
    if vmax is None:
        vmax = max(float(counts.max()) for counts in values.values())
    vmax = max(1, float(vmax))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.3), sharey=True)
    fig.subplots_adjust(left=.076, right=.886, bottom=.19, top=.84, wspace=.18)
    fig.suptitle(f"{model}：三种特征处理的混淆矩阵", y=.96,
                 fontsize=16, fontweight="bold")
    for col, method in enumerate(METHODS):
        ax = axes[col]
        counts = values[method]
        image = ax.imshow(counts, cmap=CM_CMAP, vmin=0, vmax=vmax,
                          interpolation="nearest", aspect="equal")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(int(counts[i, j])), ha="center", va="center",
                        fontsize=10.5, color="white" if counts[i, j] > .57 * vmax else "black")
        ax.set_xticks(np.arange(6), ACTIVITY_CN)
        ax.set_yticks(np.arange(6), ACTIVITY_CN)
        if col:
            ax.tick_params(axis="y", labelleft=False, left=False)
        ax.set_title(f"{METHOD_CN[method]}（{int(dimensions[method])}维）",
                     fontsize=13, pad=14)
        ax.tick_params(axis="x", length=0, pad=9)
        ax.grid(False)
    axes[0].set_ylabel("真实活动", labelpad=12)
    fig.supxlabel("预测活动", y=.075, fontsize=12)
    # 三图共用一条色标，且所有五个模型都使用相同的vmax。
    cb_ax = fig.add_axes([.907, .235, .016, .53])
    bar = fig.colorbar(image, cax=cb_ax)
    bar.set_label("样本数", labelpad=10)
    return fig


def heatmap_cells(table, model, config):
    """只绘制实际评估的参数；缺失组合保留NaN，不补0，不插值。"""
    if model == "RBF_SVM":
        rows, cols = config["svm_grid"]["model__C"], config["svm_grid"]["model__gamma"]
        key = lambda p: (p["model__C"], p["model__gamma"])
        ynames, xnames = [f"{v:g}" for v in rows], [str(v) for v in cols]
        ylabel, xlabel = "C", "gamma"
    elif model == "RandomForest":
        rows = config["rf_grid"]["model__max_depth"]
        cols = config["rf_grid"]["model__min_samples_leaf"]
        key = lambda p: (p["model__max_depth"], p["model__min_samples_leaf"])
        ynames, xnames = ["不限" if v is None else str(v) for v in rows], [str(v) for v in cols]
        ylabel, xlabel = "最大深度", "叶节点最少样本数"
    elif model == "LogisticRegression":
        rows, cols = ["L2"], config["lr_grid"]["model__C"]
        key = lambda p: ("L2", p["model__C"])
        ynames, xnames, ylabel, xlabel = rows, [f"{v:g}" for v in cols], "正则化", "C"
    elif model == "KNN":
        rows, cols = config["knn_grid"]["model__weights"], config["knn_grid"]["model__n_neighbors"]
        key = lambda p: (p["model__weights"], p["model__n_neighbors"])
        ynames = [{"uniform": "等权", "distance": "距离加权"}[v] for v in rows]
        xnames, ylabel, xlabel = [str(v) for v in cols], "投票方式", "邻居数"
    elif model == "FisherLDA":
        rows, cols = ["lsqr"], config["lda_grid"]["model__shrinkage"]
        key = lambda p: ("lsqr", p["model__shrinkage"])
        ynames = rows
        xnames = ["不收缩" if v is None else "自动收缩" if v == "auto" else str(v) for v in cols]
        ylabel, xlabel = "求解器", "协方差收缩"
    else:
        raise ValueError(model)
    values = np.full((len(rows), len(cols)), np.nan)
    for record in table.itertuples():
        r, c = key(json.loads(record.params_json))
        i, j = rows.index(r), cols.index(c)
        if np.isfinite(values[i, j]):
            raise ValueError("热力图出现重复参数组合。")
        values[i, j] = record.cv_macro_f1 * 100
    return values, ynames, xnames, ylabel, xlabel


def parameter_figure(values, ynames, xnames, ylabel, xlabel, title):
    height = max(3.7, 3.0 + .52 * len(ynames))
    width = max(8.7, 3.5 + 1.0 * len(xnames))
    fig, ax = canvas(title, size=(width, height), left=.16, right=.89, bottom=.23, top=.77)
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        raise ValueError("参数图没有实际验证分数。")
    lo = max(0, np.floor(finite.min() * 10) / 10 - .1)
    hi = min(100, max(lo + .3, np.ceil(finite.max() * 10) / 10 + .1))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="Blues", vmin=lo, vmax=hi,
                      interpolation="nearest", aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax.text(j, i, f"{val:.2f}" if np.isfinite(val) else "—", ha="center", va="center",
                    fontsize=11, color="white" if np.isfinite(val) and val > lo + .57 * (hi - lo) else "black")
    ax.set_xticks(np.arange(len(xnames)), xnames)
    ax.set_yticks(np.arange(len(ynames)), ynames)
    ax.set_xlabel(xlabel, labelpad=10); ax.set_ylabel(ylabel, labelpad=10)
    bar = fig.colorbar(image, ax=ax, fraction=.035, pad=.035)
    bar.set_label("验证 Macro-F1（%）", labelpad=10)
    return fig


def draw_all(run_dir: Path, font_path=None) -> list[Path]:
    """读取已保存结果；不改模型和实验数值，图片脚注全部移至项目说明。"""
    from .verify import verify_run
    run = Path(run_dir)
    verify_run(run)
    font = configure_font(font_path)
    read = lambda name: pd.read_csv(run / "tables" / name)
    config = json.loads((run / "config_frozen.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
    selected, test, curve = read("selected_configs.csv"), read("test_metrics.csv"), read("compression_curve.csv")
    candidates, predictions = read("cv_candidates.csv"), read("test_predictions.csv")
    subject, protocol = read("test_subjects.csv"), read("protocol_comparison.csv")
    timing, sizes = read("timing_summary.csv"), read("final_fit_and_size.csv")
    models = [m for m in MODEL_ORDER if m in set(test.model)]
    if set(models) != set(test.model): raise ValueError("出现未定义模型。")
    selected, test = _sorted(selected, models), _sorted(test, models)
    ids = selected.selection_id.tolist()
    dest = run / "figures"
    run.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=".plot_build_", dir=run)
    stage = Path(temporary.name)
    records = []
    def finish(fig, name):
        save(fig, stage / name)
        records.append({"file": name, "text_outside_canvas": 0,
                        "sha256": hashlib.sha256((stage / name).read_bytes()).hexdigest()})
    try:
        fig, ax = canvas("活动样本分布", top=.82)
        x = np.arange(6)
        for i, (part, label) in enumerate((("train", "训练集"), ("test", "测试集"))):
            ax.bar(x + (i - .5) * .34, [audit[part]["class_counts"][str(k)] for k in range(1, 7)],
                   width=.31, label=label)
        ax.set_xticks(x, ACTIVITY_CN); ax.set_xlabel("活动类别"); ax.set_ylabel("窗口样本数")
        ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01))
        ax.grid(axis="y", alpha=.18)
        finish(fig, "overview/01_activity_distribution.png")

        fig, ax = canvas("验证协议比较", size=(11.5, 6.0), left=.25, top=.80)
        y = np.arange(len(models))
        for i, (key, label, mark) in enumerate((("subject_grouped", "人员分组", "o"),
                                               ("random_windows", "随机窗口", "s"))):
            group = protocol[protocol.protocol == key].groupby("model").macro_f1.agg(["mean", "std"]).reindex(models)
            ax.errorbar(group["mean"] * 100, y + (i - .5) * .24, xerr=group["std"] * 100,
                        fmt=mark, capsize=3, label=label)
        ax.set_yticks(y, models); ax.invert_yaxis(); ax.set_xlim(60, 102)
        ax.set_xlabel("固定参数：交叉验证 Macro-F1（%）"); ax.set_ylabel("分类模型")
        ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
        ax.grid(axis="x", alpha=.18)
        finish(fig, "overview/02_protocol_comparison.png")

        for model in models:
            sub = curve[curve.model == model]
            baseline = float(sub[sub.method == "full"].cv_macro_f1.iloc[0]) * 100
            fig, ax = canvas(f"{model}：维数与验证表现", size=(10.2, 6.4), top=.76)
            lower, upper = [], []
            for method, marker in (("pca", "o"), ("anova", "s")):
                rows = sub[sub.method == method].sort_values("k")
                ax.errorbar(rows.k, rows.cv_macro_f1 * 100, yerr=rows.cv_macro_f1_std * 100,
                            marker=marker, capsize=3, linewidth=1.6, label=METHOD_CN[method])
                lower.extend((rows.cv_macro_f1 - rows.cv_macro_f1_std) * 100)
                upper.extend((rows.cv_macro_f1 + rows.cv_macro_f1_std) * 100)
            ax.axhline(baseline, linestyle="--", linewidth=1.1, label="全特征基准")
            ax.axhline(baseline - config["allowed_macro_f1_drop"] * 100, linestyle=":", linewidth=1.1,
                       label="验证容忍下限")
            ax.set_xticks(config["dimensions"])
            ax.set_ylim(max(0, np.floor(min(lower + [baseline]) - 1)), min(101, np.ceil(max(upper + [baseline]) + 1)))
            ax.set_xlabel("维数"); ax.set_ylabel("验证 Macro-F1（%）")
            ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.015), fontsize=10)
            ax.grid(alpha=.18)
            finish(fig, f"compression/{model}.png")

        fig, ax = canvas("测试集分类表现", size=(12.4, max(7, len(test) * .48 + 2.4)), left=.34, right=.86, top=.84, bottom=.10)
        y = np.arange(len(test))
        for i, (key, name, marker) in enumerate((("accuracy", "准确率", "o"), ("macro_f1", "Macro-F1", "s"))):
            ax.scatter(test[key] * 100, y + (i - .5) * .24, marker=marker, label=name, s=36)
        ax.set_yticks(y, [_label(row) for row in test.itertuples()]); ax.invert_yaxis()
        ax.set_xlim(max(0, np.floor(test[["accuracy", "macro_f1"]].min().min() * 100 - 2)), 100)
        ax.set_xlabel("测试得分（%）"); ax.set_ylabel("模型 / 特征方案")
        ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.015))
        for pos, row in zip(y, test.itertuples()):
            ax.text(1.025, pos, f"{row.accuracy:.2%} / {row.macro_f1:.2%}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.grid(axis="x", alpha=.18)
        finish(fig, "overview/03_test_metrics.png")

        people = sorted(subject.subject.unique())
        matrix = subject.pivot(index="selection_id", columns="subject", values="macro_f1").reindex(index=ids, columns=people) * 100
        fig, ax = canvas("测试人员识别表现", size=(12, max(8, .41 * len(ids) + 2.8)), left=.32, right=.90, bottom=.12)
        lo = max(0, np.floor(matrix.to_numpy().min() / 10) * 10)
        image = ax.imshow(matrix, vmin=lo, vmax=100, cmap="Blues", interpolation="nearest", aspect="auto")
        for i in range(len(ids)):
            for j in range(len(people)):
                val = matrix.iloc[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9.5,
                        color="white" if val > lo + .57 * (100 - lo) else "black")
        ax.set_yticks(np.arange(len(ids)), [_label(row) for row in selected.itertuples()])
        ax.set_xticks(np.arange(len(people)), [str(p) for p in people])
        ax.set_xlabel("测试人员编号"); ax.set_ylabel("模型 / 特征方案")
        fig.colorbar(image, ax=ax, fraction=.025, pad=.03).set_label("Macro-F1（%）", labelpad=10)
        finish(fig, "overview/04_subject_heatmap.png")

        for kind, title in (("batch", "批量预测耗时"), ("single", "单窗口预测耗时")):
            rows = timing[timing.kind == kind].set_index("selection_id").reindex(ids)
            val = rows.median_ms_per_sample.to_numpy()
            err = np.vstack([val - rows.q25_ms_per_sample, rows.q75_ms_per_sample - val])
            fig, ax = canvas(title, size=(11.8, max(7, .43 * len(ids) + 2)), left=.35, right=.86, bottom=.10)
            ax.errorbar(val, np.arange(len(ids)), xerr=err, fmt="o", capsize=3)
            ax.set_yticks(np.arange(len(ids)), [_label(row) for row in selected.itertuples()]); ax.invert_yaxis()
            ax.set_xscale("log"); ax.set_xlim(max(1e-8, float(rows.q25_ms_per_sample.min()) / 1.6), float(rows.q75_ms_per_sample.max()) * 1.8)
            for pos, value in enumerate(val):
                ax.text(1.035, pos, f"{value:.4f}", transform=ax.get_yaxis_transform(), va="center", fontsize=10)
            ax.set_xlabel("毫秒 / 窗口（对数轴；中位数及四分位区间）"); ax.set_ylabel("模型 / 特征方案")
            ax.grid(axis="x", alpha=.18)
            finish(fig, f"overview/05_timing_{kind}.png")
        vals = sizes.set_index("selection_id").reindex(ids).pipeline_bytes / (1024 * 1024)
        fig, ax = canvas("完整模型文件大小", size=(11.8, max(7, .43 * len(ids) + 2)), left=.35, right=.86, bottom=.10)
        ax.scatter(vals, np.arange(len(ids)), s=45)
        ax.set_yticks(np.arange(len(ids)), [_label(row) for row in selected.itertuples()]); ax.invert_yaxis()
        ax.set_xscale("log"); ax.set_xlim(vals.min() / 1.6, vals.max() * 1.6)
        for pos, value in enumerate(vals): ax.text(1.035, pos, f"{value:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=10)
        ax.set_xlabel("文件大小（MiB，对数轴）"); ax.set_ylabel("模型 / 特征方案"); ax.grid(axis="x", alpha=.18)
        finish(fig, "overview/06_model_size.png")

        matrices = {}
        for sid in ids:
            sub = predictions[predictions.selection_id == sid]
            matrices[sid] = confusion_matrix(sub.actual, sub.predicted, labels=list(range(1, 7)))
        vmax = max(float(cm.max()) for cm in matrices.values())
        for model in models:
            grouped = selected[selected.model == model].set_index("method")
            if set(grouped.index) != set(METHODS) or grouped.index.duplicated().any():
                raise ValueError(f"{model}缺少全特征/PCA/ANOVA中的一个，或存在重复方案。")
            model_matrices = {method: matrices[grouped.loc[method, "selection_id"]] for method in METHODS}
            dims = {method: int(grouped.loc[method, "k"]) for method in METHODS}
            fig = confusion_comparison_figure(model_matrices, model, dims, vmax=vmax)
            finish(fig, f"confusion/confusion_{model}.png")
        for item in selected.itertuples():
            sub = candidates[(candidates.model == item.model) & (candidates.method == item.method) & (candidates.k == item.k)]
            values, ys, xs, yl, xl = heatmap_cells(sub, item.model, config)
            fig = parameter_figure(values, ys, xs, yl, xl,
                        f"{item.model} / {METHOD_CN[item.method]} / {int(item.k)}维")
            finish(fig, f"parameters/parameters_{item.selection_id}.png")

        # 全部图绘制与边界检查通过后再替换旧图，避免失败时留下半套结果。
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(stage, dest)
        info = {"operation": "只重绘，不改训练结果", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "font": font, "dpi": DPI, "confusion_colormap": CM_CMAP,
                "confusion_color_quantity": "原始计数，所有方案共用相同颜色上限",
                "footnotes_in_images": False, "confusion_layout": "每模型一张，左至右full/pca/anova；共享色标",
                "n_figures": len(records), "figures": records,
                "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (run / "visualization_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return [dest / record["file"] for record in records]
    finally:
        temporary.cleanup()
