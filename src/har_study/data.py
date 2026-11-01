from __future__ import annotations
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile

import numpy as np

DATASET_URLS = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
)
REQUIRED_FILES = (
    "features.txt", "activity_labels.txt",
    "train/X_train.txt", "train/y_train.txt", "train/subject_train.txt",
    "test/X_test.txt", "test/y_test.txt", "test/subject_test.txt",
)
OPTIONAL_FILES = ("README.txt", "features_info.txt")
ACTIVITIES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    feature_names: list[str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_dataset_root(directory: Path) -> Path:
    directory = Path(directory)
    roots = [directory] + sorted({p.parent for p in directory.rglob("features.txt")})
    for root in roots:
        if all((root / name).is_file() for name in REQUIRED_FILES):
            return root
    raise FileNotFoundError(
        f"{directory} 中未找到完整官方数据；必须同时有 features.txt、train/ 和 test/，"
        "且保留 subject_train.txt 和 subject_test.txt。"
    )


def extract_official_archive(archive: Path, destination: Path) -> Path:
    """支持官方外层 ZIP 和内部 UCI HAR Dataset.zip; 只复制所需文件。"""
    destination = Path(destination)
    wanted = set(REQUIRED_FILES + OPTIONAL_FILES)
    captured: dict[str, bytes] = {}

    def visit(z: zipfile.ZipFile, depth: int = 0) -> None:
        if depth > 2:
            raise ValueError("压缩包嵌套层数异常。")
        for item in z.infolist():
            name = item.filename.replace("\\", "/")
            if "__MACOSX" in name or item.is_dir():
                continue
            if item.file_size > 1024 * 1024 * 1024:
                raise ValueError("压缩包中的单文件过大，停止解压。")
            matched = [key for key in wanted if name == key or name.endswith("/" + key)]
            for key in matched:
                # 不直接 extractall，防止包中路径跳出目标目录。
                content = z.read(item)
                if key in captured and captured[key] != content:
                    raise ValueError(f"压缩包含多个不同版本的 {key}。")
                captured[key] = content
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(z.read(item))) as nested:
                    visit(nested, depth + 1)

    with zipfile.ZipFile(archive) as package:
        visit(package)
    missing = sorted(set(REQUIRED_FILES) - captured.keys())
    if missing:
        raise ValueError(f"压缩包缺少文件：{missing}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in captured.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return destination


def prepare_dataset(source: str | Path | None, data_dir: Path) -> Path:
    """有本地数据就离线读取；否则尝试官方下载。下载失败不替换成模拟数据。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if source is not None:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            return find_dataset_root(path)
        if path.is_file() and zipfile.is_zipfile(path):
            return extract_official_archive(path, data_dir / "UCI HAR Dataset")
        raise FileNotFoundError(f"--data 需要官方 ZIP 或完整数据目录：{path}")
    try:
        return find_dataset_root(data_dir)
    except FileNotFoundError:
        pass
    errors = []
    for url in DATASET_URLS:
        try:
            print(f"[Data] Downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "HAR-Course/1.0"})
            with tempfile.TemporaryDirectory(prefix="har_download_") as td:
                target = Path(td) / "dataset.zip"
                with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                if not zipfile.is_zipfile(target):
                    raise ValueError("下载内容不是有效 ZIP。")
                root = extract_official_archive(target, data_dir / "UCI HAR Dataset")
                (data_dir / "download_provenance.json").write_text(json.dumps({
                    "source_url": url, "archive_sha256": sha256(target),
                    "dataset": "UCI HAR, DOI 10.24432/C54S4K",
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                return root
        except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "官方下载失败，尚未开始训练。请把官方 UCI HAR Dataset.zip 放到本机后运行：\n"
        'python run_experiment.py --data "UCI HAR Dataset.zip"\n' + "\n".join(errors)
    )


def file_hashes(root: Path) -> dict[str, str]:
    return {name: sha256(root / name) for name in REQUIRED_FILES}


def load_split(root: Path, name: str, *, strict: bool = True) -> Split:
    if name not in {"train", "test"}:
        raise ValueError("name 只能为 train 或 test。")
    root = Path(root)
    feature_rows = [line.strip().split(maxsplit=1) for line in
                    (root / "features.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(row) != 2 for row in feature_rows):
        raise ValueError("features.txt 格式错误。")
    ids = [int(row[0]) for row in feature_rows]
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError("特征编号不连续，不能确认特征顺序。")
    # 官方存在重名的特征，保留全部列；用编号作为稳定、唯一的列身份。
    names = [row[1] for row in feature_rows]
    X = np.loadtxt(root / name / f"X_{name}.txt", dtype=np.float64, ndmin=2)
    y = np.loadtxt(root / name / f"y_{name}.txt", dtype=np.int64, ndmin=1)
    people = np.loadtxt(root / name / f"subject_{name}.txt", dtype=np.int64, ndmin=1)
    if X.shape != (len(y), len(names)) or len(people) != len(y):
        raise ValueError("特征、类别和人员文件的行数或列数不一致。")
    if not np.isfinite(X).all():
        raise ValueError("数据出现缺失值或无穷值，不能按预设协议直接运行。")
    labels = {int(line.split()[0]): line.split()[1] for line in
              (root / "activity_labels.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    if labels != ACTIVITIES or set(np.unique(y)) != set(ACTIVITIES):
        raise ValueError("不是预期的六类 UCI HAR 数据。")
    if (people < 1).any() or (people > 30).any():
        raise ValueError("人员编号超出官方的 1～30 范围。")
    if strict:
        shape, n_people = ((7352, 561), 21) if name == "train" else ((2947, 561), 9)
        if X.shape != shape or len(np.unique(people)) != n_people:
            raise ValueError(f"{name} 数据规模与官方版本不符：实际 {X.shape}，预期 {shape}。")
    return Split(np.ascontiguousarray(X), y, people, names)


def audit_splits(train: Split, test: Split) -> dict:
    if set(train.subjects) & set(test.subjects):
        raise ValueError("训练和测试人员存在交叉，不能作为跨人员实验。")
    if train.feature_names != test.feature_names:
        raise ValueError("训练与测试的特征顺序不一致。")
    def summary(s: Split) -> dict:
        row_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in s.X]
        return {"samples": len(s.y), "features": s.X.shape[1],
                "subject_ids": sorted(map(int, np.unique(s.subjects))),
                "class_counts": {str(k): int((s.y == k).sum()) for k in ACTIVITIES},
                "duplicate_feature_rows": len(row_hashes) - len(set(row_hashes)),
                "nonfinite_values": int((~np.isfinite(s.X)).sum())}
    tr_hash = {hashlib.sha256(row.tobytes()).hexdigest() for row in train.X}
    te_hash = {hashlib.sha256(row.tobytes()).hexdigest() for row in test.X}
    return {"train": summary(train), "test": summary(test), "subject_overlap": [],
            "exact_feature_overlap": len(tr_hash & te_hash),
            "repeated_feature_names": len(train.feature_names) - len(set(train.feature_names)),
            "upstream_preprocessing": "使用官方已提取、已归一化的561维特征；不能据此审计原始特征归一化的拟合范围。"}
