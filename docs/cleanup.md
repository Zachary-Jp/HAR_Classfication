# GitHub发布整理

基于本次上传的`HAR_Integrated_Rerun.zip`整理。原上传文件未覆盖，未重新训练模型，未更改参数范围、指标或图表。

## 保留

- 全部训练、数据加载、参数搜索、预测、核验和绘图核心代码，以及五个命令行入口。
- `config.json`、`timing_config.json`、固定版本依赖和全部测试。
- 当前已完成的`outputs/20261027/`中全部41张CSV结果表、人员划分、冻结配置、15份模型参数JSON和运行清单。
- 当前结果42张图片，含五个模型各自的全特征→PCA→ANOVA三联混淆矩阵。
- `report/`的Word、PDF、Markdown初稿及所引用图表，内容保持原样。
- 历史结构化核验JSON，包括旧环境与新环境的差异记录。

## 移到本地资源包

`HAR_Local_Assets.zip`保留以下文件的原始字节：

```text
data/UCI HAR Dataset/                 10个官方数据及说明文件
outputs/20261027/models/*.joblib       15个已保存的完整Pipeline
```

合计25个文件，约157.56 MiB。它们不是无用文件，只是不与轻量源码仓库一起提交。
将资源包内容直接解压到项目根目录，即可恢复当前旧结果的完整核验、重绘、补测和预测功能。
也可以从原始未清理项目复制同一路径的文件。资源包不需要上传到普通Git仓库。

## 从发布副本中移除

- Python编译缓存和`__pycache__/`。
- 已完成实验的2400个折级断点文件。对应分数仍在`cv_candidates.csv`与`cv_folds.csv`中，原始预测也保留。
- `provenance/`里的旧运行快照和整合过程记录；原始上传ZIP仍保存这些资料。
- 已过时的`FILES_SHA256.json`、`STATUS.json`和根目录重复汇总表。
- 依赖旧快照及历史硬编码目录的一次性核查脚本、历史终端日志。
- 与本次实际文件内容不符的Notebook整合说明。当前上传包没有ipynb，未凭空补造。

本次移除的2400个断点属于已完成运行。新实验的断点生成和`--resume`功能没有删除，
但发布的旧结果目录不再用于恢复训练。

## 发布配置

README中的当前结果路径改为`outputs/20261027/`，移除不存在的Notebook使用承诺。
`.gitignore`忽略本地原始数据、二进制模型、虚拟环境、缓存和新实验目录，允许提交当前结果快照。
`.gitattributes`关闭冻结证据文件的换行转换，防止Windows或Git的换行处理破坏既有哈希。
没有擅自替作者选择代码许可证；数据来源仍按UCI原说明署名。

GitHub文档说明，浏览器上传单文件上限为25 MiB；Git推送超过50 MiB会警告，超过100 MiB被阻止。
本次发布包不包含原始特征大文件或二进制模型。发布时提交解压后的文件夹内容，
而不是把项目整体ZIP当作仓库文件。大型资源可单独存放；不需要为本次轻量版本启用LFS。

参考：
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
https://docs.github.com/en/get-started/git-basics/ignoring-files

## 本次检查

在不放入真实数据、模型的轻量目录中执行常规单元测试：53项通过。
`python run_experiment.py --plan`正常显示480个候选、2400次主折拟合。
在单独的临时副本中恢复资源包，运行：

```bash
python verify_results.py --run outputs/20261027
```

主实验及阶段计时核验均通过：15条管线、480个候选；3600条阶段计时、240行汇总。
原绘图接口`draw_all`和`draw_stage_figures`可正常导入。
这是整理后的回归检查与记录复算，未重新训练整个实验，未重新测量耗时，也未更新原图片。

检查记录：`validation/github_cleanup.json`。该文件记录本次整理范围，不替换历史运行的`verification.json`。
