# 跨人员人体活动识别与特征压缩

整合五模型项目和单文件 Notebook。保留 RBF-SVM、RandomForest、LogisticRegression、KNN、FisherLDA，以及全特征、PCA、ANOVA 三条路径。
新增不同批量的分阶段计时和对应图表，统一了 Notebook 中的 250 维候选。原始特征数据已放在 `data/UCI HAR Dataset/`，不再复制 Base64 字符串。

## 运行

Python 3.11 及以上，推荐使用独立虚拟环境。依赖版本见 `requirements.txt`。

```bash
python -m pip install -r requirements.txt
python run_experiment.py --out outputs/my_run --workers 2
```

主流程完成训练、调参、维数选择、测试、总体计时、分阶段计时、核验及绘图。每次使用新的输出目录，避免覆盖已完成结果。搜索可并行，预测计时始终单线程。

```bash
python run_experiment.py --plan
python run_experiment.py --resume outputs/my_run
```

第一条只显示工作量；第二条恢复中断的折级断点。恢复时数据、代码、依赖和两个配置文件都不能改变，否则不接受旧断点。

## 同步后的配置

`config.json` 控制模型与验证；`timing_config.json` 单独控制补充计时。

| 模型 | 参数范围 | 每种特征设置的配置数 |
|---|---|---:|
| RBF-SVM | C=1、10、100、1000；gamma=scale、0.001 | 8 |
| RandomForest | max_depth=None、10、20；min_samples_leaf=1、3；固定200棵树 | 6 |
| LogisticRegression | C=0.01、0.1、1、10、100；L2、lbfgs | 5 |
| KNN | n_neighbors=3、5、9、15；weights=uniform、distance；欧氏距离 | 8 |
| FisherLDA | shrinkage=None、auto、0.05、0.2、0.5；solver=lsqr | 5 |

压缩维数为 **20、50、100、150、200、250、300**。每个模型有15种特征设置：全561维1种、PCA7种、ANOVA7种。主搜索共 **480个候选、2400次五折拟合**，另有50次固定参数协议对照和15个最终模型拟合。

加入250维是同步上传Notebook的实际配置，不是看到本轮测试结果之后追加搜索。上传Notebook中的旧文字仍写416组，已在整合版纠正。模型网格、种子、人员折和1个百分点容忍范围未改变。

## 分阶段计时

默认批量为1、32、256、2947，预热3次、重复15次。分别测量：

- `scale`：标准化。
- `reduce`：PCA投影或ANOVA选列。全特征路线没有该操作，记录为结构性的0。
- `classifier`：在已准备好的变换结果上调用分类器。
- `whole`：从原始561维输入调用完整Pipeline。

独立阶段的中位数不能严格相加为完整预测的中位数。计时包含函数调用、输入检查等实现开销，不包含传感器采集、原始特征提取、磁盘读写、模型加载或手机功耗。
批量1只使用固定测试窗口0，是控制微基准；原有64个单窗口调用计时仍保留，二者不能混成同一口径。

每次完整运行应产生3600条阶段原始记录、240行汇总、60组分阶段预测一致性检查和10张新增图。模型和阶段顺序交错执行，减轻长时间顺序测量造成的漂移；记录精度不等于跨机器精度。

## 结果位置

本次交付的重新运行结果在 `outputs/integrated_rerun/`。文稿、图表和统计结论均以该目录为依据。
上传包原结果的必要表格保存在 `provenance/user_run_snapshot/`，只用于追溯，不与新计时拼接。

| 文件 | 内容 |
|---|---|
| `tables/selected_configs.csv` | 验证选定的参数、维数与是否达标 |
| `tables/test_metrics.csv` | 15条最终管线的测试成绩 |
| `tables/test_predictions.csv` | 逐窗口真实标签与预测 |
| `tables/test_subjects.csv` | 各测试人员的表现 |
| `tables/decision_summary.csv` | 验证达标和测试损失分别标注 |
| `tables/cv_candidates.csv`、`cv_folds.csv` | 480候选及2400个折的记录 |
| `tables/timing_raw.csv`、`timing_summary.csv` | 原有总体批量和64个单窗口调用计时 |
| `tables/stage_timing_raw.csv`、`stage_timing_summary.csv` | 新增四种批量、四阶段计时 |
| `tables/PCA_vs_full_with_timing_breakdown.csv` | 相同批量下PCA与全特征速度比及F1差异 |
| `stage_timing_environment.json` | 线程、计时器、输入索引规则、模型哈希 |
| `verification.json`、`stage_timing_verification.json` | 原结果核验与新增计时复算 |
| `figures/` | 分类表现、维数曲线、混淆矩阵、参数热力图及计时图 |
| `models/` | 15个包含预处理的完整Pipeline |
| `provenance/notebook_merge_diff.json` | Notebook与原压缩包的实测差异 |

重新核验或出图，不重新训练：

```bash
python verify_results.py --run outputs/integrated_rerun
python redraw_figures.py --run outputs/integrated_rerun
```

给一个已完成的同版本结果单独补测阶段耗时：

```bash
python measure_stages.py --run outputs/my_run
```

该命令重新计时，不重新训练；计时受当时机器负载影响，重复测量会更新该结果目录的阶段计时表。要保留首次测量，先备份目录。



## 质量检查与结论边界

```bash
python -m unittest discover -s tests -v
```

软件测试不代替真实实验。真实核验、重新预测和复现范围在 `validation/` 中逐项记录。
每个人的窗口只分到一个训练/验证侧。新增标准化、PCA和特征选择在折内拟合，测试人员不参与选参。FisherLDA作为分类器，不调用监督降维接口。
官方测试集在此前探索中已经查看过，因此本轮是扩展探索性复现，不称为全新的外部确认实验。保留未达标和变慢的结果；PCA不能自动带来加速，更不能直接证明手机省电。
