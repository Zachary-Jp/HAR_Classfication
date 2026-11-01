# 上传Notebook与压缩包的差异及处理

依据本次上传的两个文件实际内容比较，不依据先前聊天中的功能描述。
输入文件SHA-256见`provenance/input_files.json`，配置差异见`provenance/notebook_merge_diff.json`。

| Notebook中的内容 | 原压缩包 | 本次处理 |
|---|---|---|
| `TIMING_BATCH_SIZES=(1,32,256,2947)` | 只有全批与64次单窗口调用总体计时 | 新增独立timing_config.json，接入主入口 |
| scale/reduce/classifier/whole分阶段计时 | 无 | 新增stage_timing.py，逐轮数据和统计保存到tables |
| 分阶段图与批量变化图 | 无 | 五个模型各出两张独立图，不用挤在一张多子图中 |
| PCA与全特征耗时对照表 | 无阶段拆分 | 增加按批量的20行对照汇总 |
| dimensions含250维 | 不含250维 | 保留Notebook实际配置，480候选/2400主折拟合 |
| 41MB Base64编码数据 | 已有本地原始特征数据 | 八个必要文件哈希一致，只保留本地数据 |

新增核验超出原Notebook的简单计时：检查各阶段预测与完整Pipeline一致；检查完整性、重复轮次、均值/分位数复算、模型与计时原始文件哈希。
测量方法仍是实际predict调用，但改为在同一轮中随机交错模型和阶段；这个顺序改进不改变分类器，也不改变测试真值。

全特征路线没有reduce，其0不是计时器测得的零延迟。独立中位数不保证可加；B=1的固定窗口不代表所有真实用户的延迟分布。
报告中的总体性能、参数和时间只使用`outputs/integrated_rerun/`。上传包的旧结果保留作追溯，未混入本轮计算。
