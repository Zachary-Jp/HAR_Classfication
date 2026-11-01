# 实验约定

## 不变的评价流程

使用官方UCI HAR的561维已提取特征，保留21名训练人员与9名测试人员的划分。
训练集内部使用5折StratifiedGroupKFold；同一人员不跨某一折的训练/验证边界。
subject编号不进入X。原始特征和后续变换按官方列序读取，不能按重复特征名删列。

标准化、PCA和ANOVA均只在每折训练部分拟合。
代码共享每折的已拟合变换及完整SVD结果，但每个分类器候选都实际训练；不是在全数据上先降维再交叉验证。
所有特征方案使用相同人员折；同一分类器各维数使用相同参数网格。

每个分类器先确定全特征验证基准，然后对PCA、ANOVA各自选择验证Macro-F1下降不超过1个百分点的最小候选维数。
没有达标候选时，保留验证分数最高的失败诊断配置并标记`meets_cv_tolerance=False`，不自动放宽标准。
先写入选择锁，再拟合最终模型并读取测试数组。测试后不修改已经锁定的维数。

该研究比较调参后的完整管线，不声称把全部变化归因于降维一个因素。
随机森林固定max_features=sqrt，其实际每次分裂考虑的特征数量仍随输入维数变化。
固定参数的协议对照只使用官方训练集；随机窗口与人员分组分数差不是纯因果效应估计。

## FisherLDA

本版本用sklearn.discriminant_analysis.LinearDiscriminantAnalysis表示多类Fisher/LDA分类基线。
固定lsqr求解器；其支持收缩估计（默认svd求解器不支持该参数），先验由当前训练折类别比例估计。
与原先None/auto两组相比，现在加入0.05、0.2、0.5三种数值协方差收缩强度。
None为不收缩；auto由训练数据自动估计收缩强度；固定数值从弱到强覆盖更宽的正则化程度。
这五个值是事前选定的有限候选，不是针对测试表现手动调整，也不代表全局最优；按训练人员五折Macro-F1选最优值。
各个特征维数使用同一Fisher参数网格，保持比较公平。
这不是手工二分类投影公式，也不宣称完全相同于所有教材中的实现。
此处把LDA用作分类器，不把它作为额外降维路径；不设置n_components=200。
六类问题的LDA判别投影最多5维，是另一个概念。

## 图表简化后仍保留的解释边界

混淆矩阵：行是真实活动，列是预测活动；单元格数字和白→深蓝底色都是样本数。
所有模型共用同一计数颜色上限，避免同一颜色代表不同计数量级。
参数热力图：仅使用所选维数的训练人员交叉验证Macro-F1；未评估格子是NaN，不补0。
维数曲线误差线：验证折间标准差，不是置信区间，折之间不视为独立重复实验。
人员热力图：每个人各自计算六类Macro-F1；总体窗口Macro-F1不等于人员等权平均。
计时图：中位数与四分位区间；batch与single是不同负载，不混用。
模型体积图：完整joblib管线的文件大小，不是运行峰值内存。

图片不再加“†/‡”脚注；失败状态仍保存在`selected_configs.csv`和`decision_summary.csv`。
删除图内解释文字不等于删除负面结果或允许宣称所有压缩方案成功。
原始信号特征提取、手机端部署和功耗均未测量。

## 版本与复现

参考四模型结果在`outputs/reference_4models/`，保存的是过去版本的真实实验，不含Fisher的新参数成绩。
当前五模型新配置在根目录config.json，运行会建立新的outputs/compact_目录。
已查看过官方测试集，因此新增Fisher和缩小/扩展网格属于后续探索性实验。
新研究不能靠反复测试后修改网格来制造“提升”。

断点包含数据、源码、配置、软件版本与平台的指纹；配置变化时不能复用断点。
绘图只是读取结果，不重新训练；保存图片时检查文字边界。
数值复算和日志检查不等于对整个搜索独立重训，也不是不可篡改的第三方预注册。

## 原始资料

UCI HAR数据集：DOI 10.24432/C54S4K。
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

LinearDiscriminantAnalysis参数、求解器和降维限制：
https://scikit-learn.org/1.8/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

交叉验证和预处理边界：
https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
https://scikit-learn.org/1.8/common_pitfalls.html

Blues混淆矩阵示例：
https://scikit-learn.org/1.8/auto_examples/model_selection/plot_confusion_matrix.html


## Notebook功能整合
分阶段计时细则见Notebook整合说明.md及timing_config.json。正文若出现旧快照成绩，只能作为历史比较；本轮报告使用integrated_rerun。
