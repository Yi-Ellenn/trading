# 加密货币市场预测

本项目使用LightGBM实现加密货币市场预测的机器学习模型。

## 文件说明

- `benchmark_simple.py` - 基础LightGBM模型实现
- `benchmark_simple_v3.py` - 使用Optuna进行超参数优化的高级版本
- `selected_features.pkl` - 预选择的模型训练特征

## 依赖要求

- polars
- lightgbm  
- numpy
- optuna
- scikit-learn
- joblib

## 使用方法

1. 确保你有以下所需数据文件:
   - `train.parquet` - 训练数据
   - `test.parquet` - 测试数据
   - `sample_submission.csv` - 提交样本格式

2. 运行基础模型:
   ```bash
   python benchmark_simple.py
   ```

3. 运行带超参数调优的优化模型:
   ```bash
   python benchmark_simple_v3.py
   ```

## 模型特点

- 使用预计算的特征重要性进行特征选择
- 使用Optuna进行超参数优化
- 使用交叉验证进行稳健的模型评估
- 使用信息系数(IC)作为评估指标
- 使用并行处理加速训练

## 输出

模型生成可直接用于竞赛提交的文件(`submission_v0.csv`, `submission_v1.csv`)。