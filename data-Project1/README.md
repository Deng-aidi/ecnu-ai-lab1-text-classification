```markdown
# 当代人工智能实验一：文本分类

## 项目简介
本实验针对新闻文本进行 0-9 十分类，采用 TF-IDF 特征，对比朴素贝叶斯（NB）、逻辑回归（LR）、支持向量机（SVM）与多层感知机（MLP）四种经典分类器。实验严格遵循“先划分数据，再仅用训练集拟合特征”的防数据泄露规范，用验证集选出最佳模型并进一步优化和实验之后，用这个最优的模型去做测试集实验并得到predicition.csv

## 环境依赖
- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- nltk（可选，用于设置词干提取对照实验）

安装依赖：
```bash
pip install scikit-learn pandas numpy matplotlib nltk
```

## 运行方式
1. 进入项目目录：
   ```bash
   cd data-Project1
   ```
2. 运行主程序：
   ```bash
   python draft_main.py
   ```

## 实验流程
1. **数据加载与划分**  
   读取 `train_data.csv` 和 `test_data_unlabeled.csv`。原始训练集按 8:2 分层划分为训练集与验证集，测试集全程封存，仅用于最终预测。

2. **文本预处理**  
   无状态清洗：去除邮件头、URL、标点、数字，统一转为小写。可选词干提取（PorterStemmer），默认关闭。

3. **特征提取**  
   使用 TF-IDF，`max_features=5000`，`ngram_range=(1,2)`，停用词 `english`。**仅在训练集上拟合**，验证集与测试集只做特征转换。

4. **模型训练与验证**  
   训练 NB、SVM、LR、MLP 等多种模型，以验证集准确率和 Macro-F1 为指标，选择最优模型。

5. **最优模型深入分析**  
   - 类别层面错误分析：生成分类报告与混淆矩阵。  
   - MLP 训练动态：绘制训练/验证损失曲线（原始真实 vs 平滑）。  
   - 词干提取消融实验：固定 LR (C=1.0)，对比开启/关闭词干提取的效果。

6. **最终模型重训与预测**  
   将训练集与验证集合并，重新拟合 TF-IDF 词表（测试集全程未参与），用最优模型配置重新训练，预测测试集并输出 `predictions.csv`。

7. **实验元信息保存**  
   记录随机种子、数据划分比例、TF-IDF 超参、最优模型及版本信息至 `exp_log.json`。

## 输出文件说明
| 文件名 | 说明 |
|---|---|
| `val_result.csv` | 各模型在验证集上的准确率与 Macro-F1 |
| `model_acc_bar.png` | 模型验证集准确率对比柱状图 |
| `class_report.txt` | 最优模型的分类报告 |
| `confusion_matrix.png` | 最优模型的混淆矩阵热力图 |
| `loss_report.png` | MLP 训练/验证损失曲线（含原始真实与平滑） |
| `stemming_comparison.csv` | 词干提取对照实验结果（需安装 nltk） |
| `predictions.csv` | 最终测试集预测结果（提交用） |
| `exp_log.json` | 实验元信息（随机种子、超参、版本等） |

## 文件结构
```
data-Project1/
├── draft_main.py               # 主实验代码
├── train_data.csv              # 带标签训练集
├── test_data_unlabeled.csv     # 无标签测试集
├── README.md                   # 本文件
└── （运行后生成以下文件）
    ├── val_result.csv
    ├── model_acc_bar.png
    ├── class_report.txt
    ├── confusion_matrix.png
    ├── loss_report.png
    ├── stemming_comparison.csv
    ├── predictions.csv
    └── exp_log.json
```

## 实验配置
- 随机种子：`RANDOM_STATE = 42`
- 数据划分：训练集 80%，验证集 20%，分层抽样
- TF-IDF：`max_features=5000`，`ngram_range=(1,2)`，`stop_words='english'`
- 模型：MultinomialNB、SVC(linear, C=1.0)、LogisticRegression(C=0.1/1.0)、MLPClassifier((100,), (100,50))

## 防数据泄露说明
- 数据先划分，后清洗，清洗为无状态操作。
- TF-IDF 词表仅在训练集上拟合，验证集/测试集只做 `transform`。
- 最终模型在 `train+val` 上重新拟合词表并重训，测试集全程未参与任何拟合过程。
