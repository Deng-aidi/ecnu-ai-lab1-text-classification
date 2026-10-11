# ecnu-ai-lab1-text-classification
当代人工智能实验1的仓库

## 项目简介
基于 TF-IDF 特征与多种经典分类器（NB、SVM、LR、MLP）的新闻文本 10 分类实验。实验严格遵循“先划分、后拟合”的防数据泄露规范。

## 环境依赖
- Python 3.7+
- scikit-learn, pandas, numpy, matplotlib
- nltk（可选，用于词干提取对照实验）

安装依赖：
```bash
pip install scikit-learn pandas numpy matplotlib nltk

（运行实验可能需要几分钟左右，会产生一系列结果文件，详情可以看data-Project文件夹里面的README.md文件，本次实验的代码也放到了文件夹下）
