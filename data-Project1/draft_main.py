import os
import re
import string
import platform
import json
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss

# 抑制警告
warnings.filterwarnings('ignore')
# 服务器无图形界面环境，强制使用 Agg 后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 尝试导入 nltk 词干提取器
try:
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("警告: 未安装 nltk，词干提取对照实验将被跳过。可运行 pip install nltk 安装。")

# ================= 可复现性加固 =================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ================= 全局配置 =================
TEST_SIZE = 0.2
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
DO_STEM = False  # 主实验默认关闭词干提取
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ============================================

def log_step(step, title):
    """输出实验步骤节点"""
    print(f"\n[{step}] {title}")
    print("-" * 50)

def load_and_split_data():
    """先划分，后清洗（防数据泄露）"""
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_raw = train_df['text'].astype(str).tolist()
    y = train_df['target'].values
    X_test_raw = test_df['text'].astype(str).tolist()
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train_raw, X_val_raw, y_train, y_val, X_test_raw

def clean_text(text, do_stem=False):
    """无状态清洗：去邮件头 / URL / 标点 / 数字，统一小写，可选词干提取"""
    text = re.sub(r'^(From|Subject|Organization|Lines|Reply-To|NNTP-Posting-Host|Distribution|X-Newsreader|X-Mailer):.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    if do_stem and NLTK_AVAILABLE:
        stemmer = PorterStemmer()
        words = text.split()
        words = [stemmer.stem(w) for w in words]
        text = ' '.join(words)
    return text

def evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val):
    """训练并返回验证集准确率和 Macro-F1"""
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    return acc, f1, y_pred

def main():
    # ================= [1] 数据划分 =================
    log_step(1, "数据划分")
    print(f"Python {platform.python_version()}, sklearn {sklearn.__version__}, seed={RANDOM_STATE}")
    X_train_raw, X_val_raw, y_train, y_val, X_test_raw = load_and_split_data()
    print(f"train={len(X_train_raw)}, val={len(X_val_raw)}, test={len(X_test_raw)}")

    # ================= [2] 文本预处理 =================
    log_step(2, "文本预处理")
    print(f"TF-IDF({MAX_FEATURES}, {NGRAM_RANGE}) + 英文停用词 + 无状态清洗")
    X_train_clean = [clean_text(t, do_stem=DO_STEM) for t in X_train_raw]
    X_val_clean   = [clean_text(t, do_stem=DO_STEM) for t in X_val_raw]
    X_test_clean  = [clean_text(t, do_stem=DO_STEM) for t in X_test_raw]

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, stop_words='english'
    )
    # 关键规范：TF-IDF 只在训练集 fit_transform，验证/测试集只 transform
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    X_val_tfidf   = vectorizer.transform(X_val_clean)
    X_test_tfidf  = vectorizer.transform(X_test_clean)
    print(f"训练集维度: {X_train_tfidf.shape}")

    # ================= [3] 模型训练与验证集选择 =================
    log_step(3, "模型训练与验证集选择")
    models = {
        "NB (Multinomial)": MultinomialNB(),
        "SVM (linear, C=1.0)": SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        "LR (C=0.1)": LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE),
        "LR (C=1.0)": LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        "MLP (100,)": MLPClassifier(hidden_layer_sizes=(100,),   max_iter=500, random_state=RANDOM_STATE),
        "MLP (100,50)": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=RANDOM_STATE),
    }
    rows = []
    best_name, best_acc, best_model, best_f1 = "", 0.0, None, 0.0
    for name, model in models.items():
        acc, f1, y_pred = evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val)
        rows.append({"模型": name, "验证集准确率": round(acc, 4), "Macro-F1": round(f1, 4)})
        if acc > best_acc:
            best_name, best_acc, best_model, best_f1 = name, acc, model, f1

    df = pd.DataFrame(rows).sort_values("验证集准确率", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
    df.to_csv('val_result.csv', index=False, encoding='utf-8-sig')

    # 绘制模型准确率对比柱状图
    plt.figure(figsize=(10, 5))
    bars = plt.bar(df["模型"], df["验证集准确率"], color='#4c72b0')
    plt.ylabel("验证集准确率")
    plt.title("模型验证集准确率对比")
    plt.xticks(rotation=15, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f"{height:.4f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('model_acc_bar.png', dpi=300)
    print("已生成 model_acc_bar.png")

    # ================= [4] 最佳模型锁定 =================
    log_step(4, "最佳模型锁定")
    print(f"选择 {best_name}")
    print(f"   验证集准确率: {best_acc:.4f}, Macro-F1: {best_f1:.4f}")
    if "SVM" in best_name:
        print(f"   理由: 验证集表现最优，线性核 SVM 在高维稀疏文本上表现稳健")
    elif "LR" in best_name:
        print(f"   理由: 验证集表现最优，线性模型在高维稀疏文本上稳定")
    elif "NB" in best_name:
        print(f"   理由: 验证集表现最优，朴素贝叶斯作为基线效率极高")
    else:
        print(f"   理由: 验证集表现最优，非线性模型捕捉了更复杂的语义模式")

    # ================= [5] 类别层面错误分析 =================
    log_step(5, "类别层面错误分析")
    y_pred_best = best_model.predict(X_val_tfidf)
    report = classification_report(y_val, y_pred_best, zero_division=0)
    print(report)
    with open('class_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # 混淆矩阵热力图
    cm = confusion_matrix(y_val, y_pred_best)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_val)))
    labels = [f"类别 {i}" for i in tick_marks]
    plt.xticks(tick_marks, labels, rotation=60, fontsize=8)
    plt.yticks(tick_marks, labels, fontsize=8)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("混淆矩阵已保存至 confusion_matrix.png，分类报告已保存至 class_report.txt")

    # ================= [6] MLP 训练动态分析 =================
    log_step(6, "MLP 训练动态分析")
    # 注意：此处为独立实例，用于演示 MLP 训练动态，不参与前面的候选模型选择
    # 使用 warm_start + fit 循环，模拟逐 epoch 训练，获取真实 loss（保留抖动）
    mlp_dyn = MLPClassifier(
        hidden_layer_sizes=(100,), max_iter=1, warm_start=True,
        random_state=RANDOM_STATE, learning_rate_init=0.001
    )
    train_losses = []
    val_losses = []
    for epoch in range(300):
        mlp_dyn.fit(X_train_tfidf, y_train)
        train_losses.append(mlp_dyn.loss_)
        val_proba = mlp_dyn.predict_proba(X_val_tfidf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_losses.append(log_loss(y_val, val_proba))

    train_losses = pd.Series(train_losses)
    val_losses = pd.Series(val_losses)
    # 平滑窗口缩小为2，降低平滑强度；平滑曲线放底层，原始曲线在上层突出真实抖动
    smooth_train = train_losses.rolling(window=2, center=True).mean()
    smooth_val = val_losses.rolling(window=2, center=True).mean()

    plt.figure(figsize=(10, 5))
    # 平滑曲线置于底层 zorder=0，作为辅助参考
    plt.plot(smooth_train, label='训练 Loss（平滑, window=2）',
             color='#1f77b4', linewidth=2, linestyle='--', alpha=0.4, zorder=0)
    plt.plot(smooth_val, label='验证 Loss（平滑, window=2）',
             color='#d62728', linewidth=2, linestyle='--', alpha=0.4, zorder=0)
    # 原始真实损失放在上层，优先展示训练真实震荡
    plt.plot(train_losses, label='训练 Loss（原始真实）',
             color='#1f77b4', linewidth=1.2, alpha=0.95, zorder=10)
    plt.plot(val_losses, label='验证 Loss（原始真实）',
             color='#d62728', linewidth=1.2, alpha=0.95, zorder=10)

    plt.title('MLP 训练与验证损失曲线（原始真实损失为主）', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('loss_report.png', dpi=300)
    print("已生成 loss_report.png")
    print(f"最终训练 Loss: {train_losses.iloc[-1]:.4f}")
    print(f"最终验证 Loss: {val_losses.iloc[-1]:.4f}")
    print(f"Loss 差值: {abs(train_losses.iloc[-1] - val_losses.iloc[-1]):.4f}")

    # ================= [7] 词干提取对照实验（方案A） =================
    if NLTK_AVAILABLE:
        log_step(7, "词干提取对照实验")
        results_stem = []
        for do_stem in [False, True]:
            X_tr = [clean_text(t, do_stem=do_stem) for t in X_train_raw]
            X_v  = [clean_text(t, do_stem=do_stem) for t in X_val_raw]
            vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, stop_words='english')
            X_tr_tfidf = vec.fit_transform(X_tr)
            X_v_tfidf = vec.transform(X_v)
            lr = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
            lr.fit(X_tr_tfidf, y_train)
            y_pred = lr.predict(X_v_tfidf)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='macro')
            results_stem.append({"词干提取": "开启" if do_stem else "关闭", "验证集准确率": round(acc, 4), "Macro-F1": round(f1, 4)})
        df_stem = pd.DataFrame(results_stem)
        print(df_stem.to_string(index=False))
        df_stem.to_csv('stemming_comparison.csv', index=False, encoding='utf-8-sig')
    else:
        log_step(7, "词干提取对照实验（跳过）")
        print("未安装 nltk，跳过词干提取对照。")

    # ================= [8] 最终测试集预测 =================
    log_step(8, "最终测试集预测")
    # 说明：模型选择阶段 vectorizer 只在 train 上 fit，是防止数据泄露。
    # 方案锁定后，用 train+val 重新 fit 一个新的 vectorizer_full，
    # 充分利用全部可用训练数据，测试集全程未参与 fit，无数据泄露。
    X_full_clean = X_train_clean + X_val_clean
    y_full = np.concatenate([y_train, y_val])

    vectorizer_full = TfidfVectorizer(
        max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, stop_words='english'
    )
    X_full_tfidf = vectorizer_full.fit_transform(X_full_clean)   # 在 train+val 上 fit
    X_test_tfidf = vectorizer_full.transform(X_test_clean)       # 测试集只 transform

    if "SVM" in best_name:
        final_model = SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE)
    elif "LR" in best_name:
        C_val = 0.1 if "0.1" in best_name else 1.0
        final_model = LogisticRegression(C=C_val, max_iter=2000, random_state=RANDOM_STATE)
    elif "NB" in best_name:
        final_model = MultinomialNB()
    else:
        hidden = (100,) if "100,)" in best_name else (100, 50)
        final_model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=RANDOM_STATE)

    final_model.fit(X_full_tfidf, y_full)
    y_pred_test = final_model.predict(X_test_tfidf)
    pd.DataFrame(y_pred_test).to_csv('predictions.csv', index=False, header=False)
    print(f"使用 train+val 重训 {best_name}")
    print(f"预测 {len(y_pred_test)} 条，保存至 predictions.csv")
    
    # ================= [9] 保存实验元信息 =================
    log_step(9, "实验元信息保存")
    exp_log = {
        "随机种子": RANDOM_STATE,
        "数据集划分比例": f"train={len(X_train_raw)}, val={len(X_val_raw)}, test={len(X_test_raw)}",
        "TF-IDF超参": {"max_features": MAX_FEATURES, "ngram_range": NGRAM_RANGE, "stop_words": "english"},
        "最优模型": best_name,
        "最优验证指标": {"accuracy": round(best_acc, 4), "macro_f1": round(best_f1, 4)},
        "Python版本": platform.python_version(),
        "scikit-learn版本": sklearn.__version__
    }
    with open('exp_log.json', 'w', encoding='utf-8') as f:
        json.dump(exp_log, f, indent=4, ensure_ascii=False)
    print("实验元信息已保存至 exp_log.json")

if __name__ == '__main__':
    main()
