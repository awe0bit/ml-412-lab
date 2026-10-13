# -*- coding: utf-8 -*-
"""
04_过拟合可视化.py -- 对应 4-1-2 概念：过拟合

直觉：决策树的 max_depth（最大深度）= 模型能有多"钻牛角尖"。
深度 1 只能画一条线（太笨，欠拟合）；深度 15 能把每个训练样本都单独记住
（太聪明，过拟合——把噪声也当规律背下来了）。

这张图让你亲眼看到拐点：训练准确率一路涨到 100%，
而测试准确率先涨后跌——跌下去的那段，就是过拟合。
"""

import os
import matplotlib

matplotlib.use("Agg")  # 无界面环境下把图存成文件，不弹窗
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def 中文字体():
    """找一个可用的中文字体，找不到返回 None（此时用英文标签）。"""
    import os as _os

    路径候选 = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 本机 Linux
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    for 路径 in 路径候选:
        if _os.path.exists(路径):
            try:
                return FontProperties(fname=路径)
            except Exception:
                pass
    from matplotlib.font_manager import fontManager

    已装 = {f.name for f in fontManager.ttflist}
    for 名字 in ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB"]:
        if 名字 in 已装:
            return FontProperties(family=名字)
    return None


字体 = 中文字体()
有中文 = 字体 is not None
标签参数 = {"fontproperties": 字体} if 有中文 else {}
plt.rcParams["axes.unicode_minus"] = False

# 1. 数据准备：现场随机生成（和 01 同一套生成参数）
# SEED = None 表示每次运行都重新随机生成；想复现某次结果就改成任意整数
SEED = None
X, y = make_classification(
    n_samples=20000, n_features=20,
    n_informative=12, n_redundant=4,
    n_classes=2, flip_y=0.02, class_sep=1.2,
    random_state=SEED,
)
X_训练, X_测试, y_训练, y_测试 = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
scaler = StandardScaler().fit(X_训练)
X_训练, X_测试 = scaler.transform(X_训练), scaler.transform(X_测试)

# 2. 深度从 1 到 15，逐个训练并记录两条曲线
深度们, 训练分们, 测试分们 = [], [], []
for 深度 in range(1, 16):
    树 = DecisionTreeClassifier(max_depth=深度, random_state=SEED)
    树.fit(X_训练, y_训练)
    深度们.append(深度)
    训练分们.append(accuracy_score(y_训练, 树.predict(X_训练)))
    测试分们.append(accuracy_score(y_测试, 树.predict(X_测试)))

# 3. 打印拐点附近的数据
最佳深度 = 深度们[测试分们.index(max(测试分们))]
print(f"{'深度':<6}{'训练准确率':<12}{'测试准确率'}")
print("-" * 30)
for d, tr, te in zip(深度们, 训练分们, 测试分们):
    标记 = " <-- 测试集最高点" if d == 最佳深度 else ""
    print(f"{d:<6}{tr:<12.4f}{te:.4f}{标记}")
print(f"\n拐点解读：深度={最佳深度} 时测试准确率最高；")
print("再往深，训练准确率继续涨（甚至到 1.0），测试准确率反而跌——")
print("模型开始死记硬背训练集的噪声，这就是过拟合。")

# 4. 画图
图, 轴 = plt.subplots(figsize=(8, 5.5))
轴.plot(深度们, 训练分们, "o-", label="训练集准确率" if 有中文 else "train accuracy")
轴.plot(深度们, 测试分们, "s-", label="测试集准确率" if 有中文 else "test accuracy")
轴.axvline(最佳深度, color="gray", linestyle="--", linewidth=1)
轴.text(最佳深度 + 0.2, min(测试分们) + 0.01,
        f"最佳深度={最佳深度}" if 有中文 else f"best depth={最佳深度}",
        **标签参数)
轴.set_title("决策树深度 vs 准确率：过拟合拐点" if 有中文
             else "Decision tree depth vs accuracy: overfitting", **标签参数)
轴.set_xlabel("树的最大深度 max_depth" if 有中文 else "max_depth", **标签参数)
轴.set_ylabel("准确率" if 有中文 else "accuracy", **标签参数)
轴.legend(prop=字体, loc="best")
轴.set_xticks(深度们)
轴.set_ylim(min(测试分们) - 0.03, 1.02)
图.tight_layout()

输出 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "images", "04_overfitting_curve.png")
图.savefig(输出, dpi=150)
print(f"图片已保存：{输出}")
