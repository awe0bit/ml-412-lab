# -*- coding: utf-8 -*-
"""
03_划分可视化.py -- 对应 4-1-2 概念：训练集 / 测试集划分

直觉：20 维的特征人眼看不了，先用 PCA 把它压成 2 维（保留差异最大的两个方向），
再把训练集和测试集画在一张散点图上。你会亲眼看到：
1. 训练集和测试集是"混在一起"的——测试集不是偏到某个角落的特殊数据，
   这样测出来的分数才可信；
2. 分层划分（stratify）让两类颜色在两边比例一致。
"""

import os
import matplotlib

matplotlib.use("Agg")  # 无界面环境下把图存成文件，不弹窗
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示成方框
plt.rcParams["svg.fonttype"] = "path"  # SVG 里文字转成矢量路径，换电脑也不乱码

# 1. 数据：现场随机生成 -> 标准化 -> PCA 压到 2 维 -> 分层划分
# SEED = None 表示每次运行都重新随机生成；想复现某次结果就改成任意整数
SEED = None
X, y = make_classification(
    n_samples=20000, n_features=20,
    n_informative=12, n_redundant=4,
    n_classes=2, flip_y=0.02, class_sep=1.2,
    random_state=SEED,
)
X = StandardScaler().fit_transform(X)
X_2维 = PCA(n_components=2, random_state=SEED).fit_transform(X)
X_训练, X_测试, y_训练, y_测试 = train_test_split(
    X_2维, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"PCA 后形状：{X_2维.shape}（20 维 -> 2 维）")
print(f"训练集 {len(X_训练)} 个点，测试集 {len(X_测试)} 个点")

# 2. 画图：颜色 = 真实类别，形状 = 训练/测试
图, 轴 = plt.subplots(figsize=(8, 6))
类别名 = {0: ("类别0", "class 0"), 1: ("类别1", "class 1")}
颜色 = {0: "#2ca02c", 1: "#d62728"}
for 类别 in (0, 1):
    训练点 = X_训练[y_训练 == 类别]
    测试点 = X_测试[y_测试 == 类别]
    名 = 类别名[类别][0] if 有中文 else 类别名[类别][1]
    轴.scatter(训练点[:, 0], 训练点[:, 1], c=颜色[类别], s=6, alpha=0.35,
               rasterized=True,  # 2 万个点转成内嵌位图，SVG 体积才压得住
               label=(f"训练集·{名}" if 有中文 else f"train·{名}"))
    轴.scatter(测试点[:, 0], 测试点[:, 1], c=颜色[类别], s=14,
               marker="^", edgecolors="black", linewidths=0.6, alpha=0.6,
               rasterized=True,
               label=(f"测试集·{名}" if 有中文 else f"test·{名}"))

轴.set_title("训练集 / 测试集划分可视化（PCA 降到 2 维）" if 有中文
             else "Train/Test split visualized with PCA (2D)", **标签参数)
轴.set_xlabel("主成分 1" if 有中文 else "PC 1", **标签参数)
轴.set_ylabel("主成分 2" if 有中文 else "PC 2", **标签参数)
轴.legend(prop=字体, loc="best")
图.tight_layout()

# 3. 存图（和脚本同目录下的 images 文件夹）
# PNG 自己看，SVG 传 GitHub（纯文本，仓库页面直接渲染）
输出_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "images", "03_train_test_split.png")
图.savefig(输出_png, dpi=150)
print(f"图片已保存：{输出_png}")
输出_svg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "images", "03_train_test_split.svg")
图.savefig(输出_svg, format="svg")
print(f"矢量图已保存：{输出_svg}")
