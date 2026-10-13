# -*- coding: utf-8 -*-
"""
01_分类实验.py —— 对应 4-1-2 概念：分类问题

直觉：分类 = 给东西贴标签。这里用乳腺癌数据集（breast_cancer），
根据 30 个细胞核测量指标，判断肿瘤是良性(0)还是恶性(1)。

做法：同一份训练/测试集划分下，训练 4 个经典分类器，
直接对比它们在测试集上的准确率，看谁泛化得最好。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据（sklearn 自带，离线可用）
数据 = load_breast_cancer()
X, y = 数据.data, 数据.target
print(f"数据集：{数据.DESCR.splitlines()[0]}")
print(f"样本数：{X.shape[0]}，特征数：{X.shape[1]}，类别：{list(数据.target_names)}")
print(f"类别分布：良性={(y == 0).sum()} 个，恶性={(y == 1).sum()} 个\n")

# 2. 划分训练集 / 测试集
# stratify=y 是"分层划分"：保证训练集和测试集里良/恶性比例都和原数据一致，
# 避免运气不好导致某一类在测试集里太少，测不准。
X_训练, X_测试, y_训练, y_测试 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集：{len(X_训练)} 个样本，测试集：{len(X_测试)} 个样本\n")

# 3. 特征标准化：只在训练集上学均值方差，再应用到测试集
# （KNN 看距离、逻辑回归看梯度，都对特征量级敏感；树模型无所谓，但统一处理更公平。
# 注意：绝不能拿测试集的信息去 fit，这是"数据泄露"。）
scaler = StandardScaler().fit(X_训练)
X_训练 = scaler.transform(X_训练)
X_测试 = scaler.transform(X_测试)

# 4. 定义 4 个分类器：从"一条直线"到"一群树"，复杂度递增
模型们 = {
    "逻辑回归": LogisticRegression(max_iter=2000),
    "K近邻(k=5)": KNeighborsClassifier(),
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(random_state=42),
}

# 5. 逐个训练并打分
print(f"{'模型':<12}{'训练集准确率':<12}{'测试集准确率'}")
print("-" * 36)
for 名字, 模型 in 模型们.items():
    模型.fit(X_训练, y_训练)  # 只用训练集学规律
    训练分 = accuracy_score(y_训练, 模型.predict(X_训练))
    测试分 = accuracy_score(y_测试, 模型.predict(X_测试))  # 测试集是"没见过的新数据"
    print(f"{名字:<12}{训练分:<12.4f}{测试分:.4f}")

print("\n结论：只看【测试集准确率】这一列——它衡量的是模型在没见过的数据上的表现，")
print("这才是模型真正的能力。训练集分数高但测试集低，就是过拟合（见 04 脚本）。")
