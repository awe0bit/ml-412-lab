# -*- coding: utf-8 -*-
"""
02_回归实验.py —— 对应 4-1-2 概念：回归问题

直觉：回归 = 预测一个连续的数字。分类是贴标签（良性/恶性），
回归是估数值。这里用糖尿病数据集（diabetes），根据 10 个生理指标，
预测一年后病情的发展程度（一个连续分数）。

评价指标：
- R²（决定系数）：模型解释了多少比例的波动。1.0 = 完美，0 = 不如直接猜均值，负数 = 比猜均值还烂。
- RMSE（均方根误差）：预测值和真实值平均差多少，越小越好，单位和目标值一致。
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 加载数据（sklearn 自带，离线可用）
数据 = load_diabetes()
X, y = 数据.data, 数据.target
print(f"数据集：糖尿病病情进展预测")
print(f"样本数：{X.shape[0]}，特征数：{X.shape[1]}")
print(f"目标值范围：{y.min():.0f} ~ {y.max():.0f}（分数越高病情进展越快）\n")

# 2. 划分训练集 / 测试集（回归问题不需要 stratify）
X_训练, X_测试, y_训练, y_测试 = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集：{len(X_训练)} 个样本，测试集：{len(X_测试)} 个样本\n")

# 3. 定义 4 个回归器：从"一条直线"到"一群树"
模型们 = {
    "线性回归": LinearRegression(),
    "岭回归(Ridge)": Ridge(), # 线性回归 + 正则化：压住过大的系数，防过拟合
    "决策树回归": DecisionTreeRegressor(random_state=42),
    "随机森林回归": RandomForestRegressor(random_state=42),
}

# 4. 逐个训练并打分（看测试集！）
print(f"{'模型':<12}{'测试集 R²':<12}{'测试集 RMSE'}")
print("-" * 36)
for 名字, 模型 in 模型们.items():
    模型.fit(X_训练, y_训练)
    预测 = 模型.predict(X_测试)
    r2 = r2_score(y_测试, 预测)
    rmse = np.sqrt(mean_squared_error(y_测试, 预测))
    print(f"{名字:<12}{r2:<12.4f}{rmse:.2f}")

print("\n结论：R² 越接近 1 越好，RMSE 越小越好。")
print("注意单棵决策树在这里的表现——它在训练集上几乎完美，")
print("但测试集上明显不如随机森林，这就是过拟合的典型症状（见 04 脚本）。")
