#!/usr/bin/env python
# coding: utf-8


import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

# 读取数据
df = pd.read_excel("判断男女.xlsx")

# 探索数据
# 全局信息
print(df.info())

# 连续型信息
print(df.describe())

# 离散型信息,字母"O"
print(df.describe(include=["O"]))

# 输出前5个
print(df.head())

# 输出后5个
print(df.tail())

# 选择特征数据
features = ["身高", "体重", "鞋码"]
df_features = df[features]
# 选择结果数据
df_labels = df["性别"]

# 生成特征值矩阵
dvec = DictVectorizer(sparse=False)
# 新矩阵
df_features = dvec.fit_transform(df_features.to_dict(orient="record"))
# 矩阵转换
print(dvec.feature_names_)
print(df_features)

# 构造多项式朴素贝叶斯,拉普拉斯修正为1
clf = MultinomialNB(alpha=1)
# 伯努利朴素贝叶斯,拉普拉斯修正为1
# from sklearn.naive_bayes import BernoulliNB
#
# clf = BernoulliNB(alpha=1)

# 决策树训练
clf.fit(df_features, df_labels)
# 决策树预测 身高=高、体重=中、鞋码=中
pred_labels = clf.predict(pd.np.array([[1, 0, 0, 0, 0, 1, 1, 0, 0]]))
print(pred_labels)
# 决策树准确率, 数据量较少,不做准确率判断
# acc_decision_tree = round(clf.score(df_features, df_labels), 6)
# print(acc_decision_tree)
