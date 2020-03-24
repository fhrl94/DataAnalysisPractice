import graphviz
import pandas as pd

from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# 读取数据
df = pd.read_excel("判断男女.xlsx")

# 探索数据
# 全局信息
print(df.info())

# 连续型信息
print(df.describe())

# 离散型信息
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
print(dvec.feature_names_)
print(df_features)

# 构造决策树ID3
clf = DecisionTreeClassifier(criterion="entropy")
# 决策树训练
clf.fit(df_features, df_labels)
# 决策树预测
# print(type(df_features))
# print(pd.np.array([0, 1, 0, 0, 0, 1, 1, 0, 0,]))
pred_labels = clf.predict(pd.np.array([[1, 0, 0, 0, 0, 1, 1, 0, 0]]))
# 输出分类结果
print("输出分类结果:")
print(pred_labels)
pred_labels = clf.predict(pd.np.array([[0, 0, 1, 0, 0, 1, 0, 1, 0]]))
print("输出分类结果:")
print(pred_labels)
# 决策树准确率
acc_decision_tree = round(clf.score(df_features, df_labels), 6)
print(acc_decision_tree)

df_labels.unique()

# 绘制决策树
decision_tree = "判断男女决策树"
# 1.简单绘制决策树
tree.plot_tree(clf)

# 2.输出决策树, 文字版
r = tree.export_text(decision_tree=clf, feature_names=dvec.feature_names_)
print(r)

# 3.Graphviz形式输出决策树
# pip install graphviz
# 然后官网下载安装包,并将bin路径添加到path中
# 3.1Graphviz形式输出决策树(简单)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render(decision_tree)
# 3.2Graphviz形式输出决策树(视觉优化)
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=dvec.feature_names_,
    class_names=df_labels.unique(),
    filled=True,
    rounded=True,
    special_characters=True,
)
# 处理中文乱码
graph = graphviz.Source(dot_data.replace("helvetica", "FangSong"))
# graph.render("决策树")
# 生成路径在当前文件夹内
graph.view(decision_tree)
# 生成图片
graph.render(filename=decision_tree, format="png")
