import graphviz
import pandas as pd


from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# 读取数据
df = pd.read_excel("是否打篮球决策.xlsx")

# 探索数据
# 数据信息
print(df.info())
# 连续型描述
print(df.describe())
# 离散型描述,参数是字母 O, 不是0
print(df.describe(include=["O"]))
# 输出前5个数据
print(df.head())
# 输出后5个数据
print(df.tail())
# 选择特征数据
features = ["天气", "温度", "湿度", "刮风"]
df_features = df[features]
# 选择结果数据
df_labels = df["是否打篮球"]
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
pred_labels = clf.predict(df_features)
# 决策树准确率
acc_decision_tree = round(clf.score(df_features, df_labels), 6)
print(acc_decision_tree)
# 绘制决策树
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
graph.render("决策树")
# 3.2Graphviz形式输出决策树(视觉优化)
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=dvec.feature_names_,
    class_names=["不打篮球", "打篮球"],
    filled=True,
    rounded=True,
    special_characters=True,
)
# 处理中文乱码
graph = graphviz.Source(dot_data.replace("helvetica", "FangSong"))
graph.render(filename="决策树", format="png")
# graph.view("决策树")
