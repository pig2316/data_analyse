import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def anticipate(data, education):
    """
    :param data: 训练数据
    :param education: 学历
    :return: 模型得分，10年工作预测
    """
    train = data[data["education"] == education].to_numpy()
    # 特征和标签
    x = train[:, 1:2]  # 获取工龄
    y = train[:, 2]  # 获取薪资

    # model 训练
    model = LinearRegression()
    model.fit(x, y)

    # model 预测
    pur = [[i] for i in range(11)]
    return model.score(x, y), model.predict(pur)


education_list = [
    "不限",
    "小学",
    "初中",
    "中专",
    "高中",
    "大专",
    "本科",
    "硕士",
    "博士",
]
df = pd.read_csv("./train.csv")

scores, values = [], []
for education in education_list:
    score, y = anticipate(df, education)
    scores.append(score)
    values.append(y)
print(scores)
print(values)

result = pd.DataFrame()
result["学历"] = education_list
result["模型得分"] = scores
result["(1年经验)平均工资"] = [value[1] for value in values]
result["(3年经验)平均工资"] = [value[2] for value in values]
result["(5年经验)平均工资"] = [value[4] for value in values]
result["(10年经验)平均工资"] = [value[10] for value in values]
print(result)
