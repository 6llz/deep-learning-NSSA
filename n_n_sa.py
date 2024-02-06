from keras.models import load_model

import pandas as pd
import numpy as np
# 定义一个函数，根据威胁类型和程度，返回一个威胁评分
from matplotlib import pyplot as plt

import dpre

# 定义一个字典，存储每种攻击类型的威胁程度、脆弱性程度和重要性程度
attack_dict = {
    0: {'T': 0.6, 'V': 0.59, 'I': 0.65},
    1: {'T': 0.01, 'V': 0.09, 'I': 0.01},
    2: {'T': 0.29, 'V': 0.29, 'I': 0.25},
    3: {'T': 0.43, 'V': 0.37, 'I': 0.31},
    4: {'T': 0.42, 'V': 0.46, 'I': 0.33}
}


def situation_score(y_pred_se, N, M):
    # test_y是一个numpy数组，包含测试集的标签
    # y_pred_se是一个numpy数组，包含测试集的预测结果

    S = 0  # 初始化网络安全态势值为0
    for i in range(N):  #
        y_pre = y_pred_se[i]  # 获取第i个样本的预测结果
        y_pre = dummies_test.columns[int(y_pre)]  # 将数字的字符串转换为攻击类型的字符串
        T_i = attack_dict[y_pre]['T']  # 获取第i个样本的威胁程度
        V_i = attack_dict[y_pre]['V']  # 获取第i个样本的脆弱性程度
        I_i = attack_dict[y_pre]['I']  # 获取第i个样本的重要性程度
        S_i = T_i * V_i * I_i / M  # 计算第i个样本的安全态势分数
        S += S_i  # 累加所有样本的安全态势分数
    return S


# 加载你的入侵检测模型
model = load_model('cnn_model.h5')

# 加载NSL-KDD数据集的测试集
x_train_1, y_train_1, x_test_1, y_test_1 = dpre.dps()
y_test_1 = y_test_1.argmax(axis=1)  # 将标签转换为一维的数组
dummies_test = pd.get_dummies(y_test_1)  # 将标签转换为哑变量
y_pred = model.predict(x_test_1)  # 对整个测试集进行预测
N = len(x_test_1)  # 获取样本数量
M = 0.01  # 定义一个归一化因子，可以根据实际情况调整
n_segments = 50  # 分段数目

true_situation_values = []
predicted1_situation_values = []
predicted2_situation_values = []

# 使用 np.array_split 函数将测试集分为 n_segments 段
segments = np.array_split(range(N), n_segments)

# 遍历每段，调用 situation_score 函数计算态势值，并添加到列表中
for segment in segments:
    # 获取当前段的真实标签和预测标签
    y_true_segment = y_test_1[segment]
    y_pred1_segment = y_pred[segment]
    predicted1_situation_value_segment = situation_score(y_true_segment, len(segment),
                                                         M)  # 注意这里传入的是真实标签和预测标签
    predicted1_situation_values.append(predicted1_situation_value_segment)
    print(predicted1_situation_value_segment)




