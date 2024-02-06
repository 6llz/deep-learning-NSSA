from keras.models import load_model

import pandas as pd
import numpy as np
# 定义一个函数，根据威胁类型和程度，返回一个威胁评分
from matplotlib import pyplot as plt

import dpre

# 定义一个字典，存储每种攻击类型的威胁程度、脆弱性程度和重要性程度
attack_dict = {
    0: {'T': 0.7, 'V': 0.72, 'I': 0.72},
    1: {'T': 0.21, 'V': 0.29, 'I': 0.21},
    2: {'T': 0.32, 'V': 0.32, 'I': 0.35},
    3: {'T': 0.5, 'V': 0.52, 'I': 0.51},
    4: {'T': 0.01, 'V': 0.01, 'I': 0.01}
}


def situation_score(y_pred_se, N, M):
    # test_y是一个numpy数组，包含测试集的标签
    # y_pred_se是一个numpy数组，包含测试集的预测结果

    S = 0  # 初始化网络安全态势值为0
    for i in range(N):  #
        y_pre = y_pred_se[i]  # 获取第i个样本的预测结果

        T_i = attack_dict[y_pre]['T']  # 获取第i个样本的威胁程度
        V_i = attack_dict[y_pre]['V']  # 获取第i个样本的脆弱性程度
        I_i = attack_dict[y_pre]['I']  # 获取第i个样本的重要性程度
        S_i = T_i * V_i * I_i / M  # 计算第i个样本的安全态势分数
        S += S_i  # 累加所有样本的安全态势分数
    return S


# 加载你的入侵检测模型
model_cnn_gru = load_model('cnn_gru_model.h5', compile=False)
model_cnn_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn = load_model('cnn_model.h5', compile=False)  # 加载cnn模型
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn_lstm = load_model('cnn_lstm_model.h5', compile=False)  # 加载cnn_lstm模型
model_cnn_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn_bigru = load_model('n_cnn_bigru_model.h5', compile=False)  # 加载cnn_bigru模型
model_cnn_bigru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 加载NSL-KDD数据集的测试集
x_train_1, y_train_1, x_test_1, y_test_1 = dpre.dps()
y_test_1 = np.argmax(y_test_1, axis=1)  # 将标签转换为一维的数组
dummies_test = pd.get_dummies(y_test_1)  # 将标签转换为哑变量
y_pred_cnn_gru = model_cnn_gru.predict(x_test_1)  # 对整个测试集进行预测
y_pred_cnn_gru = np.argmax(y_pred_cnn_gru, axis=1)
y_pred_cnn = model_cnn.predict(x_test_1)  # 对整个测试集进行预测
y_pred_cnn = np.argmax(y_pred_cnn, axis=1)
y_pred_cnn_lstm = model_cnn_lstm.predict(x_test_1)  # 对整个测试集进行预测
y_pred_cnn_lstm = np.argmax(y_pred_cnn_lstm, axis=1)
y_pred_cnn_bigru = model_cnn_bigru.predict(x_test_1)  # 对整个测试集进行预测
y_pred_cnn_bigru = np.argmax(y_pred_cnn_bigru, axis=1)
from sklearn.metrics import accuracy_score
accuracy_cnn_bigru = accuracy_score(y_test_1, y_pred_cnn_bigru)
print("The accuracy of the cnn_bigru model is:", accuracy_cnn_bigru)
accuracy_cnn_gru = accuracy_score(y_test_1, y_pred_cnn_gru)
print("The accuracy of the cnn_gru model is:", accuracy_cnn_gru)
N = len(x_test_1)  # 获取样本数量
M = 75  # 定义一个归一化因子，可以根据实际情况调整
n_segments = 50  # 分段数目

true_situation_values = []
predicted1_situation_values = []
predicted2_situation_values = []
predicted3_situation_values = []  # 创建一个空列表，用于存储cnn模型的态势值
predicted4_situation_values = []  # 创建一个空列表，用于存储cnn_lstm模型的态势值
predicted5_situation_values = []  # 创建一个空列表，用于存储cnn_bigru模型的态势值

# 使用 np.array_split 函数将测试集分为 n_segments 段
segments = np.array_split(range(N), n_segments)

# 遍历每段，调用 situation_score 函数计算态势值，并添加到列表中
for segment in segments:
    # 获取当前段的真实标签和预测标签
    y_true_segment = y_test_1[segment]
    y_pred1_segment = y_pred_cnn_gru[segment]
    y_pred2_segment = y_pred_cnn[segment]  # 获取当前段的cnn模型的预测标签
    y_pred3_segment = y_pred_cnn_lstm[segment]  # 获取当前段的cnn_lstm模型的预测标签
    y_pred4_segment = y_pred_cnn_bigru[segment]  # 获取当前段的cnn_bigru模型的预测标签
    predicted1_situation_value_segment = situation_score(y_true_segment, len(segment),M)  # 注意这里传入的是真实标签和预测标签
    predicted1_situation_values.append(predicted1_situation_value_segment)

    predicted2_situation_value_segment = situation_score(y_pred1_segment, len(segment),M)  # 注意这里传入的是真实标签和预测标签
    predicted2_situation_values.append(predicted2_situation_value_segment)

    predicted3_situation_value_segment = situation_score(y_pred2_segment, len(segment),M)  # 注意这里传入的是cnn模型的预测标签
    predicted3_situation_values.append(predicted3_situation_value_segment)

    predicted4_situation_value_segment = situation_score(y_pred3_segment, len(segment),M)  # 注意这里传入的是cnn_lstm模型的预测标签
    predicted4_situation_values.append(predicted4_situation_value_segment)

    predicted5_situation_value_segment = situation_score(y_pred4_segment, len(segment),M)  # 注意这里传入的是cnn_bigru模型的预测标签
    predicted5_situation_values.append(predicted5_situation_value_segment)


def situation_level(value):
    if 0 <= value <= 0.3:
        return '安全'
    elif 0.3 < value <= 0.6:
        return '低风险'
    elif 0.6 < value <= 0.9:
        return '中等风险'
    elif 0.9 < value <= 1.2:
        return '高风险'
    else:
        return '未知'


# 创建一个空列表，用于存储每段的态势等级
true_situation_levels = []
predicted1_situation_levels = []
predicted2_situation_levels = []
predicted3_situation_levels = []
predicted4_situation_levels = []
predicted5_situation_levels = []

# 遍历每段的态势值，调用 situation_level 函数计算态势等级，并添加到列表中
for i in range(n_segments):
    true_situation_value = predicted1_situation_values[i]
    true_situation_level = situation_level(true_situation_value)
    true_situation_levels.append(true_situation_level)
    predicted1_situation_value = predicted2_situation_values[i]
    predicted1_situation_level = situation_level(predicted1_situation_value)
    predicted1_situation_levels.append(predicted1_situation_level)
    predicted2_situation_value = predicted3_situation_values[i]
    predicted2_situation_level = situation_level(predicted2_situation_value)
    predicted2_situation_levels.append(predicted2_situation_level)
    predicted3_situation_value = predicted4_situation_values[i]
    predicted3_situation_level = situation_level(predicted3_situation_value)
    predicted3_situation_levels.append(predicted3_situation_level)
    predicted4_situation_value = predicted5_situation_values[i]
    predicted4_situation_level = situation_level(predicted4_situation_value)
    predicted4_situation_levels.append(predicted4_situation_level)

df = pd.DataFrame({
    '时间段编号': range(1, n_segments + 1),
    'CNN-BiGRU': predicted5_situation_values,
    'CNN-BiGRU等级': predicted4_situation_levels,
    '实际情况': predicted1_situation_values,
    '实际情况等级': true_situation_levels  # 这里漏了一个实际情况等级的列
})

# 使用 to_markdown 方法将 DataFrame 对象转换为 markdown 格式的表格，并打印出来
print(df.to_markdown())

plt.plot(range(1, n_segments + 1), predicted1_situation_values, 'b-',
         label='True Predicted situation')
plt.plot(range(1, n_segments + 1), predicted2_situation_values, 'g-',
         label='Cnn_GRU Predicted situation')
plt.plot(range(1, n_segments + 1), predicted3_situation_values, 'r-',
         label='Cnn Predicted situation')  # 绘制cnn模型的态势值曲线
plt.plot(range(1, n_segments + 1), predicted4_situation_values, 'y-',
         label='Cnn_LSTM Predicted situation')  # 绘制cnn_lstm模型的态势值曲线
plt.plot(range(1, n_segments + 1), predicted5_situation_values, 'm-',
         label='Cnn_BiGRU Predicted situation')  # 绘制cnn_bilstm模型的态势值曲线
plt.xlabel('Segment number')
plt.ylabel('Situation value')
plt.legend()
plt.show()
