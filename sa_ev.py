from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import dpre

x_train_1, y_train_1, x_test_1, y_test_1 = dpre.dps()
# 加载四个模型
model1 = load_model('n_cnn_bigru_model_1.h5')
model2 = load_model('cnn_bilstm_model.h5')
model3 = load_model('E:\\cnn_bigru\\s_cnn_bigru_model.h5', compile=False)
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model4 = load_model('cnn_gru_model.h5', compile=False)
model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model5 =load_model('cnn_model.h5')
# 对测试集进行预测
y_pred1 = model1.predict(x_test_1)
y_pred2 = model2.predict(x_test_1)
y_pred3 = model3.predict(x_test_1)
y_pred4 = model4.predict(x_test_1)
y_pred5 = model5.predict(x_test_1)
# 将预测结果转换为数字标签
y_pred1 = np.argmax(y_pred1, axis=1)
y_pred2 = np.argmax(y_pred2, axis=1)
y_pred3 = np.argmax(y_pred3, axis=1)
y_pred4 = np.argmax(y_pred4, axis=1)
y_pred5 = np.argmax(y_pred5, axis=1)
# 保存预测结果为npy文件
# 导入相关库
# 读取数据集和模型预测结果
test_data = pd.read_csv('E:\\cnn_bigru\\NSL-KDD\\KDDTest+.csv')
y_true = test_data.iloc[:, -2].values  # 真实标签，即攻击类型

le = LabelEncoder()

# 对真实标签进行编码，将字符串转换为数字
y_true = le.fit_transform(y_true)

# 对预测标签进行编码，将字符串转换为数字
y_pred1 = le.fit_transform(y_pred1)
y_pred2 = le.fit_transform(y_pred2)
y_pred3 = le.fit_transform(y_pred3)
y_pred4 = le.fit_transform(y_pred4)
y_pred5 = le.fit_transform(y_pred5)




def calculate_situation_value(y_true, y_pred, N, M):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    # 打印分类报告，查看混淆矩阵的标签顺序
    print(classification_report(y_true, y_pred))
    # 计算各类攻击的威胁程度
    T = [0.6, 0.3, 0.5, 0.7, 0.05]
    # 计算各类攻击的脆弱性程度
    V = [0.6, 0.3, 0.5, 0.7, 0.05]
    # 计算各类攻击的重要性程度
    I = [0.6, 0.3, 0.5, 0.7, 0.05]
    # 初始化网络安全态势值
    S = 0
    # 遍历各类攻击，累加计算网络安全态势值
    for i in range(5):
        # 计算第i类攻击的数量
        n_i = np.sum(cm[i, :])
        # 判断n_i是否为0
        if n_i == 0:
            # 如果是，则跳过计算
            continue
        else:
            # 如果不是，则按照原来的公式计算第i类攻击的检测率
            r_i = cm[i, i] / n_i
            # 计算第i类攻击对网络安全态势值的贡献
            s_i = (T[i] * V[i] * I[i] * (1 - r_i)) / M
            # 累加计算网络安全态势值
            S += s_i * n_i / N
    # 返回网络安全态势值
    return S


# 假设网络中的资产数量为10000，归一化因子为10
N = 10000
M = 0.0005
# 将测试集分成若干段，这里假设分成10段，每段包含225个样本（测试集共有2254个样本）
n_segments = 50  # 分段数目
n_samples = 450  # 每段样本数

# 初始化四个空列表，用于存储每段的真实态势值和预测态势值
true_situation_values = []
predicted1_situation_values = []
predicted2_situation_values = []
predicted3_situation_values = []
predicted4_situation_values = []
predicted5_situation_values = []
# 遍历每一段，计算其真实态势值和预测态势值，并添加到列表中
for i in range(n_segments):
    # 获取第i段的真实标签和预测标签
    y_true_segment = y_true[i * n_samples:(i + 1) * n_samples]
    y_pred1_segment = y_pred1[i * n_samples:(i + 1) * n_samples]
    y_pred2_segment = y_pred2[i * n_samples:(i + 1) * n_samples]
    y_pred3_segment = y_pred3[i * n_samples:(i + 1) * n_samples]
    y_pred4_segment = y_pred4[i * n_samples:(i + 1) * n_samples]
    y_pred5_segment = y_pred5[i * n_samples:(i + 1) * n_samples]

    # 调用函数，计算第i段的预测态势值，并添加到列表中
    predicted1_situation_value_segment = calculate_situation_value(y_true_segment, y_pred1_segment, N,
                                                                   M)  # 注意这里传入的是真实标签和预测标签
    predicted1_situation_values.append(predicted1_situation_value_segment)
    predicted2_situation_value_segment = calculate_situation_value(y_true_segment, y_pred2_segment, N,
                                                                   M)  # 注意这里传入的是真实标签和预测标签
    predicted2_situation_values.append(predicted2_situation_value_segment)
    predicted3_situation_value_segment = calculate_situation_value(y_true_segment, y_pred3_segment, N,
                                                                   M)  # 注意这里传入的是真实标签和预测标签
    predicted3_situation_values.append(predicted3_situation_value_segment)
    predicted4_situation_value_segment = calculate_situation_value(y_true_segment, y_pred4_segment, N,
                                                                   M)  # 注意这里传入的是真实标签和预测标签
    predicted4_situation_values.append(predicted4_situation_value_segment)
    predicted5_situation_value_segment = calculate_situation_value(y_true_segment, y_pred5_segment, N,
                                                                   M)  # 注意这里传入的是真实标签和预测标签
    predicted5_situation_values.append(predicted5_situation_value_segment)

# 将列表转换为数组，并打印出来
true_situation_values = np.array(true_situation_values)

# 画出折线图，横轴为分段序号，纵轴为网络安全态势值

plt.plot(range(1, n_segments + 1), predicted1_situation_values, 'b-',
         label='n_Cnn_Bigru Predicted situation values for each segment')
plt.plot(range(1, n_segments + 1), predicted2_situation_values, 'g-',
         label='Cnn_lstm Predicted situation values for each segment')
plt.plot(range(1, n_segments + 1), predicted3_situation_values, 'p-',
         label='s_Cnn_Bigru Predicted situation values for each segment')
plt.plot(range(1, n_segments + 1), predicted4_situation_values, 'o-',
         label='Cnn_Gru Predicted situation values for each segment')
plt.plot(range(1, n_segments + 1), predicted5_situation_values, 'y-',
         label='Cnn Predicted situation values for each segment')
plt.xlabel('Segment number')
plt.ylabel('Situation value')
plt.legend()
plt.show()
