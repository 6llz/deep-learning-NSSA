# Any results you write to the current directory are saved as output.
import matplotlib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import GRU, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape
from keras import regularizers
import pickle
from keras import losses, optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Loading training set into dataframe
from keras.utils import plot_model

import mTrain
import dpre

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# 设置 TensorFlow 使用 GPU 进行计算
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

matplotlib.use('TkAgg')
x_train_1, y_train_1, x_test_1, y_test_1 = dpre.dps()

cnn_bigru = Sequential()
cnn_bigru.add(Convolution1D(64, kernel_size=2, padding="same", activation="relu", input_shape=(122, 1)))
cnn_bigru.add(MaxPooling1D(pool_size=5))
cnn_bigru.add(BatchNormalization())
cnn_bigru.add(Bidirectional(GRU(64, return_sequences=False)))
cnn_bigru.add(Reshape((128, 1), input_shape=(128,)))
cnn_bigru.add(MaxPooling1D(pool_size=5))
cnn_bigru.add(BatchNormalization())
cnn_bigru.add(Bidirectional(GRU(128, return_sequences=False)))
cnn_bigru.add(Dense(48, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
cnn_bigru.add(Dropout(0.5))
cnn_bigru.add(Dense(5, activation="softmax"))
# define optimizer and objective, compile gru
loss = losses.CategoricalCrossentropy(from_logits=False)
cnn_bigru.compile(loss=loss, optimizer=optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
#定义一个学习率衰减回调函数，当验证损失不再下降时，将学习率乘以0.3，最多等待2个周期，每次衰减后冷却1个周期，并打印信息
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, cooldown=1)
cnn_bigru.summary()
#使用训练集数据拟合模型，批量大小为32，验证集数据为测试集数据，训练10个周期，并使用学习率衰减回调函数
history = cnn_bigru.fit(x_train_1, y_train_1, batch_size=32, validation_data=(x_test_1, y_test_1),
                        epochs=30, callbacks=[rlr])
plot_model(cnn_bigru, show_shapes=True, to_file='n_cnn_bigru.png', rankdir='LR')
type = "n_cnn_bigru"
cnn_bigru.save('n_cnn_bigru_model.h5')
with open(f'{type}_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

cm, pred1, y_eval, classes_ = mTrain.eva_p(history, cnn_bigru, x_test_1, y_test_1, type)
mTrain.RoC_Curve(pred1, y_eval, classes_, title=' 0:Dos  1:normal  2:Probe  3:R2L  4:U2L \n\n ROC for CNN-BiGRU')
mTrain.eva_d(cm)

# Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
# print("ACC = ", ACC)
