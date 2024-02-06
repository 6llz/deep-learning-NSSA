# Any results you write to the current directory are saved as output.
import matplotlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape
from tensorflow.keras import regularizers
import pickle
from tensorflow.keras import losses, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Loading training set into dataframe
import mTrain
import dpre
import keras_tuner as kt

matplotlib.use('TkAgg')
x_train_1, y_train_1, x_test_1, y_test_1 = dpre.dps()


def build_model(hp):
    cnn_bigru = Sequential()
    # 添加一个Reshape层 # 修改输出形状参数为(122 * hp_filters, 1)
    hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
    cnn_bigru.add(Reshape((122 * hp_filters, 1), input_shape=(122, 1)))
    hp_kernel_size = hp.Int('kernel_size', min_value=3, max_value=15, step=2)
    hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
    cnn_bigru.add(Convolution1D(hp_filters, kernel_size=hp_kernel_size, padding="same", activation=hp_activation))
    cnn_bigru.add(MaxPooling1D(pool_size=5, data_format="channels_first"))
    cnn_bigru.add(BatchNormalization())
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    cnn_bigru.add(Bidirectional(GRU(hp_units_1, return_sequences=True)))
    cnn_bigru.add(Reshape((-1, 1), input_shape=(2 * hp_units_1,)))
    cnn_bigru.add(MaxPooling1D(pool_size=5, data_format="channels_first"))
    cnn_bigru.add(BatchNormalization())
    hp_units_2 = hp.Int('units_2', min_value=64, max_value=256, step=64)
    cnn_bigru.add(Bidirectional(GRU(hp_units_2, return_sequences=True)))
    hp_units_3 = hp.Int('units_3', min_value=16, max_value=64, step=16)
    hp_activation_3 = hp.Choice('activation_3', values=['relu', 'tanh', 'sigmoid'])
    hp_l1 = hp.Float('l1', min_value=0.0001, max_value=0.01, sampling='log')
    hp_l2 = hp.Float('l2', min_value=0.0001, max_value=0.01, sampling='log')
    cnn_bigru.add(
        Dense(hp_units_3, activation=hp_activation_3, kernel_regularizer=regularizers.l1_l2(l1=hp_l1, l2=hp_l2)))
    cnn_bigru.add(Dropout(0.3))
    hp_units_4 = hp.Int('units_4', min_value=16, max_value=64, step=16)
    hp_activation_4 = hp.Choice('activation_4', values=['relu', 'tanh', 'sigmoid'])
    cnn_bigru.add(
        Dense(hp_units_4, activation=hp_activation_4, kernel_regularizer=regularizers.l1_l2(l1=hp_l1, l2=hp_l2)))
    cnn_bigru.add(Dropout(0.3))
    cnn_bigru.add(Dense(5, activation="softmax"))
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    loss = losses.CategoricalCrossentropy(from_logits=False)
    hp_learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='log')
    cnn_bigru.compile(loss=loss, optimizer=hp_optimizer(learning_rate=hp_learning_rate), metrics=['accuracy'])
    return cnn_bigru


import tensorboard as tb

tb_callback = tf.keras.callbacks.TensorBoard(log_dir="E:/cnn_bigru/logs")

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy', max_epochs=50, directory='E:/', project_name='cnn_bigru')

tuner.search(x_train_1, y_train_1, epochs=50, validation_data=(x_test_1, y_test_1), callbacks=[rlr, tb_callback])

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

tuner.results_summary()
