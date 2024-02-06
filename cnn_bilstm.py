import matplotlib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape, \
    GlobalAveragePooling1D, Flatten
import pickle
import dpre

# Loading training set into dataframe
import mTrain

matplotlib.use('TkAgg')
x_train_1, y_train_1, x_test_1, y_test_1 = dpre.dps()
batch_size = 32
cnn_bilstm = Sequential()
cnn_bilstm.add(Convolution1D(64, kernel_size=122, padding="same", activation="relu", input_shape=(122, 1)))
cnn_bilstm.add(MaxPooling1D(pool_size=5))
cnn_bilstm.add(BatchNormalization())
cnn_bilstm.add(Bidirectional(LSTM(64, return_sequences=False)))
cnn_bilstm.add(Reshape((128, 1), input_shape=(128,)))
cnn_bilstm.add(MaxPooling1D(pool_size=5))
cnn_bilstm.add(BatchNormalization())
cnn_bilstm.add(Bidirectional(LSTM(128, return_sequences=False)))
cnn_bilstm.add(Dense(48, activation='relu'))
cnn_bilstm.add(Dropout(0.4))
cnn_bilstm.add(Dense(48, activation='relu'))
cnn_bilstm.add(Dropout(0.4))
cnn_bilstm.add(Dense(5, activation="softmax"))
cnn_bilstm.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
cnn_bilstm.summary()
type = "cnn_bilstm"
history = cnn_bilstm.fit(x_train_1, y_train_1, batch_size=32, validation_data=(x_test_1, y_test_1),
                         epochs=10)
cnn_bilstm.save('cnn_bilstm_model.h5')
with open('cnn_bilstm_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
cm, pred1, y_eval, classes_ = mTrain.eva_p(history, cnn_bilstm, x_test_1, y_test_1, type)
mTrain.RoC_Curve(pred1, y_eval, classes_, title=' 0:Dos  1:normal  2:Probe  3:R2L  4:U2L \n\n ROC for CNN-BiLSTM')
mTrain.eva_d(cm)
