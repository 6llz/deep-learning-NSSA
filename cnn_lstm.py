import matplotlib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape, \
    GlobalAveragePooling1D, Flatten
import pickle
import dpre
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
# Loading training set into dataframe
import mTrain

matplotlib.use('TkAgg')
x_train_1, y_train_1, x_test_1, y_test_1 = dpre.dps()
cnn_lstm = Sequential()
cnn_lstm.add(Convolution1D(64, 3, padding="same", activation="relu", input_shape=(122, 1)))
cnn_lstm.add(Convolution1D(64, 3, padding="same", activation="relu"))
cnn_lstm.add(MaxPooling1D(pool_size=2))
cnn_lstm.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn_lstm.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn_lstm.add(MaxPooling1D(pool_size=2))
cnn_lstm.add(LSTM(64, return_sequences=True))
cnn_lstm.add(Dropout(0.4))
cnn_lstm.add(LSTM(64, return_sequences=False))
cnn_lstm.add(Dropout(0.4))
cnn_lstm.add(Dense(48, activation='relu'))
cnn_lstm.add(Dropout(0.4))
cnn_lstm.add(Dense(48, activation='relu'))
cnn_lstm.add(Dropout(0.4))
cnn_lstm.add(Dense(5, activation='softmax'))
cnn_lstm.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
cnn_lstm.summary()
type = "cnn_lstm"
history = cnn_lstm.fit(x_train_1, y_train_1, batch_size=32, validation_data=(x_test_1, y_test_1),
                       epochs=30)
cnn_lstm.save('cnn_lstm_model.h5')
with open(f'{type}_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
cm, pred1, y_eval, classes_ = mTrain.eva_p(history, cnn_lstm, x_test_1, y_test_1, type)
mTrain.RoC_Curve(pred1, y_eval, classes_, title=' 0:Dos  1:normal  2:Probe  3:R2L  4:U2L \n\n ROC for CNN-LSTM')
mTrain.eva_d(cm)
