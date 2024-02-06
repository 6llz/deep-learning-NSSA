import matplotlib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape, \
    GlobalAveragePooling1D, Flatten
import pickle
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import dpre
import mTrain

matplotlib.use('TkAgg')
x_train_1, y_train_1, x_test_1, y_test_1 = dpre.dps()
cnn = Sequential()
cnn.add(Convolution1D(64, 3, padding="same", activation="relu", input_shape=(122, 1)))
cnn.add(Convolution1D(64, 3, padding="same", activation="relu"))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(64, activation="relu"))
cnn.add(Dropout(0.4))  # 0.2
cnn.add(Dense(64, activation="relu"))
cnn.add(Dropout(0.4))
cnn.add(Dense(48, activation="relu"))  # added
cnn.add(Dropout(0.4))
cnn.add(Dense(48, activation="relu"))  # added 2
cnn.add(Dropout(0.4))
cnn.add(Dense(5, activation="softmax"))
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
cnn.summary()
type = "cnn"
history = cnn.fit(x_train_1, y_train_1, batch_size=32, validation_data=(x_test_1, y_test_1),
                        epochs=10)
# Save the model
cnn.save('cnn_model.h5')
with open(f'{type}_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
cm, pred1, y_eval, classes_ = mTrain.eva_p(history, cnn, x_test_1, y_test_1,type)
mTrain.RoC_Curve(pred1, y_eval, classes_, title='0:Dos  1:normal  2:Probe  3:R2L  4:U2L \n\n ROC for CNN')
mTrain.eva_d(cm)
