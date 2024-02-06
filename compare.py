import pickle

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
with open('E:\\cnn_bigru\\cnn_history.pkl', 'rb') as file:
    history1 = pickle.load(file)
with open('E:\\cnn_bigru\\n_cnn_bigru_history.pkl', 'rb') as file:
    history2 = pickle.load(file)
with open('E:\\cnn_bigru\\cnn_gru_history.pkl', 'rb') as file:
    history3 = pickle.load(file)
with open('E:\\cnn_bigru\\cnn_lstm_history.pkl', 'rb') as file:
    history4 = pickle.load(file)
plt.figure(figsize=(15, 4))


plt.plot(history1['val_accuracy'], label="VALIDATION ACCURACY OF MODEL 1")
plt.plot(history2['val_accuracy'], label="VALIDATION ACCURACY OF MODEL 2")
plt.plot(history3['val_accuracy'], label="VALIDATION ACCURACY OF MODEL 3")
plt.plot(history4['val_accuracy'], label="VALIDATION ACCURACY OF MODEL 4")

plt.title("TRAINING ACCURACY vs VALIDATION ACCURACY OF TWO MODELS")
plt.xlabel("EPOCH'S")
plt.ylabel("ACCURACY")
plt.legend(loc="best")
plt.show()
