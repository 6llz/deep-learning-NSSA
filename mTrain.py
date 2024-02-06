import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from itertools import cycle
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, \
    classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, auc
from sklearn import preprocessing, metrics
import pickle

from sklearn.preprocessing import MinMaxScaler


def eva_p(history, my_modle, x_test_1, y_test_1, type):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="TRAINING LOSS")
    plt.plot(history.history['val_loss'], label="VALIDATION LOSS")
    plt.title("TRAINING LOSS vs VALIDATION LOSS")
    plt.xlabel("EPOCH'S")
    plt.ylabel("TRAINING LOSS vs VALIDATION LOSS")
    plt.legend(loc="best")
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label="TRAINING ACCURACY")
    plt.plot(history.history['val_accuracy'], label="VALIDATION ACCURACY")
    plt.title("TRAINING ACCURACY vs VALIDATION ACCURACY")
    plt.xlabel("EPOCH'S")
    plt.ylabel("TRAINING ACC vs VALIDATION ACCURACY")
    plt.legend(loc="best")
    # 调整子图间距
    plt.tight_layout()
    le = preprocessing.LabelEncoder()
    pred1 = my_modle.predict(x_test_1)
    pred2 = np.argmax(pred1, axis=1)
    pred = le.fit_transform(pred2)
    # pred = le.inverse_transform(pred)
    y_eval = np.argmax(y_test_1, axis=1)
    score = metrics.accuracy_score(y_eval, pred)
    print(f"Validation score: {score * 100:.2f}%")  # 使用f-string格式化输出
    acc = accuracy_score(y_eval, pred)
    print(f"accuracy : {acc:.4f}")  # 使用f-string格式化输出
    recall = recall_score(y_eval, pred, average=None)
    print(f"recall : {recall}")  # 使用f-string格式化输出
    precision = precision_score(y_eval, pred, average=None)
    print(f"precision : {precision}")  # 使用f-string格式化输出
    f1_scr = f1_score(y_eval, pred, average=None)
    print(f"f1_score : {f1_scr}")  # 使用f-string格式化输出
    # 0:Dos 1:normal 2:Probe 3:R2L 4:U2L
    print(f"#### 0:Dos 1:normal 2:Probe 3:R2L 4:U2L ###\n\n")  # 使用f-string格式化输出
    print(classification_report(pred, y_eval))
    cm = confusion_matrix(y_eval, pred2)
    print(cm, '\n')
    fig, ax = plt.subplots(figsize=(8, 8))
    # 指定类别名称
    le.classes_ = np.array(['Dos', 'normal', 'Probe', 'R2L', 'U2L'])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(ax=ax)
    # 设置x轴的刻度和标签，注意要与类别数一致
    ax.set_xticks(np.arange(len(le.classes_)))
    ax.set_xticklabels(le.classes_)
    # 设置y轴的刻度和标签，注意要与类别数一致
    ax.set_yticks(np.arange(len(le.classes_)))
    ax.set_yticklabels(le.classes_)
    # 使用f-string格式化标题
    plt.title(f'0:Dos 1:normal 2:Probe 3:R2L 4:U2L \n\n Confusion Marix of proposed {type} model')
    # 调整子图间距
    plt.tight_layout()
    # 显示图形
    plt.show()
    return cm, pred1, y_eval, le.classes_


def RoC_Curve(y_score, y, labels, title):
    y_cat = to_categorical(y)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2
    # First aggregate all false positive rates
    n_classes = len(labels)
    print('n_classes:', n_classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_cat[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_cat.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:0.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right")
    plt.show()


def eva_d(cm):
    """
    This function calculates and prints some evaluation metrics from a confusion matrix.
    :param cm: a numpy array of shape (n, n), where n is the number of classes.
    :return: None
    """
    # Calculate true positive, true negative, false positive and false negative
    TP = np.diag(cm)  # 对角线元素为真正例
    TN = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + TP  # 其他元素之和减去假正例和假反例为真反例
    FP = np.sum(cm, axis=0) - TP  # 列元素之和减去真正例为假正例
    FN = np.sum(cm, axis=1) - TP  # 行元素之和减去真正例为假反例
    print(f"TP = {TP}")
    print(f"TN = {TN}")
    print(f"FP = {FP}")
    print(f"FN = {FN}\n")

    # Calculate true positive rate, false positive rate, true negative rate and false negative rate
    TPR = TP / (TP + FN)  # 真正例率等于真正例除以真正例加假反例
    FPR = FP / (FP + TN)  # 假正例率等于假正例除以假正例加真反例
    TNR = TN / (TN + FP)  # 真反例率等于真反例除以真反例加假正例
    FNR = FN / (TP + FN)  # 假反例率等于假反例除以真正例加假反例
    print(f"TPR = {TPR} True positive rate, Sensitivity, hit rate, or recall")
    print(f"FPR = {FPR} False positive rate or fall out")
    print(f"TNR = {TNR} True negative rate or specificity")
    print(f"FNR = {FNR} False negative rate\n")

    # Calculate positive predictive value, negative predictive value and false discovery rate
    PPV = TP / (TP + FP)  # 正确预测值等于真正例除以真正例加假正例
    NPV = TN / (TN + FN)  # 负预测值等于真反例除以真反例加假反例
    FDR = FP / (TP + FP)  # 假发现率等于假正例除以真正例加假正例
    print(f"PPV = {PPV} Positive predictive value or precision")
    print(f"NPV = {NPV} Negative predictive value")
    print(f"FDR = {FDR} False discovery rate\n")
