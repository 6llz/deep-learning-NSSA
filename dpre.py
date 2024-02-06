import matplotlib
import pandas as pd
import numpy as np

# Loading training set into dataframe
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')
train_df = pd.read_parquet('E:\\cnn_bigru\\NSL-KDD\\KDDTrain+.parquet')
# Loading testing set into dataframe
test_df = pd.read_parquet('E:\\cnn_bigru\\NSL-KDD\\KDDTest+.parquet')
# Reset column names for training set
train_df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                    'num_access_files', 'num_outbound_cmds', 'is_host_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate',
                    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                    'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                    'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']
train_df.head()
# Reset column names for testing set
test_df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                   'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                   'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                   'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                   'num_access_files', 'num_outbound_cmds', 'is_host_login',
                   'is_guest_login', 'count', 'srv_count', 'serror_rate',
                   'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                   'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                   'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                   'dst_host_same_src_port_rate',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                   'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                   'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']
test_df.head()
# Dropping the last columns of training set
train_df = train_df.drop('difficulty_level', axis=1)  # we don't need it in this project
train_df.shape
# Dropping the last columns of testing set
test_df = test_df.drop('difficulty_level', axis=1)
test_df.shape
# defining col list
cols = ['protocol_type', 'service', 'flag']


# 独热编码操作
def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)# 对每个元素对应的列，使用pandas的get_dummies函数生成独热编码的数据框，使用元素作为前缀，不删除第一列
        df = pd.concat([df, dummies], axis=1)# 将原始数据框和独热编码的数据框按列合并
        df = df.drop(each, axis=1)# 删除原始数据框中的元素对应的列
    return df


# 定义一个函数，接受一个数据框对象和一个列名列表作为参数
def normalize(df, cols):
    # 创建一个数据框的副本，用于存储处理后的结果
    result = df.copy()
    # 遍历列名列表中的每个元素
    for feature_name in cols:
        # 获取每个元素对应的列的最大值和最小值
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        # 如果最大值大于最小值，才进行归一化操作，否则不变
        if max_value > min_value:
            # 使用异或运算符（^）和除法运算符（/）对每个元素对应的列进行归一化，将值映射到0到1之间
            result[feature_name] = (df[feature_name].astype(bool) ^ np.full(df.shape[0], min_value).astype(
                bool)).astype(float) / (max_value.astype(bool) ^ min_value.astype(bool)).astype(float)
    # 返回处理后的数据框对象
    return result


def dps():
    train_df_1 = one_hot(train_df, cols)
    test_df_1 = one_hot(test_df, cols)

    # Dropping subclass column for training  and testing set
    tmp = train_df_1.pop('subclass')
    tmp1 = test_df_1.pop('subclass')
    # Normalizing training set
    train_df_2 = normalize(train_df_1, train_df_1.columns)
    train_df_2
    # Normalizing testing set
    test_df_2 = normalize(test_df_1, test_df_1.columns)
    test_df_2
    # Fixing labels for training set
    classlist_train = []
    check1_train = (
        "apache2", "back", "land", "neptune", "mailbomb", "pod", "processtable", "smurf", "teardrop", "udpstorm",
        "worm")
    check2_train = ("ipsweep", "mscan", "nmap", "portsweep", "saint", "satan")
    check3_train = ("buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm")
    check4_train = (
        "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "Snmpgetattack",
        "spy",
        "snmpguess", "warezclient", "warezmaster", "xlock", "xsnoop")

    DoSCount_train = 0
    ProbeCount_train = 0
    U2RCount_train = 0
    R2LCount_train = 0
    NormalCount_train = 0

    for item in tmp:
        if item in check1_train:
            classlist_train.append("DoS")
            DoSCount_train = DoSCount_train + 1
        elif item in check2_train:
            classlist_train.append("Probe")
            ProbeCount_train = ProbeCount_train + 1
        elif item in check3_train:
            classlist_train.append("U2R")
            U2RCount_train = U2RCount_train + 1
        elif item in check4_train:
            classlist_train.append("R2L")
            R2LCount_train = R2LCount_train + 1
        else:
            classlist_train.append("Normal")
            NormalCount_train = NormalCount_train + 1
    print(DoSCount_train)
    print(NormalCount_train)
    print(ProbeCount_train)
    print(R2LCount_train)
    print(U2RCount_train)
    # Fixing labels for testing set
    classlist_test = []
    check1_test = (
        "apache2", "back", "land", "neptune", "mailbomb", "pod", "processtable", "smurf", "teardrop", "udpstorm",
        "worm")
    check2_test = ("ipsweep", "mscan", "nmap", "portsweep", "saint", "satan")
    check3_test = ("buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm")
    check4_test = (
        "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "Snmpgetattack",
        "spy",
        "snmpguess", "warezclient", "warezmaster", "xlock", "xsnoop")

    DoSCount_test = 0
    ProbeCount_test = 0
    U2RCount_test = 0
    R2LCount_test = 0
    NormalCount_test = 0

    for item in tmp1:
        if item in check1_test:
            classlist_test.append("DoS")
            DoSCount_test = DoSCount_test + 1
        elif item in check2_test:
            classlist_test.append("Probe")
            ProbeCount_test = ProbeCount_test + 1
        elif item in check3_test:
            classlist_test.append("U2R")
            U2RCount_test = U2RCount_test + 1
        elif item in check4_test:
            classlist_test.append("R2L")
            R2LCount_test = R2LCount_test + 1
        else:
            classlist_test.append("Normal")
            NormalCount_test = NormalCount_test + 1
    print(DoSCount_test)
    print(NormalCount_test)
    print(ProbeCount_test)
    print(R2LCount_test)
    print(U2RCount_test)
    # Appending class column to training set
    train_df_2 = pd.concat([train_df_2, pd.Series(classlist_train, name='Class')], axis=1)
    test_df_2 = pd.concat([test_df_2, pd.Series(classlist_test, name='Class')], axis=1)
    # Appending class column to testing set
    y_train = train_df_2['Class']
    y_test = test_df_2['Class']
    X_train = train_df_2.drop('Class', axis=1)
    X_test = test_df_2.drop('Class', axis=1)
    X_train

    # Split data: 80% training and 20% testing

    train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=101)

    x_columns_train = train_df_2.columns.drop('Class')
    x_train_array = train_X[x_columns_train].values
    x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))
    dummies = pd.get_dummies(train_y)  # Classification
    '''outcomes = dummies.columns
    num_classes = len(outcomes)'''
    y_train_1 = dummies.values
    x_columns_test = test_df_2.columns.drop('Class')
    x_test_array = test_df_2[x_columns_test].values
    x_test_1 = np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))
    x_test_1 = pad_sequences(x_test_1, maxlen=122)
    dummies_test = pd.get_dummies(y_test)  # Classification
    '''outcomes_test = dummies_test.columns
    num_classes = len(outcomes_test)'''
    y_test_1 = dummies_test.values
    return x_train_1, y_train_1, x_test_1, y_test_1
