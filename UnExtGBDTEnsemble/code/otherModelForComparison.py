# /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ =  'chenqiwei'

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics
import random


def load_data(file):
    """ 加载数据 """
    data = []  # 数据的特征标签
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            content = line.strip().split(',')
            len_line = len(content)
            item = []
            for i in range(len_line):
                item.append(float(content[i]))
            data.append(item)
    return data


def seprate_xy(data):
    """ 将数据的特征和类标签拆分,分别返回 """
    x = []
    y = []
    len_line = len(data[0])
    for i in range(len(data)):
        x.append(data[i][:len_line - 1])
        y.append(data[i][len_line - 1])
    return np.array(x), np.array(y)


def sub_sample(train_x, train_y):
    """ 采样训练子集 """
    num = len(train_x)
    id = random.sample(range(num), int(num * 0.7))
    sample_train_x = [train_x[i] for i in id]
    sample_train_y = [train_y[i] for i in id]
    return sample_train_x, sample_train_y


def LR(train_x, train_y, test_x, test_y):
    """ 逻辑回归模型 """
    classifier = LogisticRegression(tol=0.01, max_iter=200)
    classifier.fit(train_x, train_y)
    pred = classifier.predict_proba(test_x)
    predict_pro = []
    for pro in pred:
        predict_pro.append(pro[1])
    predict_y = classifier.predict(test_x)
    auc = evaluate_auc(predict_pro, test_y)
    evaluate(predict_y, test_y)
    return auc



def sub_LR(train_x, train_y, test_x, test_y):
    """ 逻辑回归子模型 """
    classifier = LogisticRegression(tol=0.01, max_iter=200)
    classifier.fit(train_x, train_y)
    pred = classifier.predict_proba(test_x)
    predict_y_probability = []
    for i in range(len(pred)):
        predict_y_probability.append(pred[i][1])
    return np.array(predict_y_probability)


def NB(train_x, train_y, test_x, test_y):
    """ 朴素贝叶斯 """
    classifier = GaussianNB()
    classifier.fit(train_x, train_y)
    pred = classifier.predict_proba(test_x)
    predict_pro = []
    for pro in pred:
        predict_pro.append(pro[1])
    predict_y = classifier.predict(test_x)
    auc = evaluate_auc(predict_pro, test_y)
    evaluate(predict_y, test_y)
    return auc


def sub_NB(train_x, train_y, test_x, test_y):
    """ 朴素贝叶斯子模型 """
    classifier = GaussianNB()
    classifier.fit(train_x, train_y)
    pred = classifier.predict_proba(test_x)
    predict_pro = []
    for pro in pred:
        predict_pro.append(pro[1])
    return np.array(predict_pro)


def RF(train_x, train_y, test_x, test_y):
    """ 随机森林 """
    classifier = RandomForestClassifier()
    classifier.fit(train_x, train_y)
    pred = classifier.predict_proba(test_x)
    predict_pro = []
    for pro in pred:
        predict_pro.append(pro[1])
    predict_y = classifier.predict(test_x)
    auc = evaluate_auc(predict_pro, test_y)
    evaluate(predict_y, test_y)
    return auc


def sub_RF(train_x, train_y, test_x, test_y):
    """ 随机森林子模型 """
    classifier = RandomForestClassifier()
    classifier.fit(train_x, train_y)
    pred = classifier.predict_proba(test_x)
    predict_pro = []
    for pro in pred:
        predict_pro.append(pro[1])
    return np.array(predict_pro)


def DT(train_x, train_y, test_x, test_y):
    """ 决策树 """
    classifier = DecisionTreeClassifier()
    classifier.fit(train_x, train_y)
    pred = classifier.predict_proba(test_x)
    pred = np.asarray(np.mat(pred)[:, 1])
    predict_y = classifier.predict(test_x)
    auc = evaluate_auc(pred, test_y)
    evaluate(predict_y, test_y)
    return auc


def sub_DT(train_x, train_y, test_x, test_y):
    """ 决策树子模型 """
    classifier = DecisionTreeClassifier()
    classifier.fit(train_x, train_y)
    pred = classifier.predict_proba(test_x)
    predict_y_probability = []
    for i in range(len(pred)):
        predict_y_probability.append(pred[i][1])
    return np.array(predict_y_probability)


def SVM(train_x, train_y, test_x, test_y):
    """ SVM """
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(train_x)
    train_x = min_max_scaler.transform(train_x)
    test_x = min_max_scaler.transform(test_x)
    classifier = svm.LinearSVC()
    classifier.fit(train_x, train_y)
    predict_y = classifier.predict(test_x)
    evaluate(predict_y, test_y)


def sub_SVM(train_x, train_y, test_x, test_y):
    """ SVM子模型 """
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(train_x)
    train_x = min_max_scaler.transform(train_x)
    test_x = min_max_scaler.transform(test_x)
    classifier = svm.LinearSVC()
    classifier.fit(train_x, train_y)
    predict_y = classifier.predict(test_x)
    return np.array(predict_y)


def ExtGBDT(train_x, train_y, test_x, test_y):
    """ Ext-GBDT """
    num_round = 100
    param = {'objective': 'binary:logistic', 'booster': 'gbtree', 'eta': 0.03, 'max_depth': 3, 'eval_metric': 'auc',
             'silent': 1, 'min_child_weight': 0.1, 'subsample': 0.7, 'colsample_bytree': 0.8, 'nthread': 4,
             'max_delta_step': 0}
    train_X = xgb.DMatrix(train_x, train_y)
    test_X = xgb.DMatrix(test_x)
    bst = xgb.train(param, train_X, num_round)
    pred = bst.predict(test_X)
    predict_y = []
    for i in range(len(pred)):
        if pred[i] < 0.5:
            predict_y.append(0)
        else:
            predict_y.append(1)
    auc = evaluate_auc(pred, test_y)
    evaluate(predict_y, test_y)
    return auc


def subExtGBDT(train_x, train_y, test_x, test_y):
    """ Ext-GBDT子模型 """
    num_round = 100
    param = {'objective': 'binary:logistic', 'booster': 'gbtree', 'eta': 0.03, 'max_depth': 3, 'eval_metric': 'auc',
             'silent': 1, 'min_child_weight': 0.1, 'subsample': 0.7, 'colsample_bytree': 0.8, 'nthread': 4,}
    train_X = xgb.DMatrix(train_x, train_y)
    test_X = xgb.DMatrix(test_x)
    bst = xgb.train(param, train_X, num_round)
    pred = bst.predict(test_X)
    return np.array(pred)


def DTEnsemble(train_x, train_y, test_x, test_y):
    """ 决策树 集成  """
    total = np.zeros(len(test_y))
    sub_num = 10
    for i in range(sub_num):
        sub_train_x, sub_train_y = sub_sample(train_x, train_y)
        pred = sub_DT(sub_train_x, sub_train_y, test_x, test_y)
        total += pred
    avg_pred = total / sub_num
    avg_predict = []
    for i in range(len(avg_pred)):
        if avg_pred[i] < 0.5:
            avg_predict.append(0)
        else:
            avg_predict.append(1)
    auc = evaluate_auc(avg_pred, test_y)
    evaluate(avg_predict, test_y)
    return auc


def LREnsemble(train_x, train_y, test_x, test_y):
    """ 逻辑回归 集成 """
    total = np.zeros(len(test_y))
    sub_num = 10
    for i in range(sub_num):
        sub_train_x, sub_train_y = sub_sample(train_x, train_y)
        pred = sub_LR(sub_train_x, sub_train_y, test_x, test_y)
        total += pred
    avg_pred = total / sub_num
    avg_predict = []
    for i in range(len(avg_pred)):
        if avg_pred[i] < 0.5:
            avg_predict.append(0)
        else:
            avg_predict.append(1)
    auc = evaluate_auc(avg_pred, test_y)
    evaluate(avg_predict, test_y)
    return auc


def NBEnsemble(train_x, train_y, test_x, test_y):
    """ 朴素贝叶斯 集成  """
    total = np.zeros(len(test_y))
    sub_num = 10
    for i in range(sub_num):
        sub_train_x, sub_train_y = sub_sample(train_x, train_y)
        pred = sub_NB(sub_train_x, sub_train_y, test_x, test_y)
        total += pred
    avg_pred = total / sub_num
    avg_predict = []
    for i in range(len(avg_pred)):
        if avg_pred[i] < 0.5:
            avg_predict.append(0)
        else:
            avg_predict.append(1)
    auc = evaluate_auc(avg_pred, test_y)
    evaluate(avg_predict, test_y)
    return auc


def SVMEnsemble(train_x, train_y, test_x, test_y):
    """ SVM 集成 """
    total = np.zeros(len(test_y))
    sub_num = 100
    for i in range(100):
        sub_train_x, sub_train_y = sub_sample(train_x, train_y)
        pred = sub_SVM(sub_train_x, sub_train_y, test_x, test_y)
        total += pred
    avg_predict = []
    for i in range(len(total)):
        if total[i] < 50:
            avg_predict.append(0)
        else:
            avg_predict.append(1)
    evaluate(avg_predict, test_y)


def RFEnsemble(train_x, train_y, test_x, test_y):
    """ 随机森林 集成 """
    total = np.zeros(len(test_y))
    sub_num = 10
    for i in range(sub_num):
        sub_train_x, sub_train_y = sub_sample(train_x, train_y)
        pred = sub_RF(sub_train_x, sub_train_y, test_x, test_y)
        total += pred
    avg_pred = total / sub_num
    avg_predict = []
    for i in range(len(avg_pred)):
        if avg_pred[i] < 0.5:
            avg_predict.append(0)
        else:
            avg_predict.append(1)
    auc = evaluate_auc(avg_pred, test_y)
    evaluate(avg_predict, test_y)
    return auc


def ExtGBDTEnsemble(train_x, train_y, test_x, test_y):
    """ Ext-GBDT 集成 """
    total = np.zeros(len(test_y))
    sub_num = 10
    for i in range(sub_num):
        sub_train_x, sub_train_y = sub_sample(train_x, train_y)
        pred = subExtGBDT(train_x, train_y, test_x, test_y)
        total += pred
    avg_pred = total / sub_num
    avg_predict = []
    for i in range(len(avg_pred)):
        if avg_pred[i] < 0.5:
            avg_predict.append(0)
        else:
            avg_predict.append(1)
    auc = evaluate_auc(avg_pred, test_y)
    evaluate(avg_predict, test_y)
    return auc


def evaluate(predict_y, true_y):
    """ 输出代价敏感矩阵 """
    TP, TN, FP, FN = 0, 0, 0, 0
    len_num = len(true_y)
    for i in xrange(len_num):
        if int(true_y[i]) == 1 and int(predict_y[i]) == 1:
            TP += 1
        elif int(true_y[i]) == 1 and int(predict_y[i]) == 0:
            FN += 1
        elif int(true_y[i]) == 0 and int(predict_y[i]) == 1:
            FP += 1
        else:
            TN += 1
    print "代价敏感矩阵: "
    print "----------------------------------------"
    print "         | Predict good | Predict bad   "
    print "----------------------------------------"
    print "True good|              | {0} * cost_01 ".format(FN)
    print "----------------------------------------"
    print "True bad |{0} * cost_10 |               ".format(FP)


def evaluate_auc(pred, y):
    """ AUC """
    fpr, tpr, thredholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print "Auc : ", auc
    return auc


def get_data_set(fold_id):
    file_good = os.listdir('../data/fold5_data/good')
    for i in range(len(file_good)):
        file_good[i] = '../data/fold5_data/good/' + file_good[i]
    file_bad = os.listdir('../data/fold5_data/bad')
    for i in range(len(file_bad)):
        file_bad[i] = '../data/fold5_data/bad/' + file_bad[i]
    # i = 0,1,2,3,4是5折交叉验证,分别取4份数据作为训练数据,另外一份没有被训练过的数据作为验证数据
    # i = 5是训练最终的模型,用于未来预测新的样本,因此使用全部训练数据,未来预测的样本还未知,没有在模型中被训练过.
    if fold_id == 0:
        train_good = load_data(file_good[1]) + load_data(file_good[2]) + load_data(file_good[3]) + load_data(file_good[4])
        train_bad = load_data(file_bad[1]) + load_data(file_bad[2]) + load_data(file_bad[3]) + load_data(file_bad[4])
        test = load_data(file_good[0]) + load_data(file_bad[0])
    elif fold_id == 1:
        train_good = load_data(file_good[0]) + load_data(file_good[2]) + load_data(file_good[3]) + load_data(file_good[4])
        train_bad = load_data(file_bad[0]) + load_data(file_bad[2]) + load_data(file_bad[3]) + load_data(file_bad[4])
        test = load_data(file_good[1]) + load_data(file_bad[1])
    elif fold_id == 2:
        train_good = load_data(file_good[0]) + load_data(file_good[1]) + load_data(file_good[3]) + load_data(file_good[4])
        train_bad = load_data(file_bad[0]) + load_data(file_bad[1]) + load_data(file_bad[3]) + load_data(file_bad[4])
        test = load_data(file_good[2]) + load_data(file_bad[2])
    elif fold_id == 3:
        train_good = load_data(file_good[0]) + load_data(file_good[1]) + load_data(file_good[2]) + load_data(file_good[4])
        train_bad = load_data(file_bad[0]) + load_data(file_bad[1]) + load_data(file_bad[2]) + load_data(file_bad[4])
        test = load_data(file_good[3]) + load_data(file_bad[3])
    elif fold_id == 4:
        train_good = load_data(file_good[0]) + load_data(file_good[1]) + load_data(file_good[2]) + load_data(file_good[3])
        train_bad = load_data(file_bad[0]) + load_data(file_bad[1]) + load_data(file_bad[2]) + load_data(file_bad[3])
        test = load_data(file_good[4]) + load_data(file_bad[4])
    elif fold_id == 5:
        train_good = load_data(file_good[0]) + load_data(file_good[1]) + load_data(file_good[2]) + load_data(file_good[3]) + load_data(file_good[4])
        train_bad = load_data(file_bad[0]) + load_data(file_bad[1]) + load_data(file_bad[2]) + load_data(file_bad[3]) + load_data(file_bad[4])
        return train_good, train_bad
    return train_good + train_bad, test


def main():
    print "功能: "
    print "[1] 决策树"
    print "[2] 逻辑回归"
    print "[3] 朴素贝叶斯"
    print "[4] 支持向量机"
    print "[5] 随机森林"
    print "[6] Ext-GBDT"
    print "[7] 决策树 集成"
    print "[8] 逻辑回归 集成"
    print "[9] 朴素贝叶斯 集成"
    print "[10] 支持向量机 集成"
    print "[11] 随机森林 集成"
    print "[12] Ext-GBDT 集成"
    flag = raw_input("请选择:")
    auc = 0
    for i in range(5):
        print '\n---------- Fold {0} 模型预测结果 ----------'.format(i + 1)
        train, test = get_data_set(i)
        random.shuffle(train)
        random.shuffle(test)
        train_x, train_y = seprate_xy(train)
        test_x, test_y = seprate_xy(test)
        if str(flag) == '1':
            auc += DT(train_x, train_y, test_x, test_y)
        elif str(flag) == '2':
            auc += LR(train_x, train_y, test_x, test_y)
        elif str(flag) == '3':
            auc += NB(train_x, train_y, test_x, test_y)
        elif str(flag) == '4':
            SVM(train_x, train_y, test_x, test_y)
        elif str(flag) == '5':
            auc += RF(train_x, train_y, test_x, test_y)
        elif str(flag) == '6':
            auc += ExtGBDT(train_x, train_y, test_x, test_y)
        elif str(flag) == '7':
            auc += DTEnsemble(train_x, train_y, test_x, test_y)
        elif str(flag) == '8':
            auc += LREnsemble(train_x, train_y, test_x, test_y)
        elif str(flag) == '9':
            auc += NBEnsemble(train_x, train_y, test_x, test_y)
        elif str(flag) == '10':
            SVMEnsemble(train_x, train_y, test_x, test_y)
        elif str(flag) == '11':
            auc += RFEnsemble(train_x, train_y, test_x, test_y)
        elif str(flag) == '12':
            auc += ExtGBDTEnsemble(train_x, train_y, test_x, test_y)
        else:
            print "输入错误..."
            break
    if str(flag) == '4' or str(flag) == '10':
        print
    else:
        print "\n\n平均AUC: ", auc / 5


if __name__ == "__main__":
    main()
