# /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ =  'chenqiwei'

import os
import random
import numpy as np
import xgboost as xgb
from sklearn import metrics
from collections import Counter
import pickle


def load_data(file):
    """
    读取数据
    :param file: 数据文件
    :return: data: list,数据列表
    """
    data = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            lineArr = line.strip().split(',')
            len_line = len(lineArr)
            item = []
            for i in range(len_line):
                item.append(float(lineArr[i]))
            data.append(item)
    return data


def seprate_xy(data):
    """
    将数据的特征和类标签拆分,分别返回
    :param data: 数据(特征+类标签)
    :return: x: ndarray, 特征
             y: ndarray, 类标签
    """
    x = []
    y = []
    len_line = len(data[0])
    for i in range(len(data)):
        x.append(data[i][:len_line - 1])
        y.append(data[i][len_line - 1])
    return np.array(x), np.array(y)


def ExtGBDT(cross_validation, model_id, train_x, train_y, test_x, num_round=100, eta=0.03, max_depth=6, colsample_bytree=0.8):
    """
    训练ExtGBDT子模型:
    如果cross_validation = 0, 是训练模型,保存子模型的二进制文件到当前目录下的model目录下
    如果cross_validation = 1, 是训练模型+交叉验证,保存子模型到前面目录下的model目录下,并返回子模型对验证数据的预测结果
    :param cross_validation: num, 是否是交叉验证
    :param model_id: num, 子模型id
    :param train_x: ndarray, 训练数据的feature
    :param train_y: ndarray, 训练数据的类标签
    :param test_x: ndarray, 验证数据的feature
    :param num_round: 模型迭代次数
    :param eta: 学习率
    :param max_depth: 树的深度
    :param colsample_bytree: 特征采样比
    :return: predict_y: 验证数据的预测结果
             feature_score: dict, 子模型训练的特征权重
    """
    # 子模型参数
    param = {'objective': 'binary:logistic', 'booster': 'gbtree', 'eta': eta, 'max_depth': max_depth, 'eval_metric': 'auc',
             'silent': 1, 'min_child_weight': 0.1, 'subsample': 0.7, 'colsample_bytree': colsample_bytree, 'nthread': 4}
    # 转化训练数据格式
    train_X = xgb.DMatrix(train_x, train_y)
    # 训练ExtGBDT模型
    bst = xgb.train(param, train_X, num_round)
    # 保存子模型的二进制文件
    if not os.path.exists('../model'):
        os.makedirs('../model')
    model_file = '../model/model' + str(model_id)
    pickle.dump(bst, open(model_file, 'w'))
    # 如果是交叉验证,则对测试数据进行预测,并返回预测结果和特征权重
    if cross_validation:
        test_X = xgb.DMatrix(test_x)
        predict_y = bst.predict(test_X)
        feature_score = bst.get_score()
        return np.array(predict_y), feature_score


def ExtGBDTEnsembleTrain(sub_clf_num, train_good, train_bad):
    """
    训练模型
    :param sub_clf_num: 子模型的个数
    :param train_good: list, 训练数据好客户样本
    :param train_bad: list, 训练数据坏客户样本
    """
    num_train_good = len(train_good)    # 训练样本good客户的个数
    num_train_bad = len(train_bad)      # 训练样本bad可以的个数
    eta_list = [0.01, 0.02, 0.03]       # 学习率eta的扰动范围
    max_depth_list = [5, 6, 7]          # 树深度的扰动范围
    colsample_bytree_list = [0.7, 0.8]  # 属性采样比的扰动范围
    round_num_list = range(100, 200)    # 子模型迭代次数扰动范围
    # 训练每一个子模型
    for model_id in range(sub_clf_num):
        # 从训练数据的所有好客户样本中采样出与训练数据所有坏客户等量的样本, 并和所有坏客户样本构成训练数据集
        train_good_sample_id = random.sample(range(num_train_good), num_train_bad)
        train_good_sample = [train_good[i] for i in train_good_sample_id]
        train = train_good_sample + train_bad
        random.shuffle(train)
        train_x, train_y = seprate_xy(train)  # 将训练数据拆分成feature 和 类标签
        round_num = random.choice(round_num_list)                # 随机采样子模型迭代次数
        eta = random.choice(eta_list)                            # 随机采样学习率
        max_depth = random.choice(max_depth_list)                # 随机采样树最大深度
        colsample_bytree = random.choice(colsample_bytree_list)  # 随机采样属性采样比
        # 训练子模型, 返回子模型对验证数据的预测结果和特征权重,第2个train_x只是未来代码的通用新,并没有被使用
        ExtGBDT(0, model_id, train_x, train_y, train_x, round_num, eta, max_depth, colsample_bytree)


def ExtGBDTEnsemblePredict(sub_clf_num, predict_x):
    """
    对新数据进行预测
    :param sub_clf_num: 子模型的个数
    :param predict_x: 待预测数据的feature
    :return: socre: ndarray, 预测结果
    """
    total_score = np.zeros(len(predict_x))  # 保存所有子模型对预测数据评分的总分
    for i in range(sub_clf_num):
        predict_X = xgb.DMatrix(predict_x)
        model_file = '../model/model' + str(i)
        bst = pickle.load(open(model_file, 'r'))
        predict_y = bst.predict(predict_X)
        total_score += predict_y
    score = total_score / sub_clf_num
    return score


def ExtGBDTEnsembleCrossValidation(fold_id, sub_clf_num, train_good, train_bad, test):
    """
    训练模型并进行交叉验证
    :param sub_clf_num: 子模型的个数
    :param train_good: list, 训练数据好客户样本
    :param train_bad: list, 训练数据坏客户样本
    :param test: 验证数据
    """
    total_score = np.zeros(len(test))  # 保存所有子模型对预测数据评分的总分
    total_feature_score = {  # 子模型feature权重的总得分
        'f0': 0, 'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0,
        'f7': 0, 'f8': 0, 'f9': 0, 'f10': 0, 'f11': 0, 'f12': 0, 'f13': 0,
        'f14': 0, 'f15': 0, 'f16': 0, 'f17': 0, 'f18': 0, 'f19': 0, 'f20': 0,
        'f21': 0, 'f22': 0, 'f23': 0
    }
    avg_feature_score = {  # 子模型feature权重的平均分
        'f0': 0, 'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0,
        'f7': 0, 'f8': 0, 'f9': 0, 'f10': 0, 'f11': 0, 'f12': 0, 'f13': 0,
        'f14': 0, 'f15': 0, 'f16': 0, 'f17': 0, 'f18': 0, 'f19': 0, 'f20': 0,
        'f21': 0, 'f22': 0, 'f23': 0
    }
    counter_total_feature_score = Counter(total_feature_score)
    test_x, test_y = seprate_xy(test)   # 将验证数据拆分成feature 和 类标签
    num_train_good = len(train_good)    # 训练样本good客户的个数
    num_train_bad = len(train_bad)      # 训练样本bad可以的个数
    eta_list = [0.01, 0.02, 0.03]       # 学习率eta的扰动范围
    max_depth_list = [5, 6, 7]          # 树深度的扰动范围
    colsample_bytree_list = [0.7, 0.8]  # 属性采样比的扰动范围
    round_num_list = range(100, 200)    # 子模型迭代次数扰动范围
    # 训练每一个子模型
    for model_id in range(sub_clf_num):
        # 从训练数据的所有好客户样本中采样出与训练数据所有坏客户等量的样本, 并和所有坏客户样本构成训练数据集
        train_good_sample_id = random.sample(range(num_train_good), num_train_bad)
        train_good_sample = [train_good[i] for i in train_good_sample_id]
        train = train_good_sample + train_bad
        random.shuffle(train)
        train_x, train_y = seprate_xy(train)  # 将训练数据拆分成feature 和 类标签
        round_num = random.choice(round_num_list)                # 随机采样子模型迭代次数
        eta = random.choice(eta_list)                            # 随机采样学习率
        max_depth = random.choice(max_depth_list)                # 随机采样树最大深度
        colsample_bytree = random.choice(colsample_bytree_list)  # 随机采样属性采样比
        # 训练子模型, 返回子模型对验证数据的预测结果和特征权重
        pred, feature_score = ExtGBDT(1, model_id, train_x, train_y, test_x, round_num, eta, max_depth, colsample_bytree)
        total_score += pred                                      # 子模型概率总和
        counter_total_feature_score += Counter(feature_score)    # 特征权重概率总和
    # 融合所有子模型的预测结果
    # 每个样本为好客户的概率,1减去该值就为坏客户的概率
    score = total_score / sub_clf_num
    # 融合模型的feature权重
    total_feature_score = dict(counter_total_feature_score)
    for k in total_feature_score.keys():
        avg_feature_score[k] = total_feature_score[k] / (sub_clf_num * 1.0)
    avg_feature_score = sorted(avg_feature_score.items(), key=lambda x:x[1], reverse=True)
    # 模型评估
    print '\n\n---------- Fold {0} 模型预测结果 ----------'.format(fold_id + 1)
    # 1.AUC评估
    auc = evaluate_auc(score, test_y)
    # 2.混淆矩阵评估
    predict_y = []
    for i in range(len(score)):
        if score[i] < 0.5:
            predict_y.append(0)
        else:
            predict_y.append(1)
    evaluate(predict_y, test_y)
    # 3.输出模型的特征权重
    # print '特征权重 : ', avg_feature_score
    return auc


def evaluate(predict_y, true_y):
    """
    输出代价敏感矩阵
    :param predict_y: 预测的类标签
    :param true_y: 真实的类标签
    """
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
    """
    计算AUC
    :param pred: 预测的概率
    :param y: 真实的类标签
    """
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
    return train_good, train_bad, test


def remove_model_file():
    if os.path.exists('../model/'):
        all_file = os.listdir('../model/')
        for file in all_file:
            os.remove('../model/' + file)


def train(sub_clf_num = 40):
    """
    训练模型
    :param sub_clf_num: 子模型个数
    """
    remove_model_file()  # 删除已经存在的模型文件
    train_good, train_bad = get_data_set(5)
    ExtGBDTEnsembleTrain(sub_clf_num, train_good, train_bad)


def predict(sub_clf_num = 40):
    """
    预测新样本
    :param sub_clf_num: 子模型个数
    """
    file_predict_x = raw_input('请输入新样本文件: ')
    predict_x = load_data(file_predict_x)
    predict_y = ExtGBDTEnsemblePredict(sub_clf_num, predict_x)
    print predict_y


def train_cross_validation(sub_clf_num = 40):
    """
    训练模型并进行交叉验证,检验模型效果
    :param sub_clf_num: 子模型个数
    :return:
    """
    remove_model_file()  # 删除已经存在的模型文件
    auc = 0
    for i in range(5):
        train_good, train_bad, test = get_data_set(i)
        auc += ExtGBDTEnsembleCrossValidation(i, sub_clf_num, train_good, train_bad, test)
    print "\n\n平均AUC: ", auc / 5


def main():
    sub_clf_num = 40
    print "功能:"
    print "[1] 训练模型并进行验证"
    print "[2] 训练模型"
    print "[3] 预测新数据"
    function = raw_input('请选择功能(1,2或3):')
    if str(function) == '1':
        train_cross_validation(sub_clf_num)
    elif str(function) == '2':
        train(sub_clf_num)
    elif str(function) == '3':
        predict(sub_clf_num)
    else:
        print "输入错误..."


if __name__ == '__main__':
    main()
