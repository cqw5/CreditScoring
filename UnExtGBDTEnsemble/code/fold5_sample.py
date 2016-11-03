#! /usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'chenqiwei'

import random
import os

def read_data(file):
    """
    读取数据
    读取原始数据,将其中的好客户保存在列表data_good中,坏客户保存在列表data_bad中
    :param file: 原始数据文件
    :return: data_good: list, 好客户列表
             data_bad : list, 坏客户列表
    """
    data_good = []   # 好客户
    data_bad = []    # 坏客户
    with open(file, 'r') as f:
        all_line = f.readlines()
        for line in all_line:
            content = line.strip().split(',')
            if content[len(content) - 1] == '0':
                content[len(content) - 1] = '1'
                data_good.append(content)
            else:
                content[len(content) - 1] = '0'
                data_bad.append(content)
    return data_good, data_bad


def write_data(file, data):
    """
    写数据
    将data中的数据写入到文件file中
    :param file: 数据写入(保存)的文件
    :param data: 数据
    :return:
    """
    with open(file, 'w') as f:
        for line in data:
            f.write(','.join(line) + '\n')


def main():
    """
    输入数据读取路径和保存路径
    读取数据,并将数据分为好客户和坏客户,用随机函数扰乱数据顺序
    将所有好客户和所有坏客户分别分为5折,分别保存
    """
    file_in = '../data/src_data/german_data_numeric.csv'
    file_good = '../data/fold5_data/good/'
    file_bad = '../data/fold5_data/bad/'
    if not os.path.exists(file_good):
        os.makedirs(file_good)
    if not os.path.exists(file_bad):
        os.makedirs(file_bad)
    file_in = raw_input("Input src file: ")
    file_fold1_good = file_good + raw_input("Input fold1 reliable customers save file: ")
    file_fold1_bad = file_bad + raw_input("Input fold1 defaulters save file: ")
    file_fold2_good = file_good + raw_input("Input fold2 reliable customers save file: ")
    file_fold2_bad = file_bad + raw_input("Input fold2 defaulters save file: ")
    file_fold3_good = file_good + raw_input("Input fold3 reliable customers save file: ")
    file_fold3_bad = file_bad + raw_input("Input fold3 defaulters save file: ")
    file_fold4_good = file_good + raw_input("Input fold4 reliable customers save file: ")
    file_fold4_bad = file_bad + raw_input("Input fold4 defaulters save file: ")
    file_fold5_good = file_good + raw_input("Input fold5 reliable customers save file: ")
    file_fold5_bad = file_bad + raw_input("Input fold5 defaulters save file: ")
    data_good, data_bad = read_data(file_in)
    random.shuffle(data_good)
    random.shuffle(data_bad)
    data_good_fold1 = data_good[0: 140]
    data_good_fold2 = data_good[140: 280]
    data_good_fold3 = data_good[280: 420]
    data_good_fold4 = data_good[420: 560]
    data_good_fold5 = data_good[560: 700]
    data_bad_fold1 = data_bad[0: 60]
    data_bad_fold2 = data_bad[60: 120]
    data_bad_fold3 = data_bad[120: 180]
    data_bad_fold4 = data_bad[180: 240]
    data_bad_fold5 = data_bad[240: 300]
    write_data(file_fold1_good, data_good_fold1)
    write_data(file_fold2_good, data_good_fold2)
    write_data(file_fold3_good, data_good_fold3)
    write_data(file_fold4_good, data_good_fold4)
    write_data(file_fold5_good, data_good_fold5)
    write_data(file_fold1_bad, data_bad_fold1)
    write_data(file_fold2_bad, data_bad_fold2)
    write_data(file_fold3_bad, data_bad_fold3)
    write_data(file_fold4_bad, data_bad_fold4)
    write_data(file_fold5_bad, data_bad_fold5)


if __name__ == '__main__':
    main()
