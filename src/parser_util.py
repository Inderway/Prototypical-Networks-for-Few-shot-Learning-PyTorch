# coding=utf-8
import os
import argparse


def get_parser():
    # 创建解析对象
    parser = argparse.ArgumentParser()
    # 添加参数
    # os.sep根据操作系统自动选择路径分隔符\或者/
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='..' + os.sep + 'dataset')
    # 命令行解析的参数命令名为dataset_root，简写为root, 参数类型

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='..' + os.sep + 'output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    # scheduler与optimizer不同，其在每个epoch调整学习率，每个step为一个epoch
    # 每20个epoch调整一次学习率变为lr*gamma
    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    # 每个epoch的小批数
    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    # 每个小批中指定的随机类的数目，默认为60
    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=60)

    # 每个类的支持集样本数
    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    # 每个类的查询集样本数
    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)

    # 每个小批的验证类数
    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)

    # 每个类的验证支持样本数
    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    # 每个类的验证查询样本数
    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=15)

    # 使初始随机值一致，如果不设置，那么seed就是当前时间，每次生成的随机数都不一样
    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    # 一旦命令中有该参数，则其值为true
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    return parser
