# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception
    # unsqueeze，加指定维度前一个维度, 比如原tensor的size为1x2x3, unsqueeze(0)，则变为1x1x2x3
    # expand增大维度的数值,比如原tensor的size为1x2x3, expand(2,2,3)以后变为2x2x3,若参数为-1则不变
    # x.size: NxD->Nx1xD->nxmxd
    # y.size: MxD->1xMxD->nxmxd
    # 至于为什么一个是0一个是1, 因为expand只加不减
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # sum中2的意思就是对第2维的数据求和，第0维和第1维保留
    # 比如tensor([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],
    #             [[13,14,15],[16,17,18],[19,20,21],[22,23,24]]])
    # 该tensor的size为2x4x3，对第2维求和就是tensor([[6,15,24,33],[42,51,60,69]]), size为2x4
    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    # 通过平均每个类中n_support个样本特征来获得每个类的中心
    # 对于每个类，计算该类样本与中心的距离
    # 计算n_query个样本与每个类中心的距离
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    # ground truth即为golden label
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    # input: 600x64  target:600 n_support=5
    # loss在cpu上计算，与GPU并行
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        # nonzero返回输入中非零元素组成的tensor, 其维度为 非0元素个数n x input的维数
        # c为classes中的一个类, 从target中取5个，返回5x1的tensor, squeeze去掉第一维的维度，返回长为5的tensor, 元素值为其在target中索引
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    # 不重复的label列表 600
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    # eq返回target中与索引为0的类相同的类数(结果即为样本数, 因为每个类有10个样本)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    # 每个类对应一个元组, 其中为5个其在target中的索引
    support_idxs = list(map(supp_idxs, classes))
    # 每个idx_list为一个类的元组
    # stack用于tensor拼接
    # input为600x64, idx_list为5, input[idx_list]为 以idx_list的每个元素作为索引，其在input中的tensor的组合
    # 比如idx_list=[1,2,3,4,5], 则input[idx_list]则为[input[1], input[2], input[3], input[4], input[5]] 5x64
    # mean(0)即对第0维求均值, size为64
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val
