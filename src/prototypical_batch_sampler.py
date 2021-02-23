# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        # 所有类初始成tensor
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        # np.nan为一个标记值，意为not a number
        # 创建一个m为类数，n为类众数的矩阵并设为nan
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        # tensor化
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        # labels为dataset.y
        for idx, label in enumerate(self.labels):
            # argwhere返回满足条件的索引
            # 获得dataset.y中的label在类数组中的索引
            label_idx = np.argwhere(self.classes == label).item()
            # indexes的每一行代表一个类
            # np.where输出满足条件的在数组indexes[label_idx]中的索引列表，第一个0获得索引列表，第二个0获得第一个满足条件的索引
            # 在最前的nan值中填上该类在dataset.y中的索引
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            # 该类出现次数加1
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            # randperm返回从0到类数的整数随机排列
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                # 切片, 起始为i*spc, 结束为(i+1)*spc-1
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                # arange生成一个从0到len-1的tensor并设为long型， 获得classes中c的索引
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                # 获得类c在indexes中保存的索引。如果spc为3，随机到的索引为2, 0, 3, 那么就取indexes[label_idx][2,0,3]里的值
                # 该值为其对应的在dataset.y中的索引
                batch[s] = self.indexes[label_idx][sample_idxs]
            # 打乱小批的顺序
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
