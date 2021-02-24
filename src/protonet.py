import torch.nn as nn


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        # out_channels决定了卷积核的数量，padding为补0.该卷积核维度为3，尺寸为3xin_channels
        # 由于padding为1，最后输出尺寸为in_channels x out_channels
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        # 正则化，以减小深度网络中前一层网络的参数对于后面的影响，提高SGD的效率
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # 最大池， 把2x2的池化核贴上
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        # 以下操作保留x尺寸的第一个参数的维度，剩下数据合并成一个列表
        # 如: x的size为2x3x4, tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[,17,18,19,20],[21,22,23,24]]])
        # 通过下面处理后变为tensor([[...],[...]])
        return x.view(x.size(0), -1)
