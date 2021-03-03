# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}


class OmniglotDataset(data.Dataset):
    vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='..' + os.sep + 'dataset', transform=None, target_transform=None, download=False):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(OmniglotDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        # print('debug0')
        if download:
            print('debug1')
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')
        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        # dataset/data/
        # [(file, label, root, rot),(),...]
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes)

        # {'language/character/rot': index, ...}
        self.idx_classes = index_classes(self.all_items)

        # __len__重写了len， 返回all_items的元组数
        # zip+*表示将元组还原成列表
        # [(root/xxx.pngrot, language/character/rot)...]
        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])
        # a 1x28x28 img
        self.x = map(load_img, paths, range(len(paths)))
        self.x = list(self.x)
        # 数据处理完毕，x为图片, y为标注

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        # 获得png文件
        filename = self.all_items[index][0]
        # print(filename)
        #获得角度
        rot = self.all_items[index][-1]
        # root/xxx.png + rot
        img = str.join(os.sep, [self.all_items[index][2], filename]) + rot
        # language/character/rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))

    def download(self):
        from six.moves import urllib
        import zipfile

        #if self._check_exists():
        #   return

        try:
            os.makedirs(os.path.join(self.root, self.splits_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k, url in self.vinyals_split_sizes.items():
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[-1]

            file_path = os.path.join(self.root, self.splits_folder, filename)
            print(file_path)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            print(file_path)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            # dataset/raw
            orig_root = os.path.join(self.root, self.raw_folder)
            print("== Unzip from " + file_path + " to " + orig_root)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(orig_root)
            zip_ref.close()
        # dataset/data
        file_processed = os.path.join(self.root, self.processed_folder)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(orig_root, p)):
                # dataset/data/image_background
                # move all directories to dataset/data
                shutil.move(os.path.join(orig_root, p, f), file_processed)
            os.rmdir(os.path.join(orig_root, p))
        print("Download finished.")


def find_items(root_dir, classes):
    retour = []
    # 4个旋转角度
    rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            # D:\Github\Prototypical-Networks-for-Few-shot-Learning-PyTorch\dataset\data\Alphabet_of_the_Magi\character01
            r = root.split(os.sep)
            lr = len(r)
            # Alphabet_of_the_Magi/character01
            label = r[lr - 2] + os.sep + r[lr - 1]
            for rot in rots:
                # classes中有该label的png图片
                if label + rot in classes and (f.endswith("png")):
                    # png文件, 标注, 路径, 旋转角度
                    retour.extend([(f, label, root, rot)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    # [(file, label, root, rot),(),...]
    for i in items:
        # label+rot不在idx中
        if not i[1] + i[-1] in idx:
            # hash表, key为language/character/rot, value为index
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    print(fname)
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    # 返回一个列表
    # 每个元素的格式: <language>/character<idx>/rot<degree>
    # 最后一个是旋转角度，共4个(0,90,180,270)
    return classes


def load_img(path, idx):
    # root/xxx.png, degree
    path, rot = path.split(os.sep + 'rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        # 暂存
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    # 1x28x28
    shape = 1, x.size[0], x.size[1]
    # 28x28
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    # transpose转置
    # contiguous 返回内存连续的tensor, 不然不能使用view
    # 把x变成shape的维度
    x = x.transpose(0, 1).contiguous().view(shape)

    return x
