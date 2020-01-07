# -*- coding: utf-8 -*-
"""
    @author: samuel ko
    @date: 2019.07.18
    @readme: The implementation of PRNet Network DataLoader.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import cv2
import random
import numbers
import numpy as np
from PIL import Image
from skimage import io
import glob

np.random.seed(777)

data_transform = {'train': transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}


class PRNetDataset(Dataset):
    """Pedestrian Attribute Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dict = dict()
        self._max_idx()

    def get_img_path(self, img_id):
        img_path = self.dict.get(img_id)
        if img_path != None: #if img_id 这句话限制了数据集的文件名不能为0，否则就会出错
            original = os.path.join(img_path, 'original.jpg')
            uv_map = glob.glob(os.path.join(img_path,'*.npy'))[0]
            return original, uv_map
        else :
            print('数据集的名字有问题')


    # def _max_idx(self):
    #     _tmp_lst = map(lambda x: int(x), os.listdir(self.root_dir))
    #     _sorted_lst = sorted(_tmp_lst)
    #     for idx, item in enumerate(_sorted_lst):
    #         self.dict[idx] = item

    def _max_idx(self):
        image_list = []
        for root_dir in self.root_dir:
            image_list.extend(glob.glob(os.path.join(root_dir,'*')))
        np.random.shuffle(image_list)
        self.dict = dict(zip(range(len(image_list)),image_list))

    def __len__(self):
        image_len = list(map(lambda root_dir : len(os.listdir(root_dir)),self.root_dir))
        return sum(image_len)

    def __getitem__(self, idx):
        
        #获取图像和uv映射图
        original, uv_map = self.get_img_path(idx)
        origin = cv2.imread(original)
        uv_map = np.load(uv_map)

        #返回字典
        sample = {'uv_map': uv_map, 'origin': origin}
        
        #数据增强
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        uv_map, origin = sample['uv_map'], sample['origin']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        uv_map = uv_map.transpose((2, 0, 1))
        origin = origin.transpose((2, 0, 1))

        # uv_map = uv_map.astype("float32") / 255.
        uv_map = uv_map.astype("float32") / 255.
        origin = origin.astype("float32") / 255.
        return {'uv_map': torch.from_numpy(uv_map), 'origin': torch.from_numpy(origin)}


class ToNormalize(object):
    """Normalized process on origin Tensors."""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        uv_map, origin = sample['uv_map'], sample['origin']
        origin = F.normalize(origin, self.mean, self.std, self.inplace)
        return {'uv_map': uv_map, 'origin': origin}
