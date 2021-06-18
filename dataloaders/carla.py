from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CHANGE_LABEL_OLD = {0: 0, 
                1: 1, 
                2: 3, 
                3: 3, 
                4: 4, 
                5: 5, 
                6: 6,
                7: 7, 
                8: 8, 
                9: 9, 
                10: 10, 
                11: 11, 
                12: 12, 
                13: 3,
                14: 3, 
                15: 3, 
                16: 3, 
                17: 3,
                18: 3, 
                19: 3, 
                20: 3, 
                21: 3, 
                22: 3, 
                23: 2, 
                24: 3}

CHANGE_LABEL = {0: 0, 
                210: 1, 
                180: 2, 
                225: 3, 
                300: 4, 
                459: 5, 
                441: 6,
                320: 7, 
                511: 8, 
                284: 9, 
                142: 10, 
                360: 11, 
                440: 12, 
                380: 13,
                162: 14, 
                350: 15, 
                520: 16, 
                525: 17,
                450: 18, 
                460: 19, 
                340: 20, 
                255: 21, 
                415: 22}

class CarlaDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 23
        self.palette = palette.Carla_palatte
        self.change_babel = CHANGE_LABEL
        super(CarlaDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, 'img')
        self.label_dir = os.path.join(self.root, 'gt')

        file_list = os.path.join(self.root, self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.bmp')
        label_path = os.path.join(self.label_dir, image_id + '.bmp')
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        label_origin = np.asarray(Image.open(label_path).convert("RGB"), dtype=np.int32)
        label = label_origin.sum(axis=2)
        for k, v in self.change_babel.items():
            label[label == k] = v
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class Carla(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):
        
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
    
        if split in ["train", "val", "test"]:
            self.dataset = CarlaDataset(**kwargs)
        else: 
            raise ValueError(f"Invalid split name {split}")
        super(Carla, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

