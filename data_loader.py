import os
import cv2
import copy
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MonoDataset(Dataset):
    def __init__(self, data_root,
                 train_txt,
                 frame_ids=[0, -1, 1],
                 num_scales=4,
                 img_shape=(192, 640),
                 is_train=True,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()
        self.data_root = data_root
        self.frame_ids = frame_ids
        self.num_scales = num_scales
        self.img_shape = img_shape
        self.is_train = is_train
        self.img_ext = img_ext

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.img_shape[0]//s, self.img_shape[1]//s),
                                               interpolation=Image.ANTIALIAS)
            
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        
        self.to_tensor = transforms.ToTensor()

        self.lines = self.get_lines(train_txt)

    def get_lines(self, train_txt):
        with open(train_txt, 'r') as f:
            lines = f.read().splitlines()
        return lines

    def preprocess_img(self, sample, color_aug):
        for k in list(sample):
            if "color" in k:
                n, id, s = k
                for s in range(self.num_scales):
                    sample[(n, id, s)] = self.resize[s](sample[(n, id, s-1)])

        for k in list(sample):
            if "color" in k:
                img = sample[k]
                n, id, s = k
                sample[(n, id, s)] = self.to_tensor(img)
                sample[(n+'_aug', id, s)] = self.to_tensor(color_aug(img))
    
    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        '''
        Returns:
        - img		torch.Tensor (N * H, W, 3)
        - K	torch.Tensor (num_scales, 3, 3)
        - inv_K	torch.Tensor (num_scales, 3, 3)
        '''
        sample = {}

        do_color_aug = self.is_train and np.random.rand() > 0.5
        do_flip = self.is_train and np.random.rand() > 0.5


        line = self.lines[idx].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_ids:
            sample[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
            
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        
        self.preprocess_img(sample, color_aug)

        for i in self.frame_ids:
            del sample[("color", i, -1)]
            del sample[("color_aug", i, -1)]

        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.img_shape[1] // (2**scale) # x
            K[1, :] *= self.img_shape[0] // (2**scale) # y

            inv_K = np.linalg.pinv(K)

            sample[("K", scale)] = torch.from_numpy(K)
            sample[("inv_K", scale)] = torch.from_numpy(inv_K)
        
        return sample
    
class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

class GetKITTIRaw(KITTIDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(GetKITTIRaw, self).__init__(*args, **kwargs)

    def get_color(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_root,
                                  folder,
                                  "image_0{}/data".format(self.side_map[side]),
                                  f_str)

        color = Image.open(image_path)
        color = color.convert('RGB')
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color

class GetKITTIOdom(KITTIDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(GetKITTIOdom, self).__init__(*args, **kwargs)

    def get_color(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_root,
                                  "sequences/{:02d}".format(int(folder)),
                                  "image_{}".format(self.side_map[side]),
                                  f_str)

        color = Image.open(image_path)
        color = color.convert('RGB')
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color
    
def get_data_loader(
                config,
                transforms=None,
                sampler=None,
                drop_last=True,
                ):
    """
    Return batch data for training.
    """
    if config.dataset =='kitti_raw':
        dataset = GetKITTIRaw(config.kitti_raw_root,
                                config.kitti_raw_txt,
                                frame_ids=config.frame_ids,
                                num_scales=config.num_scales,
                                img_shape=config.kitti_hw,
                                is_train=True,
                                img_ext=config.img_ext)
    elif config.dataset =='kitti_odom':
        dataset = GetKITTIOdom(config.kitti_odom_root,
                                config.kitti_odom_txt,
                                frame_ids=config.frame_ids,
                                num_scales=config.num_scales,
                                img_shape=config.kitti_hw,
                                is_train=True,
                                img_ext=config.img_ext)

    train_loader = DataLoader(
                        dataset,
                        batch_size=config.batch_size,
                        shuffle=config.shuffle,
                        sampler=sampler,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory,
                        drop_last=drop_last
                        )

    return train_loader