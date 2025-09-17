import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import torchio as tio # python library for medical imagign


class BraTSDataset(Dataset):
    def __init__(self, data_dir, patch_size=(64, 64, 64), is_train=True):
        self.data_files = sorted(glob(os.oath.join(data_dir, '*_img.npy')))
        self.patch_size = patch_size
        self.is_train = is_train

        # define augmentation pipeline for training
        if self.is_train:
            self.transform = tio.Compose([
                tio.RandomAffine(scales=(0.9, 1.2), degrees=15, istrropic=True),
                tio.RandomFlip(axes=('LR',)),
                tio.RandomGamma(log_gamma=(-0.3, 0.3)),
                tio.RandomNoise(std=0.1)
            ])
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        img_path = self.data_files[idx]
        seg_path = img_path.replace("_img.npy", "_seg.npy")

        image = np.load(img_path) # shape: (4, H, W, D): 4 means 4 MRI modalities that were stacked togethe 
        label = np.load(seg_path) # shape: (H, W, D)

        # torchio subject for each augmentation and patcing
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            label=tio.LabelMap(tensor=label[np.newaxis, :])
        )

        # define a patch sampler
        sampler = tio.data.LabelSampler(patch_size=self.patch_size, label_name='label')
        # extract one patch
        patch = list(sampler(subject, num_patches=1))[0]

        # apply augmentation if the patch is in training set
        if self.is_train:
            patch = self.transform(patch)

        image_patch = patch.image.data
        label_patch = patch.label.data

        return image_patch, label_patch

