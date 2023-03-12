from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
# from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
import os
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir,args):
        self.args=args
        patch_size = [self.args.patch_x,self.args.patch_y,self.args.patch_z]
        self.image_paths=images_dir
        self.label_paths=labels_dir
        queue_length = 5
        samples_per_volume = 5
        self.subjects = []
        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)


        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            self.training_set,
            queue_length,
            samples_per_volume,
            UniformSampler(patch_size),
        )




    def transform(self):
        if self.args.aug:
            print('With data augmentation!!!')
            training_transform = Compose([
            RandomBiasField(),
            ZNormalization(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
            OneOf({
                RandomAffine(): 0.8,
                RandomElasticDeformation(): 0.2,
            }),])
        else:
            print('Without data augmentation!!!')
            training_transform = Compose([
            ZNormalization(),
            ])
        return training_transform




class MedData_val(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir,args):
        self.args=args
        patch_size = [self.args.patch_x,self.args.patch_y,self.args.patch_z]
        self.image_paths=images_dir
        self.label_paths=labels_dir
        queue_length = 5
        samples_per_volume = 5
        self.subjects = []
        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)
        self.transforms = self.transform()
        self.val_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)
        self.queue_dataset = Queue(
            self.val_set,
            queue_length,
            samples_per_volume,
            UniformSampler(patch_size),
        )




    def transform(self):
        val_transform = Compose([
        ZNormalization(),
        ])
        return val_transform
if __name__ == '__main__':
    path = r'C:\Users\Administrator\Desktop\rightcrop'
    from tools import data_splitting

    list = os.listdir(path)
    train_list, val_list = data_splitting.get_split_deterministic(list, fold=0, num_splits=5, random_state=12345)
    train_data = [os.path.join(path, x) for x in train_list]
    train_label = [os.path.join(path, x) for x in train_list]

    val_data = [os.path.join(path, x) for x in val_list]
    val_label = [os.path.join(path, x) for x in val_list]


    train_dataset = MedData_train(train_data, train_label)
    print(train_dataset)



