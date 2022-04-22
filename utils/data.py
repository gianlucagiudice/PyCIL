import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

PROJECT_ROOT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT_PATH)

from config import METADATA_CROPPED_IMAGE_PATH, DATASET_PATH, LOGODET_3K_NORMAL_PATH


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def init_class_cropped_logodet3k(dataset_dir):
    df = read_df_cropped_logodet3k(dataset_dir)
    n_classes = df['brand'].unique().size
    print(f'The total number of classes is  {n_classes}')
    return np.arange(n_classes).tolist()


def read_instances(split, dataset_dir, prefix='../'):
    with open(prefix / Path(DATASET_PATH) / dataset_dir / split) as file:
        return [Path(x.strip()) for x in file.readlines()]


def read_df_cropped_logodet3k(dataset_dir, root=f'../{DATASET_PATH}'):
    df = pd.read_pickle(Path(root) / dataset_dir / METADATA_CROPPED_IMAGE_PATH)

    train = read_instances('train.txt', dataset_dir)
    validation = read_instances('validation.txt', dataset_dir)
    test = read_instances('test.txt', dataset_dir)

    all_instances = [Path(x.name) for x in train + validation + test]
    sub_df = df[df['cropped_image_path'].isin(all_instances)]
    return sub_df


class iLogoDet3K(iData):
    use_path = True

    train_trsf = []
    test_trsf = []

    common_trsf = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]

    class_order = None

    DATASET_PATH = Path(PROJECT_ROOT_PATH) / 'dataset' / LOGODET_3K_NORMAL_PATH

    def download_data(self, data_augmentation=False):
        # Init data augmentation
        if data_augmentation:
            iLogoDet3K.train_trsf = iLogoDet3K.init_data_augmentation()

        self.df_cropped = read_df_cropped_logodet3k(iLogoDet3K.DATASET_PATH)
        self.train_instances = read_instances('train.txt', iLogoDet3K.DATASET_PATH)
        self.validation_instances = read_instances('validation.txt', iLogoDet3K.DATASET_PATH)
        self.test_instances = read_instances('test.txt', iLogoDet3K.DATASET_PATH)

        iLogoDet3K.class_order = np.arange(self.df_cropped['brand'].unique().size).tolist()

        # Class to index
        self.classes = self.df_cropped['brand'].unique()
        self.class_to_idx = {b: i for i, b in enumerate(self.classes)}
        self.class_order = np.arange(0, len(self.classes))
        print(f'Class to idx len: {len(self.class_to_idx.keys())}')

        # Split df
        # TODO: Use train and validation separatly
        train_instances = [Path(x.name) for x in self.train_instances + self.validation_instances]
        train_df = self.df_cropped[self.df_cropped['cropped_image_path'].isin(train_instances)]

        test_instances = [Path(x.name) for x in self.test_instances]
        test_df = self.df_cropped[self.df_cropped['cropped_image_path'].isin(test_instances)]

        train_dir = iLogoDet3K.DATASET_PATH / 'train'
        test_dir = iLogoDet3K.DATASET_PATH / 'val'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Copy train images
        self.copy_images(train_df, 'train')
        # Copy test images
        self.copy_images(test_df, 'val')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

    @staticmethod
    def init_data_augmentation():
        affine_transformation = [
            transforms.RandomAffine(10),
            transforms.RandomPerspective(0.4, 1)
        ]
        image_distortion = [
            transforms.RandomAdjustSharpness(10, 1),
            transforms.RandomPosterize(5),
            transforms.ColorJitter((0.9, 1), (0.9, 1), (0.9, 1), (-0.01, 0.01))
        ]

        final_transformation = transforms.Compose([
            transforms.RandomChoice(affine_transformation),
            transforms.RandomChoice(image_distortion)
        ])

        return [final_transformation]

    def copy_images(self, dataframe, split):
        for _, row in tqdm.tqdm(dataframe.iterrows(), total=len(dataframe)):
            os.makedirs(iLogoDet3K.DATASET_PATH / split / str(self.class_to_idx[row.brand]), exist_ok=True)
            src = iLogoDet3K.DATASET_PATH / 'cropped' / str(row.cropped_image_path)
            dst = iLogoDet3K.DATASET_PATH / split / str(self.class_to_idx[row.brand]) / str(row.cropped_image_path)

            if not os.path.exists(dst):
                shutil.copy(src, dst)


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
