#!/usr/bin/env python
import os
import shutil
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from .CUB200 import classes
try:
    from .download_gdrive import download_file_from_google_drive
except ImportError:
    from download_gdrive import download_file_from_google_drive


gdrive_id = "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
dataset_zip = "CUB_200_2011.tgz"
orig_folder = "CUB_200_2011"
orig_attribute_file = "attributes.txt"
orig_attribute_class_file = "class_attribute_labels_continuous.txt"
image_file = "images.txt"
image_folder = "images"
target_file = "targets.npy"
class_name_file = "class_names.txt"
main_classes = 200


# noinspection PyUnresolvedReferences
def download_cub(root=".", dataset_folder="cub200", force=False):
    origin_dir = os.path.abspath(os.curdir)
    os.chdir(root)

    if not os.path.isfile(dataset_zip):
        download_file_from_google_drive(gdrive_id, dataset_zip)
        print("Dataset downloaded")

    if not os.path.isdir(dataset_folder) or force:
        if os.path.isdir(dataset_folder):
            shutil.rmtree(dataset_folder)
        os.system(f"tar -zxvf {dataset_zip}")
        os.rename(orig_folder, dataset_folder)
        print("\nDataset extracted")

        os.chdir(dataset_folder)

        attribute_per_class = pd.read_csv(os.path.join("attributes",
                                                       orig_attribute_class_file),
                                          sep=" ",
                                          header=None).to_numpy()
        print("Mean attribute presence for class loaded")

        classes = pd.read_csv("image_class_labels.txt", " ", header=None).to_numpy()[:, 1]
        print("Image labels loaded")

        n_samples = classes.shape[0]
        n_attributes = attribute_per_class.shape[1]
        n_classes = np.max(classes)
        n_attributes += n_classes
        very_denoised_attributes = np.zeros((n_samples, n_attributes))
        attribute_sparsity = np.zeros(n_attributes)
        # classes were indexed starting from 1
        classes = classes - 1
        for c in np.unique(classes):
            # setting labels for all instances of the same class at once
            imgs = classes == c
            class_attributes = attribute_per_class[c - 1, :] > 50
            # setting attributes one hot
            very_denoised_attributes[imgs, n_classes:] = class_attributes
            # setting class one hot
            very_denoised_attributes[imgs, c] = 1
            # setting frequency of the attributes
            attribute_sparsity[n_classes:] += class_attributes
            # setting frequency of the class
            attribute_sparsity[c] += np.sum(imgs)
        attributes_to_filter = attribute_sparsity < 10
        attributes_filtered = np.sum(attributes_to_filter)
        print("Number of attributes filtered", attributes_filtered)

        final_targets = very_denoised_attributes[:, ~attributes_to_filter]
        final_targets.dump(target_file)
        print("Targets saved")

        if os.path.exists(os.path.join("..", orig_attribute_file)):
            shutil.move(os.path.join("..", orig_attribute_file), ".")
        with open(orig_attribute_file, "r") as f:
            attribute_names = [line.split(" ")[1] for line in f.readlines()]
        attribute_names = np.asarray(attribute_names)
        attribute_names = attribute_names[~attributes_to_filter[n_classes:]]

        class_names = os.listdir(image_folder)
        class_names = [name.split(".")[1] for name in class_names]
        class_names += attribute_names.tolist()

        with open(class_name_file, "w") as f:
            for name in class_names:
                f.write(name + "\n")
        print("Class names saved")

        with open(image_file, "r") as f:
            lines = f.readlines()
            for line in tqdm.tqdm(lines, desc="Renaming image files"):
                fields = line.split(" ")
                img_idx = f"{int(fields[0]):05d}"
                img_name = fields[1][:-1]
                ext = os.path.splitext(img_name)[1]
                img_name = os.path.join(image_folder, img_name)
                new_name = os.path.join(image_folder, img_idx) + ext
                shutil.move(img_name, new_name)
        print("Images sorted and renamed")

        for item in os.listdir():
            if item not in [image_folder, class_name_file, target_file]:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
        for item in os.listdir(image_folder):
            item = os.path.join(image_folder, item)
            if os.path.isdir(item):
                shutil.rmtree(item)
        print("Temporary file cleaned")

    print("Dataset configured correctly")
    os.chdir(origin_dir)


class CUBDataset(Dataset):

    def __init__(self, root: str, transforms: torchvision.transforms.Compose):
        self.root = root
        self.transforms = transforms

        # Check if we need to download the dataset
        data_folder = os.path.dirname(root)
        dataset_folder = os.path.basename(root)
        download_cub(data_folder, dataset_folder)

        # Load the targets
        self.targets = np.load(os.path.join(root, target_file), allow_pickle=True)
        self.n_classes = self.targets.shape[1]
        self.main_classes = [*range(main_classes)]
        self.attributes = [*range(main_classes, self.n_classes)]

        # Load class-attributes combinations (200, 108)
        self.class_attr_comb = []
        for c in self.main_classes:
            comb = np.unique(self.targets[np.where(self.targets[:, c])]
                             [:, main_classes:], axis=0)
            self.class_attr_comb.append(comb)
        self.class_attr_comb = np.asarray(self.class_attr_comb).squeeze()
        self.imgs = {i: None for i in range(len(self))}

        # with open(os.path.join(root, class_name_file), "r") as f:
        #     class_names = f.read().splitlines()
        #     self.class_names = class_names
        self.class_names = classes

    def __getitem__(self, index: int) -> [torch.Tensor, torch.Tensor]:
        # Load the image
        img_index = f"{index+1:05d}"
        img_path = os.path.join(self.root, image_folder, img_index) + ".jpg"
        if self.imgs[index] is None:
            img = Image.open(img_path).convert('RGB')
            img = self.transforms(img)
            self.imgs[index] = img
        else:
            img = self.imgs[index]

        # Load the label
        label = self.targets[index]

        return img, label

    def __len__(self):
        return self.targets.shape[0]


if __name__ == "__main__":
    download_cub(force=False)
