import os
import shutil
import zipfile
from typing import List

import numpy as np
import requests
import tqdm
from PIL import Image
from sortedcontainers import SortedSet


def download(url, file_path):
    print("Downloading %s" % file_path)
    response = requests.get(url, allow_redirects=True, stream=True)
    total_length = response.headers.get('content-length')

    with open(file_path, "wb") as f:
        pbar = tqdm.tqdm(total=int(total_length))
        for data in response.iter_content(chunk_size=4096):
            f.write(data)
            pbar.update(len(data))
    pbar.close()


dataset_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
dataset_folder = "PennFudanPed"
zip_path = "PennFudanPed.zip"
config_file = "PennFudanPed.data"

orig_image_folder = "PNGImages"
orig_mask_folder = "PedMasks"
image_folder = "images"
label_folder = "labels"


def create_PennFudanPed_dataset(root_folder: str,
                                train_idx: List[int] = None,
                                test_idx: List[int] = None):

    orig_dir = os.path.abspath(os.curdir)
    os.chdir(root_folder)

    # Check if folder exists and it is not empty
    if not os.path.exists(dataset_folder) or len(os.listdir(dataset_folder)) == 0:

        # Check if zip file already downloaded
        if not os.path.exists(zip_path):
            download(dataset_url, zip_path)
        print("Dataset Downloaded")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    print("Dataset extracted")

    os.chdir(dataset_folder)

    # Creating class name files
    if not os.path.exists("classes.names"):
        with open("classes.names", "w") as f:
            f.write("Pedestrian")
    print("Class name file created")

    # Creating labels
    if not os.path.exists(label_folder) or \
            len(os.listdir(label_folder)) < len(os.listdir(orig_image_folder)):
        if os.path.exists(label_folder):
            shutil.rmtree(label_folder)
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)

        os.makedirs(label_folder)
        os.makedirs(image_folder)

        print("Creating the labels...")
        images = sorted(os.listdir(orig_image_folder))
        masks = sorted(os.listdir(orig_mask_folder))
        for idx, (img_path, mask_path) in enumerate(zip(images, masks)):
            mask_path = os.path.join(orig_mask_folder, mask_path)
            orig_img_name = os.path.join(orig_image_folder, img_path)
            img_name = os.path.join(image_folder, f"{idx}.png")
            label_path = os.path.join(label_folder, f"{idx}.txt")

            shutil.copy(orig_img_name, img_name)

            mask = Image.open(mask_path)

            # instances are encoded as different colors
            mask_np = np.array(mask)
            obj_ids = np.unique(mask_np)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set of binary masks
            masks = mask_np == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)

            with open(label_path, "w") as f:
                for i in range(num_objs):
                    pos = np.where(masks[i])
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])

                    img_width = mask.width
                    img_heigth = mask.height
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_heigth
                    width = np.abs(xmax - xmin)/img_width
                    height = np.abs(ymax - ymin)/img_heigth

                    assert 0 < x_center <= 1 and \
                           0 < y_center <= 1 and \
                           0 < width <= 1 and \
                           0 < height <= 1, f"Error in creating mask " \
                                            f"{x_center, y_center} {width, height}"

                    class_idx = 0  # there is only one class

                    f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
    print("Labels created")

    unuseful_folders = ["Annotation", orig_mask_folder, "PNGImages"]
    for folder in unuseful_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    unuseful_files = ["added-object-list.txt", "readme.txt",]
                      # os.path.join("..", "PennFudanPed.zip")]
    for file in unuseful_files:
        if os.path.exists(file):
            os.remove(file)
    print("Folder cleaned")

    dataset_files = os.listdir(image_folder)
    dataset_len = len(dataset_files)
    if train_idx is None or test_idx is None:
        test_files = SortedSet(np.random.choice(dataset_files,
                                                size=dataset_len // 10, replace=False))
        train_files = SortedSet(dataset_files) - test_files
        print("Created train test split idx (90/10)")
    else:
        train_files = [dataset_files[idx] for idx in train_idx]
        test_files = [dataset_files[idx] for idx in test_idx]

    for split, split_files in zip(["train", "test"], [train_files, test_files]):
        file = f"{split}.txt"
        with open(file, "w") as f:
            for file in split_files:
                img_path = os.path.join("data", "PennFudanPed", "images", f"{file}")
                f.write(f"{img_path}\n")
    print("Train test files created")

    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write("classes=1\n")
            f.write(f"train={os.path.join('data', 'PennFudanPed', 'train.txt')}\n")
            f.write(f"valid={os.path.join('data', 'PennFudanPed', 'test.txt')}\n")
            f.write(f"names={os.path.join('data', 'PennFudanPed', 'classes.names')}\n")

    print("Dataset created")

    os.chdir(orig_dir)


if __name__ == "__main__":
    create_PennFudanPed_dataset(root_folder=".", )
