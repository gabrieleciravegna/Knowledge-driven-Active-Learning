import os
import shutil
import warnings
from collections import OrderedDict

import numpy as np
import torch
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from scipy import io
from sortedcontainers import SortedSet
from torch.utils.data import Dataset


from pytorchyolo import xywh2xyxy_np
import vis_utils
import my_utils


def download(url, file_path):
    print("Downloading %s" % file_path)
    import requests
    response = requests.get(url, allow_redirects=True, stream=True)
    total_length = response.headers.get('content-length')

    with open(file_path, "wb") as f:
        pbar = tqdm.tqdm(total=int(total_length))
        for data in response.iter_content(chunk_size=4096):
            f.write(data)
            pbar.update(len(data))
    pbar.close()


dog_mappings = {
    "head": "dog_head",
    "lbleg": "dog_leg",
    "lbpa": "dog_paw",
    "lear": "dog_ear",
    "leye": "dog_eye",
    "lfleg": "dog_leg",
    "lfpa": "dog_paw",
    "muzzle": "dog_muzzle",
    "neck": "dog_neck",
    "nose": "dog_nose",
    "rbleg": "dog_leg",
    "rbpa": "dog_paw",
    "rear": "dog_ear",
    "reye": "dog_eye",
    "rfleg": "dog_leg",
    "rfpa": "dog_paw",
    "tail": "dog_tail",
    "torso": "dog_torso"
}

person_mappings = {
    "hair": "person_hair",
    "head": "person_head",
    "lear": "person_ear",
    "leye": "person_eye",
    "lebrow": "person_ebrow",
    "lfoot": "person_foot",
    "lhand": "person_hand",
    "llarm": "person_arm",
    "llleg": "person_leg",
    "luarm": "person_arm",
    "luleg": "person_leg",
    "mouth": "person_mouth",
    "neck": "person_neck",
    "nose": "person_nose",
    "rear": "person_ear",
    "reye": "person_eye",
    "rebrow": "person_ebrow",
    "rfoot": "person_foot",
    "rhand": "person_hand",
    "rlarm": "person_arm",
    "rlleg": "person_leg",
    "ruarm": "person_arm",
    "ruleg": "person_leg",
    "torso": "person_torso",
}

# Classes with < 100 objects after filtering those with areas < 1% of img area
bad_classes = ['dog_eye', 'person_ear', 'person_ebrow', 'person_eye', 'person_mouth']

# name_list = np.unique(np.asarray([v for v in mappings.values()]))
name_list = ["dog", "person"]
name_list += [v for v in dog_mappings.values() if v not in name_list and v not in bad_classes]
name_list += [v for v in person_mappings.values() if v not in name_list and v not in bad_classes]
name_list = np.unique(np.asarray(name_list))

name_ids = {name: i for i, name in enumerate(name_list)}

bad_targets_idx = [
    11,   20,   22,   29,   35,   46,   52,   60,   78,   79,   86,   95,
    121,  131,  139,  144,  149,  154,  203,  204,  239,  263,  277,  279,
    284,  287,  295,  298,  303,  304,  316,  323,  329,  335,  341,  360,
    363,  367,  384,  408,  420,  484,  490,  540,  552,  561,  563,  564,
    573,  584,  595,  627,  718,  730,  731,  734,  754,  761,  795,  812,
    815,  829,  874,  941,  944,  946,  973,  985,  988,  989, 1001, 1009,
    1017, 1021, 1034, 1087, 1091, 1103, 1111, 1112, 1120, 1138, 1139, 1159,
    1166, 1215, 1219, 1224, 1225, 1230, 1251, 1260, 1263, 1281, 1283, 1285,
    1305, 1308, 1312, 1366, 1375, 1385, 1391, 1412, 1431, 1434, 1452, 1454,
    1458, 1459, 1469, 1476, 1479, 1492, 1496, 1518, 1536, 1549, 1554, 1558,
    1562, 1578, 1585, 1588, 1590, 1596, 1605, 1611, 1651, 1663, 1711, 1712,
    1714, 1725, 1741, 1747, 1749, 1755, 1764, 1774, 1790, 1792, 1808, 1812,
    1823, 1825, 1849, 1855, 1857, 1863, 1891, 1900, 1929, 1930, 1933, 1934,
    1940, 1947, 1950, 1951, 1962, 1964, 1967, 1975, 1985, 1987, 1993, 1999,
    2011, 2015, 2020, 2021, 2022, 2023, 2025, 2035, 2041, 2044, 2065, 2066,
    2068, 2072, 2075, 2091, 2096, 2102, 2111, 2115, 2121, 2124, 2133, 2135,
    2143, 2146, 2152, 2154, 2157, 2165, 2169, 2178, 2184, 2185, 2201, 2206,
    2214, 2218, 2226, 2228, 2230, 2234, 2237, 2239, 2255, 2259, 2277, 2286,
    2296, 2300, 2303, 2308, 2309, 2320, 2353, 2354, 2357, 2360, 2361, 2400,
    2405, 2409, 2425, 2435, 2458, 2462, 2473, 2474, 2480, 2499, 2506, 2527,
    2539, 2550, 2552, 2554, 2555, 2556, 2561, 2562, 2574, 2575, 2578, 2579,
    2587, 2593, 2595, 2602, 2621, 2639, 2645, 2659, 2669, 2679, 2702, 2717,
    2738, 2741, 2791, 2799, 2800, 2806, 2808, 2822, 2835, 2838, 2840, 2848,
    2855, 2861, 2908, 2934, 2937, 2945, 2951, 2967, 2981, 2983, 2996, 2998,
    3014, 3023, 3024, 3033, 3036, 3041, 3044, 3057, 3061, 3062, 3084, 3095,
    3114, 3132, 3136, 3158, 3166, 3189, 3237, 3239, 3250, 3253, 3258, 3296,
    3304, 3308, 3318, 3328, 3334, 3348, 3362, 3365, 3366, 3379, 3388, 3390,
    3395, 3406, 3408, 3421, 3425, 3444, 3452, 3456, 3470, 3481, 3489, 3490,
    3495, 3522, 3528, 3534, 3561, 3571, 3591, 3603, 3605, 3635, 3638, 3656,
    3659, 3681, 3694, 3698, 3735, 3738, 3758, 3761, 3764, 3782, 3796, 3802,
    3813, 3814, 3829, 3838, 3841, 3865, 3866, 3871, 3872, 3876, 3928, 3933,
    3939, 3964, 3978, 3987, 4001, 4005, 4011, 4014, 4042, 4052, 4059, 4062,
    4066, 4071, 4075, 4078, 4121, 4133, 4137, 4141, 4148, 4175, 4182, 4183,
    4185, 4187, 4194, 4195, 4215, 4239, 4243, 4267, 4276, 4289, 4291
]


dataset_folder = "DogvsPerson"
orig_image_folder = "JPEGImages"
orig_label_folder = "Annotations_Part"
image_folder = "images"
label_folder = "labels"

label_url = "http://roozbehm.info/pascal-parts/trainval.tar.gz"
image_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
label_path = "trainval.tar.gz"
image_path = "VOCtrainval_03-May-2010.tar"

config_file = f"{dataset_folder}.data"
n_classes = len(name_list)
n_samples = 4304
class_zero = 11111


def create_data_config_file(file: str):
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write(f"classes={n_classes}\n")
            f.write(f"train={os.path.join('data', dataset_folder, 'train.txt')}\n")
            f.write(f"valid={os.path.join('data', dataset_folder, 'test.txt')}\n")
            f.write(f"names={os.path.join('data', dataset_folder, 'classes.names')}\n")


def create_train_test_split(train_idx=None, test_idx=None, root="."):
    dataset_files = sorted(os.listdir(os.path.join(root, image_folder)))
    dataset_len = len(dataset_files)
    # filtering files without incorrect labels
    dataset_files = [file for i, file in enumerate(dataset_files)
                     if file not in bad_targets_idx]
    if train_idx is None or test_idx is None:
        test_size = 0.1
        test_len = int(dataset_len * test_size)
        test_files = SortedSet(np.random.choice(dataset_files,
                                                size=test_len,
                                                replace=False))
        train_files = SortedSet(dataset_files) - test_files
        print(f"Created train: {dataset_len - test_len} "
              f"test: {test_len} splits")
    else:
        train_files = [dataset_files[idx] for idx in train_idx]
        test_files = [dataset_files[idx] for idx in test_idx]

    for split, split_files in zip(["train", "test"], [train_files, test_files]):
        file = f"{split}.txt"
        with open(file, "w") as f:
            for file in split_files:
                img_path = os.path.join("data", dataset_folder, "images", f"{file}")
                f.write(f"{img_path}\n")
    print("Train test files created")


def create_dogvsperson_dataset(root_folder=".",
                                 train_idx=None,
                                 test_idx=None,
                                 override=False,
                                 ):
    orig_dir = os.path.abspath(os.curdir)
    os.chdir(root_folder)

    # Check if folder exists and it is not empty
    if override and os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    os.chdir(dataset_folder)

    # Check if labels already downloaded
    if not os.path.exists(orig_label_folder):
        if not os.path.exists(label_path):
            download(label_url, label_path)
            print("Labels Downloaded")
        shutil.unpack_archive(label_path, "Labels")

        # Cleaning label files
        shutil.move(os.path.join("Labels", orig_label_folder), ".")
        shutil.rmtree("Labels")
    print("Labels extracted")

    # Check if data already downloaded
    if not os.path.exists(orig_image_folder):
        if not os.path.exists(image_path):
            download(image_url, image_path)
            print("Images Downloaded")
        shutil.unpack_archive(image_path, "Images")
        shutil.move(os.path.join("Images", "VOCdevkit", "VOC2010",
                                 orig_image_folder), orig_image_folder)
        shutil.rmtree("Images")
    print("Images extracted")

    if not os.path.exists("classes.names"):
        with open("classes.names", "w") as f:
            for c in name_list:
                f.write(f"{c}\n")
    print("Class name file created")

    # Creating object labels
    print("Creating the labels...")

    mask_per_class = {}
    if not os.path.exists(label_folder) or \
            not os.path.exists(image_folder) or \
            len(os.listdir(label_folder)) < n_samples or \
            len(os.listdir(image_folder)) < n_samples:

        if os.path.exists(label_folder):
            shutil.rmtree(label_folder)
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)

        os.makedirs(label_folder)
        os.makedirs(image_folder)

        # Reading the mask
        discarded = 0
        idx = -1
        areas = []
        data_idx = sorted(os.listdir(orig_label_folder))
        for filename in tqdm.tqdm(data_idx):
            path = os.path.join(orig_label_folder, filename)
            anno = io.loadmat(path)['anno'][0][0]
            objs = anno[1][0]
            masks = []
            img_name = anno[0].item()
            for obj in objs:
                object_name = obj[0][0]
                if object_name not in name_list:
                    continue
                object_id = name_ids[object_name]
                if object_id == 0:
                    object_id = class_zero
                masks.append(obj[2] * object_id)
                part_list = obj[3]
                if len(part_list) > 0:
                    for part in part_list[0]:
                        part_name = part[0].item()
                        if object_name == "person":
                            part_name = person_mappings[part_name]
                        elif object_name == "dog":
                            part_name = dog_mappings[part_name]
                        else:
                            raise NotImplementedError()
                        if part_name not in name_list:
                            continue
                        part_id = name_ids[part_name]
                        masks.append(part[1] * part_id)

            labels = []
            for mask in masks:
                pos = np.where(mask)
                assert len(pos[0]) > 0, "Error in reading masks"

                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                img_width = mask.shape[1]
                img_height = mask.shape[0]
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = np.abs(xmax - xmin) / img_width
                height = np.abs(ymax - ymin) / img_height
                area = width * height

                if area < 0.01:
                    discarded += 1
                    continue
                areas.append(area)

                class_idx = np.max(mask)
                if class_idx == class_zero:
                    class_idx = 0
                name = name_list[class_idx]
                if name in mask_per_class:
                    mask_per_class[name] += 1
                else:
                    mask_per_class[name] = 1

                assert 0 < x_center <= 1 and \
                       0 < y_center <= 1 and \
                       0 < width <= 1 and \
                       0 < height <= 1, f"Error in creating mask " \
                                        f"{x_center, y_center} {width, height}"
                labels.append(f"{class_idx} {x_center} {y_center} {width} {height}")

            # Check if any Person or Dog are present in the image, otherwise skip
            if len(labels) == 0:
                continue
            else:
                idx += 1

            # Moving and renaming the images
            orig_img_name = os.path.join(orig_image_folder, f"{img_name}.jpg")
            img_name = os.path.join(image_folder, f"{idx:05d}.png")
            l_path = os.path.join(label_folder, f"{idx:05d}.txt")
            shutil.copy(orig_img_name, img_name)

            # Writing the object detection label files
            with open(l_path, "w") as f:
                for label in labels:
                    f.write(f"{label}\n")

        # print(f"Discarding {discarded} label with small areas")
        # areas = np.asarray(areas)
        # logbins = np.geomspace(np.min(areas), np.max(areas), 12)
        # plt.hist(areas, bins=logbins)
        # plt.xscale("log")
        # plt.savefig("Object areas distribution.png")
        # plt.show()

    else:
        dataset = DogvsPersonDataset(".")
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  collate_fn=vis_utils.collate_fn)
        for _, target in data_loader:
            class_idx = target[0]['labels']
            for idx in class_idx:
                # labels takes into account also the background, name_list does not
                name = name_list[idx - 1]
                if name in mask_per_class:
                    mask_per_class[name] += 1
                else:
                    mask_per_class[name] = 1

        mask_per_class = OrderedDict(sorted(mask_per_class.items()))
        print(f"Masks distributions", mask_per_class)
        plt.rcParams.update({'font.size': 12})
        plt.bar(mask_per_class.keys(), mask_per_class.values())
        plt.yscale("log")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.ylim([8*10, 20*1000])
        plt.savefig("Masks distributions per class.png")
        plt.show()

    create_train_test_split(train_idx, test_idx)

    create_data_config_file(config_file)

    os.chdir(orig_dir)

    return os.path.join(root_folder, dataset_folder, config_file)


class DogvsPersonDataset(Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

        for idx in sorted(bad_targets_idx, reverse=True):
            self.imgs.pop(idx)
            self.labels.pop(idx)

    def __getitem__(self, idx):
        # check image mask consistency
        image = self.imgs[idx]  # , self.masks[idx]
        # assert image.split(".")[0] in mask, f"Error in loading image {image} or mask {mask}"

        # load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError as e:
            print(image)
            raise e

        l_path = os.path.join(self.root, "labels", self.labels[idx])
        try:
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(l_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{l_path}'.")
            return

        labels = boxes[:, 0] + 1  # Adding background label 0
        boxes = boxes[:, 1:5]
        boxes = xywh2xyxy_np(boxes)
        boxes[:, [0, 2]] *= img.width
        boxes[:, [1, 3]] *= img.height

        assert (boxes[:, [0, 2]] < img.width).all() and \
               (boxes[:, [1, 3]] < img.height).all(), "Error in loading label"

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        degenerated_aerea = area <= 0.1
        if degenerated_aerea.any():
            boxes = boxes[area >= 0.1, :]
            labels = labels[area >= 0.1]

        # check for crowded instances (not counted in test evaluation)
        num_objs = boxes.shape[0]
        iscrowd = torch.tensor([num_objs > 50] * num_objs, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def check_dataset():
    print("Checking the dataset")
    dataset = DogvsPersonDataset(dataset_folder, my_utils.get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=vis_utils.collate_fn)
    pbar = tqdm.trange(len(data_loader))
    for _, _ in enumerate(data_loader):
        pbar.update()
    print("Dataset loaded correctly!")


if __name__ == "__main__":
    create_dogvsperson_dataset()
    check_dataset()
