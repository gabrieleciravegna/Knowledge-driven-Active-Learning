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


mappings = {
    'background': 'background',
    'aeroplane': 'aeroplane',
    'bicycle': 'bicycle',
    'bird': 'bird',
    'boat': 'boat',
    'bottle': 'bottle',
    'bus': 'bus',
    'car': 'car',
    'cat': 'cat',
    'chair': 'chair',
    'cow': 'cow',
    'diningtable': 'table',
    'table': 'table',
    'dog': 'dog',
    'horse': 'horse',
    'motorbike': 'motorbike',
    'person': 'person',
    'pottedplant': 'pottedplant',
    'sheep': 'sheep',
    'sofa': 'sofa',
    'train': 'train',
    'tvmonitor': 'tvmonitor',
    'lfoot': 'foot',
    'hair': 'hair',
    'fwheel': "wheel",
    'lhand': "hand",
    'rfoot': "foot",
    'llleg': "leg",
    'chainwheel': 'chainwheel',
    'ruarm': 'arm',
    'rlarm': 'arm',
    'rlleg': 'leg',
    'rhand': 'hand',
    'llarm': 'arm',
    'luarm': 'arm',
    'luleg': 'leg',
    'saddle': 'saddle',
    'bwheel': "wheel",
    'handlebar': "handlebar",
    'ruleg': "leg",
    "head": "head",
    "torso": "torso",
    "beak": "beak",
    "rleg": "leg",
    "tail": "tail",
    "frontside": "frontside",
    "rightside": "rightside",
    "leftside": "leftside",
    'window_1': "window",
    'window_2': "window",
    'window_3': "window",
    'window_4': "window",
    'window_5': "window",
    'window_6': "window",
    'window_7': "window",
    'window_8': "window",
    'window_9': "window",
    'window_10': "window",
    'window_11': "window",
    'window_12': "window",
    'window_13': "window",
    'window_14': "window",
    'window_15': "window",
    'window_16': "window",
    'window_17': "window",
    'window_18': "window",
    'window_19': "window",
    'window_20': "window",
    'window_21': "window",
    'window_22': "window",
    'window_23': "window",
    'window_24': "window",
    'window_25': "window",
    'window_26': "window",
    'window_27': "window",
    'window_28': "window",
    'window_29': "window",
    'headlight_1': "headlight",
    'headlight_2': "headlight",
    'headlight_3': "headlight",
    'headlight_4': "headlight",
    'headlight_5': "headlight",
    'headlight_6': "headlight",
    'headlight_7': "headlight",
    'headlight_8': "headlight",
    'headlight_9': "headlight",
    "wheel_1": "wheel",
    "wheel_2": "wheel",
    "wheel_3": "wheel",
    "wheel_4": "wheel",
    "wheel_5": "wheel",
    "wheel_6": "wheel",
    "wheel_7": "wheel",
    "wheel_8": "wheel",
    "wheel_9": "wheel",
    "door_1": "door",
    "door_2": "door",
    "door_3": "door",
    "door_4": "door",
    "fliplate": "plate",
    "bliplate": "plate",
    'rightmirror': "mirror",
    'leftmirror': "mirror",
    "lear": "ear",
    "rear": "ear",
    "leye": "eye",
    "reye": "eye",
    "lebrow": "ebrow",
    "rebrow": "ebrow",
    "mouth": "mouth",
    "nose": "nose",
    "neck": "neck",
    "pot": "pot",
    "plant": "plant",
    "cap": "cap",
    "lwing": "wing",
    "rwing": "wing",
    "muzzle": "muzzle",
    "lfpa": "paw",
    "rfpa": "paw",
    "rbpa": "paw",
    "lbpa": "paw",
    'lfleg': 'leg',
    'rfleg': 'leg',
    'rbleg': 'leg',
    'lbleg': 'leg',
    'lleg': 'leg',
    'screen': 'screen',
    'coach_1': 'coach',
    'coach_2': 'coach',
    'coach_3': 'coach',
    'coach_4': 'coach',
    'coach_5': 'coach',
    'coach_6': 'coach',
    'coach_7': 'coach',
    'coach_8': 'coach',
    'coach_9': 'coach',
    'lflleg': 'leg',
    'lfuleg': 'leg',
    'rflleg': 'leg',
    'lblleg': 'leg',
    'lbuleg': 'leg',
    'rfuleg': 'leg',
    'rbuleg': 'leg',
    'rblleg': 'leg',
    'lfho': 'hoof',
    'rfho': 'hoof',
    'lbho': 'hoof',
    'rbho': 'hoof',
    'stern': 'stern',
    'engine_1': 'engine',
    'engine_2': 'engine',
    'engine_3': 'engine',
    'engine_4': 'engine',
    'engine_5': 'engine',
    'engine_6': 'engine',
    'engine_7': 'engine',
    'engine_8': 'engine',
    'engine_9': 'engine',
    'cleftside_1': "leftside",
    'cleftside_2': "leftside",
    'cleftside_3': "leftside",
    'cleftside_4': "leftside",
    'cleftside_5': "leftside",
    'cleftside_6': "leftside",
    'cleftside_7': "leftside",
    'cleftside_8': "leftside",
    'cleftside_9': "leftside",
    'crightside_1': "rightside",
    'crightside_2': "rightside",
    'crightside_3': "rightside",
    'crightside_4': "rightside",
    'crightside_5': "rightside",
    'crightside_6': "rightside",
    'crightside_7': "rightside",
    'crightside_8': "rightside",
    'crightside_9': "rightside",
    'cfrontside_1': "frontside",
    'cfrontside_2': "frontside",
    'cfrontside_3': "frontside",
    'cfrontside_4': "frontside",
    'cfrontside_5': "frontside",
    'cfrontside_6': "frontside",
    'cfrontside_7': "frontside",
    'cfrontside_8': "frontside",
    'cfrontside_9': "frontside",
    'hfrontside': "frontside",
    'hleftside': "leftside",
    'hrightside': "rightside",
    'hroofside': "roofside",
    'hbackside': "backside",
    'backside': "backside",
    "croofside_1": "roofside",
    "roofside": "roofside",
    "croofside_2": "roofside",
    "croofside_3": "roofside",
    "croofside_4": "roofside",
    "croofside_5": "roofside",
    "croofside_6": "roofside",
    "croofside_7": "roofside",
    "croofside_8": "roofside",
    "croofside_9": "roofside",
    "lhorn": "horn",
    "rhorn": "horn",
    "cbackside_1": "backside",
    "cbackside_2": "backside",
    "cbackside_3": "backside",
    "cbackside_4": "backside",
    "cbackside_5": "backside",
    "cbackside_6": "backside",
    "cbackside_7": "backside",
    "cbackside_8": "backside",
    "cbackside_9": "backside",
    "body": "body",
}

main_classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

# name_list = np.unique(np.asarray([v for v in mappings.values()]))
# name_ids = {name: it for it, name in enumerate(name_list)}
# n_classes = len(name_list)

# bad_targets_idx = [125,   154,   181,   287,   396,   400,   447,   494,   935,  1124,  1249,  1319,  2028,  2046,
#                    2374,  2619,  2661,  2722,  2924,  2927,  2974,  3107,  3457,  3558,  3680,  3790,  4192,  4335,
#                    4351,  4530,  4700,  4860,  4902,  5016,  5017,  5250,  5488,  5588,  5789,  6039,  6066,  6101,
#                    6108,  6648, 6773,  6792,  6874,  7142,  7327,  7346,  7355,  7449,  7772,  7964,  8231,  8248,
#                    8450,  8755,  8848,  9111,  9535, 10036]
bad_targets_idx = [125, 154, 181, 287, 396, 400, 447, 494, 935, 1124, 1249, 1319, 2028, 2046, 2374, 2619, 2661, 2722,
                   2924, 2927, 2974, 3107, 3457, 3558, 3680, 3790, 4192, 4335, 4351, 4530, 4700, 4860, 4902, 5016,
                   5017, 5250, 5458, 5488, 5588, 5789, 6039, 6066, 6101, 6108, 6648, 6773, 6792, 6874, 7142, 7327,
                   7346, 7355, 7449, 7772, 7964, 8231, 8248, 8450, 8755, 8848, 9111, 9535, 10036]

bad_classes = ['person_ear', 'person_eye', 'person_ebrow', 'person_mouth', 'horse_hoof', 'cow_eye', 'cow_neck',
               'cow_tail', 'dog_eye', 'aeroplane_wheel', 'bottle_cap', 'car_mirror', 'car_plate', 'bus_plate',
               'bus_mirror', 'bus_headlight', 'train_headlight', 'bird_eye', 'bird_beak', 'bird_foot', 'cat_eye',
               'cat_nose', 'horse_eye', 'sheep_ear', 'sheep_eye', 'sheep_muzzle', 'sheep_leg', 'sheep_tail',
               'motorbike_headlight', 'aeroplane_tail', 'bus_backside', 'motorbike_handlebar', 'cow_horn',
               'train_backside', 'bus_roofside', 'train_roofside', 'sheep_horn', 'motorbike_saddle', 'bicycle_headlight']
len(bad_classes)

dataset_folder = "PascalPartYolo"
orig_image_folder = "JPEGImages"
orig_label_folder = "Annotations_Part"
image_folder = "images"
label_folder = "labels"

label_url = "http://roozbehm.info/pascal-parts/trainval.tar.gz"
image_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
label_path = "trainval.tar.gz"
image_path = "VOCtrainval_03-May-2010.tar"
names_path = "classes.names"
combinations_path = "combinations.npy"

config_file = "PascalPart.data"
n_samples = 10103
class_zero = 11111


def create_data_config_file(file: str, n_classes):
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write(f"classes={n_classes}\n")
            f.write(f"train={os.path.join('data', 'PascalPart', 'train.txt')}\n")
            f.write(f"valid={os.path.join('data', 'PascalPart', 'test.txt')}\n")
            f.write(f"names={os.path.join('data', 'PascalPart', 'classes.names')}\n")


def create_pascal_part_dataset(root_folder=".",
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
        shutil.move(os.path.join("Images", "VOCdevkit", "VOC2010", orig_image_folder),
                    orig_image_folder)
        shutil.rmtree("Images")
    print("Images extracted")

    # Creating object labels
    print("Creating the labels...")

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
        areas = []
        mask_per_class = {}
        # Ensuring that the main classes are enumerated first
        names_ids = {c: idx for idx, c in enumerate(main_classes)}
        data_idx = sorted(os.listdir(orig_label_folder))
        for idx, filename in enumerate(tqdm.tqdm(data_idx)):
            path = os.path.join(orig_label_folder, filename)
            anno = io.loadmat(path)['anno'][0][0]
            objs = anno[1][0]
            masks = []
            img_name = anno[0].item()

            # Extracting object first
            for obj in objs:
                object_name = obj[0][0]
                object_name = mappings[object_name]

                # get object id
                if object_name in names_ids:
                    object_id = names_ids[object_name]
                elif object_name not in bad_classes:
                    object_id = len(names_ids)
                    names_ids[object_name] = object_id
                else:
                    continue

                # create mask
                if object_id == 0:
                    masks.append(obj[2] * class_zero)
                else:
                    masks.append(obj[2] * object_id)

                # extract parts
                part_list = obj[3]
                if len(part_list) > 0:
                    for part in part_list[0]:
                        part_name = part[0].item()
                        part_name = mappings[part_name]
                        part_name = object_name + "_" + part_name
                        if part_name in names_ids:
                            part_id = names_ids[part_name]
                        elif part_name not in bad_classes:
                            part_id = len(names_ids)
                            names_ids[part_name] = part_id
                        else:
                            continue
                        masks.append(part[1] * part_id)

            # Moving and renaming the images
            orig_img_name = os.path.join(orig_image_folder, f"{img_name}.jpg")
            img_name = os.path.join(image_folder, f"{idx:05d}.png")
            l_path = os.path.join(label_folder, f"{idx:05d}.txt")
            shutil.copy(orig_img_name, img_name)

            # Writing the object detection label files
            with open(l_path, "w") as f:
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
                    if class_idx in mask_per_class:
                        mask_per_class[class_idx] += 1
                    else:
                        mask_per_class[class_idx] = 1

                    assert 0 < x_center <= 1 and \
                           0 < y_center <= 1 and \
                           0 < width <= 1 and \
                           0 < height <= 1, f"Error in creating mask " \
                                            f"{x_center, y_center} {width, height}"
                    f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")

        print(f"Discarding {discarded} label with small areas")
        areas = np.asarray(areas)
        logbins = np.geomspace(np.min(areas), np.max(areas), 12)
        plt.hist(areas, bins=logbins)
        plt.xscale("log")
        plt.savefig("Object areas distribution.png")
        plt.show()

        mask_per_class = OrderedDict(sorted(mask_per_class.items()))
        print(f"Masks distributions", mask_per_class)
        plt.bar(mask_per_class.keys(), mask_per_class.values())
        plt.yscale("log")
        plt.savefig("Masks distributions per class.png")
        plt.show()

        mask_per_class = [(0, 661), (1, 579), (2, 826), (3, 560), (4, 650), (5, 474), (6, 1266), (7, 1125), (8, 2093), (9, 413), (11, 1390), (12, 598), (13, 574), (14, 6783), (15, 635), (16, 580), (17, 547), (18, 503), (19, 631), (20, 492), (21, 485), (22, 5216), (23, 5362), (24, 3895), (25, 83), (26, 20), (27, 21), (28, 89), (29, 2725), (30, 167), (31, 866), (32, 6814), (33, 428), (34, 102), (35, 222), (36, 552), (37, 341), (38, 783), (39, 1), (40, 181), (41, 264), (42, 120), (43, 1), (44, 143), (45, 362), (46, 61), (47, 229), (48, 39), (49, 625), (50, 1215), (51, 1047), (52, 1275), (53, 393), (54, 1861), (55, 385), (56, 31), (57, 783), (58, 142), (59, 292), (60, 562), (61, 615), (62, 23), (63, 473), (64, 64), (65, 1436), (66, 583), (67, 438), (68, 511), (69, 22), (70, 158), (71, 672), (72, 852), (73, 85), (74, 126), (75, 308), (76, 455), (77, 6), (78, 313), (79, 196), (80, 149), (81, 33), (82, 4), (83, 867), (84, 307), (85, 442), (86, 829), (87, 121), (88, 110), (89, 276), (90, 608), (91, 285), (92, 325), (93, 5), (94, 296), (95, 390), (96, 6), (97, 97), (98, 688), (99, 169), (100, 241), (101, 59), (102, 254), (103, 1053), (104, 1007), (105, 93), (106, 58), (107, 1079), (108, 584), (109, 1532), (110, 437), (111, 121), (112, 2), (113, 729), (114, 263), (115, 84), (116, 2), (117, 88), (118, 502), (119, 153), (120, 96), (121, 25), (122, 380), (123, 65), (124, 340), (125, 206), (126, 348), (127, 571), (128, 245), (129, 1), (130, 60), (131, 22), (132, 44), (133, 29), (134, 21), (135, 71), (136, 31), (137, 15), (138, 1)]
        class_to_remove = [class_name for class_name, freq in mask_per_class if freq < 10]
        print("Class to remove", class_to_remove)

        # Saving names_ids sorted by id
        with open(names_path, "w") as f:
            for name, id in sorted(names_ids.items(),
                               key=lambda item: item[1]):
                f.write(f"{name}\n")
        print("Class name file created")

        # Saving combinations
        n_classes = len(main_classes)
        n_attributes = len(names_ids) - n_classes
        combinations = np.zeros((len(main_classes), n_attributes))
        for name, idx in names_ids:
            if name not in main_classes:
                attr_id = idx - n_classes
                class_name = name.split("_")[0]
                class_id = names_ids[class_name]
                combinations[class_id, attr_id] = 1
        combinations.dump(combinations_path)
        print("Class attribute combinations file created")

    # Reading names_ids
    with open(names_path, "w") as f:
        names_ids = []
        for name in f.readlines():
            names_ids.append(name)
    print("Class name file read")

    n_classes = len(names_ids)
    create_data_config_file(config_file, n_classes)

    os.chdir(orig_dir)

    return os.path.join(root_folder, dataset_folder, config_file)


class PascalPartDatasetYolo(Dataset):

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
    dataset = PascalPartDatasetYolo(dataset_folder,
                                    my_utils.get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=vis_utils.collate_fn)
    pbar = tqdm.trange(len(data_loader))
    for i, data in enumerate(data_loader):
        pbar.update()


if __name__ == "__main__":
    create_pascal_part_dataset()
    check_dataset()
