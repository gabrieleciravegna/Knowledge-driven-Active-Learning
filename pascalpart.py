import numpy as np
import scipy.io
import os
import pickle
import torch
import tqdm

import torch.utils.data
from PIL import Image
from vision_utils import utils, my_utils

mappings = {
	'0-background': '0-background',
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
	"body": "body",
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
	"train_head": "train_head",
	"aeroplane_body": "aeroplane_body",
	"bottle_body": "bottle_body",
}

name_list = np.unique(np.asarray([v for v in mappings.values()]))
name_ids = {name: i for i, name in enumerate(name_list)}

data_folder = os.path.join("data", "PascalPart", "Annotations", "Annotations_Part")
mask_folder = os.path.join("data", "PascalPart", "Masks")
imgs_folder = os.path.join("data", "PascalPart", "Images")

img_list = os.listdir(imgs_folder)
mask_list = os.listdir(mask_folder)


def create_pascal_part_dataset(override=False):
	if os.path.exists(mask_folder):
		if not override:
			return
	else:
		os.makedirs(mask_folder)
	print("Creating Pascalpart dataset")

	pbar = tqdm.trange(len(os.listdir(data_folder)))
	for i, filename in enumerate(os.listdir(data_folder)):
		pbar.update()
		path = os.path.join(data_folder, filename)
		anno = scipy.io.loadmat(path)['anno'][0][0]
		img_name = anno[0].item()
		objs = anno[1][0]
		masks = []
		for obj in objs:
			object_name = obj[0][0]
			object_name = mappings[object_name]
			object_id = name_ids[object_name]
			object_mask = obj[2]
			mask = {
				'target_id': object_id,
				'target_name': object_name,
				'mask': object_mask
			}
			masks.append(mask.copy())
			part_list = obj[3]
			if len(part_list) > 0:
				for part in part_list[0]:
					part_name = part[0].item()
					if object_name == "train" and part_name == "head":
						part_name = "train_head"
					elif object_name == "aeroplane" and part_name == "body":
						part_name = "aeroplane_body"
					elif object_name == "bottle" and part_name == "body":
						part_name = "bottle_body"
					part_name = mappings[part_name]
					part_id = name_ids[part_name]
					part_mask = part[1]
					mask['target_id'] = part_id
					mask['target_name'] = part_name
					mask['mask'] = part_mask
					masks.append(mask.copy())
		final_masks = []
		for mask in masks:
			final_masks.append(mask['mask'] * mask['target_id'])
		final_masks = np.stack(final_masks)

		filename = os.path.join(mask_folder, img_name + "_mask.pckl")
		with open(filename, "wb") as f:
			pickle.dump(final_masks, f)
	pbar.close()

	return


def delete_not_annotated_images():
	i = 0
	for image in img_list:
		check_list = [1 for mask in mask_list if image.split(".")[0] in mask]
		if len(check_list) == 0:
			print(image.split(".")[0])
			os.remove(os.path.join(imgs_folder, image))
			i +=1
	print(f"{i} deleted images")


def check_img_mask_consistency():
	i, j = 0, 0
	for image, mask in zip(img_list, mask_list):
		if image.split(".")[0] not in mask:
			print(image, mask)
			i += 1
		else:
			j += 1
	print(f"{j} Consistent mask-image")
	print(f"{i} Inconsistent mask-image")


def check_box_consistency():
	pbar = tqdm.trange(len(mask_list))
	for mask_name in mask_list:
		pbar.update()
		mask = np.load(os.path.join(mask_folder, mask_name), allow_pickle=True)
		obj_ids = np.unique(mask)
		num_objs = len(obj_ids)
		boxes = []
		for i in range(num_objs):
			pos = np.where(mask[i] != 0)
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			boxes.append([xmin, ymin, xmax, ymax])
		boxes = torch.as_tensor(boxes)
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		assert (area >= 0.1).all(), f"Error in bounding box area, {area} {boxes}"

	pbar.close()


class PascalPartDataset(torch.utils.data.Dataset):
	def __init__(self, root, transforms=None):
		self.root = root
		self.transforms = transforms
		# load all image files, sorting them to
		# ensure that they are aligned
		self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
		self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

	def __getitem__(self, idx):
		# check image mask consistency
		image, mask = self.imgs[idx], self.masks[idx]
		assert image.split(".")[0] in mask, f"Error in loading image {image} or mask {mask}"

		# load images ad masks
		img_path = os.path.join(self.root, "Images", self.imgs[idx])
		mask_path = os.path.join(self.root, "Masks", self.masks[idx])
		try:
			img = Image.open(img_path).convert("RGB")
		except OSError as e:
			print(image)
			raise e
		masks = np.load(mask_path, allow_pickle=True)
		# instances are encoded as different channels
		num_objs = masks.shape[0]

		boxes = []
		labels = []
		for i in range(num_objs):
			pos = np.where(masks[i] !=0)
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			boxes.append([xmin, ymin, xmax, ymax])
			obj_ids = np.unique(masks[i])
			try:
				labels.append(obj_ids[obj_ids != 0].item())
			except BaseException:
				print(f"Error in getting label {obj_ids}")
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.as_tensor(labels, dtype=torch.int64)

		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		degenerated_aerea = area <= 0.1
		if degenerated_aerea.any():
			boxes = boxes[area >= 0.1, :]
			labels = labels[area >= 0.1]

		# check for crowded instances (not counted in test evaluation)
		iscrowd = torch.tensor([num_objs > 50] * num_objs, dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target

	def __len__(self):
		return len(self.masks)


def check_dataset():
	dataset = PascalPartDataset('data/PascalPart', my_utils.get_transform(train=True))
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=2, shuffle=True, num_workers=8,
		collate_fn=utils.collate_fn)
	pbar = tqdm.trange(len(data_loader))
	for i, data in enumerate(data_loader):
		pbar.update()


if __name__ == "__main__":
	# print(name_list)
	# print(name_ids)
	#
	# create_pascal_part_dataset(override=True)
	# print(os.listdir(mask_folder))
	#
	# delete_not_annotated_images()
	#
	# check_img_mask_consistency()

	# check_box_consistency()

	check_dataset()


