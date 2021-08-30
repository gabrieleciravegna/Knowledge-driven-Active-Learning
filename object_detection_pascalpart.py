import numpy as np
import tqdm

from pascalpart import PascalPartDataset, name_ids
from vision_utils import utils, my_utils
import torch
from vision_utils.engine import train_one_epoch, evaluate

dataset = PascalPartDataset('data/PascalPart', my_utils.get_transform(train=True))
dataset_test = PascalPartDataset('data/PascalPart/', my_utils.get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
n_train = int(np.round(len(indices) * 1 / 100))
dataset = torch.utils.data.Subset(dataset, indices[:n_train])
dataset_test = torch.utils.data.Subset(dataset_test, indices[n_train:n_train*2])  # TODO: remove n_train*2

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=8,
    collate_fn=utils.collate_fn)

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# our dataset has 66 classes
num_classes = len(name_ids)

# get the model using our helper function
model = my_utils.get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)


# let's train it for 10 epochs
num_epochs = 200
pbar = tqdm.trange(num_epochs, ncols=100)
for epoch in range(num_epochs):
    print("")
    # train for one epoch, printing every 10 iterations
    metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, verbose=True)
    loss = metrics.meters['loss'].global_avg

    if epoch % 10 == 0:
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the train dataset
        coco_eval = evaluate(model, data_loader, device=device, verbose=True)
        mAP = coco_eval.coco_eval['bbox'].stats[0]

        # evaluate on the test dataset
        coco_eval_test = evaluate(model, data_loader_test, device=device, verbose=True)
        mAP_test = coco_eval_test.coco_eval['bbox'].stats[0]

        pbar.set_description(f"Train l {loss:.2f}, Train mAP {mAP:.3f}, Test mAP {mAP_test:.3f}")
        pbar.update(10)
