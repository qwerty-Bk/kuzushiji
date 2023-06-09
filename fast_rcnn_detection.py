import os
from datetime import datetime

import pickle

import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.rpn import AnchorGenerator

import torch.utils.data

from PIL import Image, ImageFile

from tqdm import tqdm

from dataset import DetectionDataset, class2sym

DEVICE =  'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'

LR = 0.0005
MOMENTUM = 0.9
WD = 0.005
GAMMA = 0.1
N_EPOCH = 45


def create_model(pretrained):
#   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=pretrained)

    anchor_sizes = [12, 24, 32, 64, 96]
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=pretrained,
        rpn_anchor_generator=AnchorGenerator(
            sizes=tuple((s,) for s in anchor_sizes),
            aspect_ratios=tuple((0.5, 1.0, 2.0) for _ in anchor_sizes),
        ),
        box_detections_per_img=1000,
        box_nms_thresh=0.25,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    #model = torch.load("./2_checkpoints/fast-crnn/2023-06-06-06-48-11_epoch_44.path")

    return model


def build_target(image, bboxes, labels):
    images = image.to(DEVICE)
    labels = labels.to(DEVICE)
    
    bboxes = bboxes.type(torch.FloatTensor).to(DEVICE)

    true_labels = torch.ones(labels.size()).to(DEVICE)
    
    #images = images.reshape(1, 3, 512, 512)
    #bboxes = bboxes.reshape(1, bboxes.shape[0], -1)
    #labels = labels.reshape(1,labels.shape[0], -1)
    
    bboxes[:,:,2] = bboxes[:,:,0] + bboxes[:,:,2]
    bboxes[:,:,3] = bboxes[:,:,1] + bboxes[:,:,3]
    
    
    bboxes[bboxes < 0] = 0.
    mask_bad = torch.zeros(bboxes.shape[0], bboxes.shape[1], bboxes.shape[2]).to(DEVICE)
    mask_bad[:,:,2] = 1
    mask_bad[:,:,3] = 1

    bboxes += mask_bad * (bboxes < 0.001) * 0.001
    targets = []
    #print(labels)
    labels[labels != 0] = 1
    #print(labels)

    for i in range(bboxes.shape[0]):
        targets.append({'boxes': bboxes[i], 
                          'labels': labels[i].reshape(-1)})
    #print(targets)
    return images, targets


def train_one_epoch(model, optimizer, scheduler, dataloader):
    running_losses = []
    print(len(dataloader))
    #for image, bboxes, labels in tqdm(iter(dataloader)):
    pcs = tqdm(enumerate(dataloader))
    for batch_idx, (image, bboxes, labels) in pcs:
        
        images, targets = build_target(image, torch.Tensor(bboxes).to(DEVICE), torch.LongTensor(labels))
        
        predict_loss = model(images, targets)

        loss = sum(loss for loss in predict_loss.values())
        #print(loss)

        pcs.set_description(f"Loss {loss}")

        running_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx % 40) == 0:
            print(loss)

    return running_losses


def train(model, optimizer, scheduler, dataloader, n_epoch):
    running_losses = []
    for i in range(n_epoch):
        print('Starting epoch', i)
        epoch_losses = train_one_epoch(model, optimizer, scheduler, dataloader)
        running_losses.extend(epoch_losses)

        file_name = '6_checkpoints/fast-crnn/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_epoch_' + str(i + 45) + '.path'
        os.makedirs("6_checkpoints", exist_ok=True)
        os.makedirs("6_checkpoints/fast-crnn", exist_ok=True)
        torch.save(model, file_name)

    return running_losses


def evaluate(model, dataloader):
    running_losses = []

    for image, bboxes, labels in iter(dataloader):
        images = image.to(DEVICE)
        images, targets = build_target(image, torch.Tensor(bboxes).to(DEVICE), torch.LongTensor(labels))

        predict_loss = model(images, targets)

        loss = sum(loss for loss in predict_loss.values())
        running_losses.append(loss.item())

    return running_losses


if __name__ == '__main__':

    print(DEVICE)

    transform = T.Compose([T.ToTensor()])

    #dataloader = DetectionDataset('train', max_size=2048, crop_size=(512, 512), transforms=transform)
    dataset = DetectionDataset('train', max_size=2048, crop_size=(512, 512), transforms=transform, max_labels=300)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    model = create_model(None).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=GAMMA)

    final_losses = train(model, optimizer, scheduler, dataloader, N_EPOCH)


    with open("6_losses", "wb") as fp:   #Pickling
        pickle.dump(final_losses, fp)

    with open("6_file_losses.txt", 'w') as f:
        for s in final_losses:
            f.write(str(s) + '\n')
    
    os.makedirs("6_checkpoints", exist_ok=True)
    os.makedirs("6_checkpoints/fast-crnn", exist_ok=True)
    file_name = '6_checkpoints/fast-crnn/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_45_' + '.path'
    torch.save(model, file_name)