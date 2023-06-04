import os
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

import torch.utils.data

from PIL import Image, ImageFile

from tqdm import tqdm

from dataset import DetectionDataset, class2sym

DEVICE =  'cuda' if torch.cuda.is_available() else 'cpu'

LR = 0.005
MOMENTUM = 0.9
WD = 0.0005
GAMMA = 0.1
N_EPOCH = 100


def create_model(pretrained):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    return model


def build_target(image, bboxes, labels):
    transform = T.Compose([T.ColorJitter(0.3, 0.5, 0.5, 0.1),
                            T.ToTensor(),
                            T.Normalize(0, 255)])

    images = transform(image).to(DEVICE)

    true_labels = torch.ones(labels.size()).to(DEVICE)
    
    images = images.reshape(1, 3, 1024, 1024)
    bboxes = bboxes.reshape(1, bboxes.shape[0], -1)

    bboxes[:,:,0] = bboxes[:,:,0] - bboxes[:,:,3] 
    bboxes[:,:,3] = bboxes[:,:,0] + bboxes[:,:,3]
    bboxes[:,:,1] = bboxes[:,:,1] + bboxes[:,:,2]
    temp = torch.zeros(bboxes.shape[0], bboxes.shape[1], 1)
    #temp = bboxes[:,:,1]
    #bboxes[:,:,1] = bboxes[:,:,2]
    #bboxes[:,:,2] = temp
    bboxes[bboxes < 0] = 0.

    targets = []

    for i in range(bboxes.shape[0]):
        targets.append({'boxes': bboxes[i], 
                          'labels': true_labels[i]})

    #print(images.shape)
    return images, targets
    

def train_one_epoch(model, optimizer, scheduler, dataloader):
    running_losses = []
    print(len(dataloader))
    for image, bboxes, labels in tqdm(iter(dataloader)):
        images, targets = build_target(image, torch.Tensor(bboxes).to(DEVICE), torch.LongTensor(labels))
        try:
            predict_loss = model(images, targets)

            loss = sum(loss for loss in predict_loss.values())
            running_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Good")
        except:
            #print('Cropped')
            pass

    return running_losses


def train(model, optimizer, scheduler, dataloader, n_epoch):
    running_losses = []
    for i in range(n_epoch):
        epoch_losses = train_one_epoch(model, optimizer, scheduler, dataloader)
        running_losses.extend(epoch_losses)
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

    dataloader = DetectionDataset('train', max_size=2048, transforms=T.Compose([T.ColorJitter(0.3, 0.5, 0.5, 0.1)]))

    model = create_model(False)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=GAMMA)

    train(model, optimizer, scheduler, dataloader, N_EPOCH)
    
    file_name = 'checkpoints/fast-crnn/' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.path'
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/fast-crnn", exist_ok=True)
    torch.save(model, file_name)