import os

import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

import torch.utils.data

from PIL import Image, ImageFile

from tqdm import tqdm

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


def build_target(bboxes, labels):
    true_labels = torch.ones(labels.size())

    targets = []
    for i in range(b_data.shape[0]):
        b_targets.append({'boxes': b_label[i], 
                          'labels': true_labels[i]})
    
    return targets
    


def train_one_epoch(model, optimizer, scheduler, dataloader):
    running_losses = []

    for image, bboxes, labels in iter(dataloader):
        images = image.to(DEVICE)
        targets = build_target(bboxes.to(DEVICE), labels)

        predict_loss = model(images, targets)

        loss = sum(loss for loss in predict_loss.values())
        running_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_losses


def train(model, optimizer, scheduler, dataloader, n_epoch)
    running_losses = []
    for i in range(n_epoch):
        epoch_losses = train_one_epoch(model, optimizer, scheduler, dataloader)
        running_losses.extend(epoch_losses)
    return running_losses

def evaluate(model, dataloader):
    return None


if __name__ == '__main__':

    dataloader = DetectionDataset('train', max_size=2048, transforms=T.Compose([T.ColorJitter(0.3, 0.5, 0.5, 0.1)]))

    model = create_model(False)

    optimizer = torch.optim.SGD(model1.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=GAMMA)

    train(model, optimizer, scheduler, dataloader, N_EPOCH)

    torch.save(model, 'checkpoints/fast-crnn/temp_.pth')


    