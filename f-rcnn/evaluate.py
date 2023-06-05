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
import torch.utils.data

from PIL import Image, ImageFile

from tqdm import tqdm

from dataset import DetectionDataset, class2sym
from

DEVICE =  'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'

def create_model(pretrained):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(class2sym))

    #model.load_state_dict(torch.load("./checkpoints/fast-crnn/2023-06-05-13-24-45_epoch_49.path"))
    model = torch.load("./checkpoints/fast-crnn/2023-06-05-13-24-45_epoch_49.path")

    return model

def prepare_prediction(prediction, batch_images):
    for i in len(predicion):
        boxes = predicion[i]['bboxes']
        labels = predicion[i]['labels']
        scores = predicion[i]['scores']

        images = batch_images[i]

        boxes[:,0] = (boxes[:,0] + boxes[:,2]) / 2
        boxes[:,1] = (boxes[:,1] + boxes[:,3]) / 2
        boxes[:,2] = boxes[:,2] - boxes[:,0]
        boxes[:,3] = boxes[:,3] - boxes[:,1]


        

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

    for i in range(bboxes.shape[0]):
        targets.append({'boxes': bboxes[i], 
                          'labels': labels[i].reshape(-1)})
    #print(targets)
    return images, targets



def evaluate(model, dataloader):
    
    
    running_losses = []
    print(len(dataloader))
    #for image, bboxes, labels in tqdm(iter(dataloader)):
    pcs = tqdm(enumerate(dataloader))
    for batch_idx, (image, bboxes, labels) in pcs:
        
        images, targets = build_target(image, torch.Tensor(bboxes).to(DEVICE), torch.LongTensor(labels))
        
        predict_loss = model(images)
        print(predict_loss)
        loss = sum(loss for loss in predict_loss.values())
        #print(loss)

        pcs.set_description(f"Loss {loss}")

        running_losses.append(loss.item())

        if (batch_idx % 40) == 0:
            print(loss)

    return running_losses


if __name__ == '__main__':

    print(DEVICE)

    transform = T.Compose([T.ColorJitter(0.3, 0.5, 0.5, 0.1),
                            T.ToTensor(),
                            T.Normalize(0, 255)])

    #dataloader = DetectionDataset('train', max_size=2048, crop_size=(512, 512), transforms=transform)
    dataset = DetectionDataset('train', max_size=2048, crop_size=(512, 512), 
                               transforms=transform, max_labels=15, mode='valid')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    model = create_model(None).to(DEVICE)
    model.eval()
    eval_losses = evaluate(model, dataloader)


    with open("eval_losses", "wb") as fp:   #Pickling
        pickle.dump(final_losses, fp)

    with open("eval_file_losses.txt", 'w') as f:
        for s in final_losses:
            f.write(str(s) + '\n')
    