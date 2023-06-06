import os
from datetime import datetime
import csv

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
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataset import DetectionDataset, class2sym, class2uni
from utils import draw_bboxes

#DEVICE =  'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

def create_model(pretrained):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    #model.load_state_dict(torch.load("./checkpoints/fast-crnn/2023-06-05-13-24-45_epoch_49.path"))
    model = torch.load("./3_checkpoints/0.001fast-crnn/2023-06-06-18-20-55_epoch_6.path")

    return model

def prepare_prediction(batch_image, prediction):
    #print(prediction)
    for i in range(len(prediction)):
        print(prediction[i])
        boxes = prediction[i]['boxes']
        labels = prediction[i]['labels']
        scores = prediction[i]['scores']

        boxes[:,2] = (boxes[:,2] - boxes[:,0]) 
        boxes[:,3] = (boxes[:,3] - boxes[:,1]) 


        #clip_value = (scores >= 0.4).sum().item()

        #boxes = boxes[:clip_value]
        #labels = labels[:clip_value]

        image_marked = draw_bboxes(batch_image, boxes, labels)
          
        plt.imshow(image_marked)
        plt.show()

def fill_submission(image_id, prediction):
    #print(prediction)
    str_labels = ''
    for i in range(len(prediction)):
        boxes = prediction[i]['boxes']
        labels = prediction[i]['labels']
        scores = prediction[i]['scores']

        #boxes[:,2] = (boxes[:,2] - boxes[:,0]) 
        #boxes[:,3] = (boxes[:,3] - boxes[:,1]) 


        clip_value = (scores >= 0.4).sum().item()

        boxes = boxes[:clip_value]
        labels = labels[:clip_value]

        for j in range(labels.shape[0]):
            str_labels += str(class2uni[labels[j].item()]) + ' ' + str( round((boxes[j,2].item() + boxes[j,0].item()) / 2)) + " "
            str_labels +=  str( round((boxes[j,3].item() + boxes[j,1].item()) / 2)) + " "
    
    str_labels = str_labels[:-1]    
    str_labels += '\n'

    csv_row = str(image_id) + ',' + str_labels
    with open('submission.csv','a') as fd:
        fd.write(csv_row)

    return None

def build_target(image, bboxes, labels):
    images = image.to(DEVICE)
    labels = labels.to(DEVICE)
    
    bboxes = bboxes.type(torch.FloatTensor).to(DEVICE)

    true_labels = torch.ones(labels.size()).to(DEVICE)
    
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
    return images, targets

def fix_pred_loss(predict_loss, x_ind, y_ind, ):
    #print(predict_loss)
    for i in range(len(predict_loss)):
        predict_loss[i]['boxes'][:,0] += x_ind
        predict_loss[i]['boxes'][:,2] += x_ind
        predict_loss[i]['boxes'][:,1] += y_ind
        predict_loss[i]['boxes'][:,3] += y_ind
    #print(predict_loss)
    return predict_loss

def evaluate(model, dataloader):
    print(len(dataloader))
    pcs = tqdm(enumerate(dataloader))
    for batch_idx, (image, image_id, bboxes, labels) in pcs:
        image_t = image[0].clone() * 255
        image_t = np.moveaxis(image_t.numpy()*255, 0, -1)
        image_t = Image.fromarray(image_t.astype(np.uint8))

        _, _, image_height, image_width = image.shape

        size_step = 512
        stride = 480
        answer = []

        for i in range(0, image_width, stride):
            for j in range(0, image_height, stride):
                image_cropped = image[:,
                                      :, 
                                      j : min(image_height, j + size_step), 
                                      i : min(image_width, i + size_step)]  #c, h, w

                images, targets = build_target(image_cropped, torch.Tensor(bboxes).to(DEVICE), torch.LongTensor(labels))
                #print(images.size())
                #print(images, targets)
                predict_loss = fix_pred_loss(model(images), i, j)
                answer.extend(predict_loss)

        #image_t = image[0].clone() * 255
        #image_t = np.moveaxis(image_t.numpy()*255, 0, -1)
        # #print(image_t)
        #image_t = Image.fromarray(image_t.astype(np.uint8))
        # #print(answer)

        prepare_prediction(image_t, answer)
        #fill_submission(image_id[0], answer)

    return

if __name__ == '__main__':

    print(DEVICE)

    csv_row = 'image_id' + ',' + 'labels\n'
    with open('submission.csv','a') as fd:
        fd.write(csv_row)

    transform = T.Compose([T.ColorJitter(0.3, 0.5, 0.5, 0.1),
                          T.ToTensor(),
                          T.Normalize(0, 255)])

    #dataloader = DetectionDataset('train', max_size=2048, crop_size=(512, 512), transforms=transform)
    dataset = DetectionDataset('train', max_size=None, crop_size=(2048, 2048), 
                               transforms=transform, max_labels=15, mode='valid', 
                               apply_crop=False, return_ids=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = create_model(None).to(DEVICE)
    model.eval()
    with torch.no_grad():
        evaluate(model, dataloader)
    