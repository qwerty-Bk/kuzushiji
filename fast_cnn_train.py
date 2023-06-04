import os

import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.utils.data

from PIL import Image, ImageFile

from tqdm import tqdm


def train_one_epoch(model, optimizer, scheduler, dataloader):
    
    return None
