import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
from logger import Logger
from torch.utils.data import DataLoader
from dataset import ClassificationDataset
import argparse
from tqdm.notebook import tqdm
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--learning-rate", type=float, default=0.03, help="learning rate")
parser.add_argument("--batch-size", type=int, default=32, help="size of each image batch")
parser.add_argument("--weights-path", type=str, default="", help="path to weights file")
parser.add_argument("--img-size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--log-every", type=int, default=1, help="")
parser.add_argument("--checkpoint-interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint-dir", type=str, default="classification/resnet_cp", help="directory where model checkpoints are saved"
)
parser.add_argument("--use-cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--cuda-id", type=int, default=0, help="which cuda to use")
opt = parser.parse_args()

device = f'cuda:{opt.cuda_id}' if torch.cuda.is_available() and opt.use_cuda else 'cpu'


if __name__ == '__main__':
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    train_transforms = transforms.Compose([
        transforms.ColorJitter(0.3, 0.5, 0.5, 0.1),
        transforms.RandomCrop(opt.img_size, padding=20, padding_mode='edge'),
        transforms.ToTensor(),
    ])
    train_ds = ClassificationDataset(opt.img_size, mode='train', transforms=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True)

    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    valid_ds = ClassificationDataset(opt.img_size, mode='valid', transforms=valid_transforms)
    valid_loader = DataLoader(valid_ds, batch_size=opt.batch_size, shuffle=False)

    model = models.resnext101_32x8d(num_classes=4781).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate)

    logger = Logger()

    for epoch in range(opt.epochs):
        print(f'Epoch {epoch}')
        train_running_loss = 0
        train_running_correct = 0
        running_counter = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            image, labels = data
            running_counter += image.shape[0]
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(image)

            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            train_running_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            if (i + 1) % opt.log_every == 0:
                logger.log({'train loss': train_running_loss / opt.log_every, 'train acc': train_running_correct / running_counter})
                train_running_correct = 0
                train_running_loss = 0
                running_counter = 0

            loss.backward()
            optimizer.step()
        if (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), opt.checkpoint_dir + f'/{epoch}.pth')
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                valid_correct = 0
                for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    image, labels = data
                    image = image.to(device)
                    labels = labels.to(device)

                    outputs = model(image)

                    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                    valid_loss += loss.item()

                    _, preds = torch.max(outputs.data, 1)
                    valid_correct += (preds == labels).sum().item()

            logger.log({'valid loss': valid_loss / len(valid_loader), 'valid acc': valid_correct / len(valid_ds)})
            model.train()
