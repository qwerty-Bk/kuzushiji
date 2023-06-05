from __future__ import division

import argparse
import os
import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader

from dataset import DetectionDataset
from logger import Logger
from yolo.models import Darknet
from yolo.utils import parse_model_config, weights_init_normal


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--batch-size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model-config-path", type=str, default="yolo/yolo-tiny.cfg", help="path to model config file")
parser.add_argument("--weights-path", type=str, default="", help="path to weights file")
parser.add_argument("--conf-thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms-thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
parser.add_argument("--img-size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint-interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--max-label", type=int, default=300, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint-dir", type=str, default="yolo/checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use-cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--cuda-id", type=int, default=0, help="which cuda to use")
opt = parser.parse_args()

device = f'cuda:{opt.cuda_id}' if torch.cuda.is_available() and opt.use_cuda else 'cpu'

if __name__ == '__main__':
    os.makedirs("yolo/output", exist_ok=True)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    hyperparams = parse_model_config(opt.model_config_path)[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])

    model = Darknet(opt.model_config_path)
    model.apply(weights_init_normal)
    if len(opt.weights_path) > 1:
        model.load_weights(opt.weights_path)

    model = model.to(device)
    model.train()

    logger = Logger()

    transforms = T.Compose([
        T.ColorJitter(0.3, 0.5, 0.5, 0.1),
        T.ToTensor()
    ])
    train_set = DetectionDataset('train', 1024, opt.img_size, transforms)
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(opt.epochs):
        for batch_i, (image, bboxes, labels) in enumerate(dataloader):
            imgs = image.to(device)
            yolo_bboxes = bboxes.to(device)
            yolo_bboxes[..., 2:] = yolo_bboxes[..., 2:] / 2
            yolo_bboxes[..., 0] += yolo_bboxes[..., 2]
            yolo_bboxes[..., 1] += yolo_bboxes[..., 3]
            yolo_bboxes = yolo_bboxes / opt.img_size

            labels = labels.to(device)
            targets = torch.concat([labels, yolo_bboxes], -1)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()

            logger.log({'x': model.losses["x"], 'y': model.losses["y"], 'w': model.losses["w"], 'h': model.losses["h"],
                        'conf': model.losses["conf"], 'cls': model.losses["cls"], 'loss': loss.item(),
                        'pr': model.losses["precision"], 'rec': model.losses["recall"]})
            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    opt.epochs,
                    batch_i,
                    len(dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )

            model.seen += imgs.size(0)

        if epoch % opt.checkpoint_interval == 0:
            model.save_weights("%s/%d" % (opt.checkpoint_dir, epoch))
