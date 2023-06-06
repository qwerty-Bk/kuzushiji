from yolo.models import Darknet
from torchvision import models
from tqdm import tqdm
import argparse
import torch
from torchvision import transforms as T
from dataset import DetectionDataset
from yolo.utils import non_max_suppression
from torch.nn import functional as F
from dataset import class2uni
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from os.path import normpath, basename

num_classes = 4781

detectors = {
    "yolo": Darknet("yolo/yolo.cfg"),
    "yolo_tiny": Darknet("yolo/yolo-tiny.cfg"),
}

classifiers = {
    "resnext101": models.resnext101_32x8d(num_classes=num_classes)
}

class_img_size = {
    "resnext101": 224,
}


def load_weights(model_name, path):
    model = None
    if model_name in ("yolo", "yolo_tiny"):
        model = detectors[model_name]
        model.load_weights(path, device)
    elif model_name == "resnext101":
        model = classifiers[model_name]
        model.load_state_dict(torch.load(path, map_location=device))
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--class-model", type=str, help="classifier: " + ' '.join(classifiers.keys()))
parser.add_argument("--det-model", type=str, help="detector: " + ' '.join(detectors.keys()))
parser.add_argument("--class-weights", type=str, help="path to class weights")
parser.add_argument("--det-weights", type=str, help="path to detector weights")
parser.add_argument("--conf-thres", type=float, default=0.9, help="object confidence threshold")
parser.add_argument("--nms-thres", type=float, default=0.2, help="iou threshold for non-maximum suppression")
parser.add_argument("--max-size", type=int, default=1024, help="hyperpar of dataset")
parser.add_argument("--draw-every", type=int, default=0, help="which image to show, 0 shows nothing")
parser.add_argument("--csv-path", type=str, default="answer.csv", help="file for submission")
parser.add_argument("--use-cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--cuda-id", type=int, default=0, help="which cuda to use")
opt = parser.parse_args()

device = f'cuda:{opt.cuda_id}' if torch.cuda.is_available() and opt.use_cuda else 'cpu'


def get_all_outputs(model, image_tensor, stride=208, img_size=416):
    all_outputs = []
    for x0 in range(0, image_tensor.shape[2], stride):
        x = x0
        if x0 + img_size > image_tensor.shape[2]:
            x = image_tensor.shape[2] - img_size
        for y0 in range(0, image_tensor.shape[1], stride):
            y = y0
            if y0 + img_size > image_tensor.shape[1]:
                y = image_tensor.shape[1] - img_size
            current_image = image_tensor[:, y:y + img_size, x:x + img_size].to(device).unsqueeze(0)
            output = model(current_image)
            output[0][:, 0] += x
            output[0][:, 1] += y
            all_outputs.append(output)

            if y == image_tensor.shape[1] - img_size:
                break
        if x == image_tensor.shape[2] - img_size:
            break
    all_outputs = torch.cat(all_outputs, 1)
    return all_outputs


if __name__ == '__main__':
    assert opt.class_model in classifiers.keys()
    assert opt.det_model in detectors.keys()
    classifier = load_weights(opt.class_model, opt.class_weights)
    detector = load_weights(opt.det_model, opt.det_weights)
    classifier.to(device)
    detector.to(device)
    classifier.eval()
    detector.eval()

    transforms = T.Compose([
        T.ToTensor()
    ])
    valid_set = DetectionDataset('train', None if opt.max_size == 0 else opt.max_size, transforms=transforms,
                                 apply_crop=False, mode='valid', return_ids=True)

    pred_df = pd.DataFrame()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(valid_set)):
            image, _, bboxes, _ = sample
            file_name = basename(normpath(valid_set.images[i]['file']))[:-4]
            labels = ""
            if opt.draw_every != 0 and (i + 1) % opt.draw_every == 0:
                picture = Image.fromarray(np.moveaxis(image.to("cpu").numpy() * 255, 0, -1).astype(np.uint8))
                draw = ImageDraw.Draw(picture)
            if opt.det_model in ('yolo', 'yolo_tiny'):
                all_outputs = get_all_outputs(detector, image)
                all_outputs[0, :, 5:5 + num_classes] = 1 / num_classes  # all boxes are now of the same class
                outputs = non_max_suppression(all_outputs, num_classes,
                                              conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
                predictions = outputs[0][:, :4] if outputs[0] is not None else None
            else:
                raise NotImplementedError(f"Can't get predictions for {opt.det_model}")
            if predictions is None:
                print(f"No bboxes found for sample {i}")
                labels += " "
            else:
                for prediction in predictions:
                    x0, y0, x1, y1 = prediction
                    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
                    orig_img = Image.open(f'data/train/{file_name}.jpg')
                    k = 1 if opt.max_size is None else max(orig_img.width, orig_img.height) / opt.max_size
                    xc, yc = xc.item() * k, yc.item() * k
                    x0, y0, x1, y1 = [int(x.item()) for x in [x0, y0, x1, y1]]
                    if opt.draw_every != 0 and (i + 1) % opt.draw_every == 0:
                        draw.rectangle((x0, y0, x1, y1), outline=(240, 0, 200), width=1)
                    crop = image[:, y0:y1, x0:x1]
                    padded_size = max(crop.shape[1:])
                    padded_add = padded_size - min(crop.shape[1:])
                    if crop.shape[1] > crop.shape[2]:
                        padding = (padded_add // 2, padded_add - padded_add // 2)
                    else:
                        padding = (0, 0, padded_add // 2, padded_add - padded_add // 2)
                    crop = F.pad(crop, padding, value=128 / 255)
                    crop = T.Resize(class_img_size[opt.class_model])(crop)
                    outputs = classifier(crop.unsqueeze(0).to(device))
                    _, pred = torch.max(outputs.data, 1)
                    pred = int(pred.item())
                    labels += class2uni[pred] + ' ' + str(int(xc)) + ' ' + str(int(yc)) + ' '

                if opt.draw_every != 0 and (i + 1) % opt.draw_every == 0:
                    picture.show()

            needed_row = pd.DataFrame({'image_id': [file_name], 'labels': [labels[:-1]], 'Useage': ['Public']})
            pred_df = pd.concat([pred_df, needed_row])

    pred_df.to_csv(opt.csv_path, index=False)
