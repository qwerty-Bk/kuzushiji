from yolo.utils import non_max_suppression
from yolo.models import Darknet
import argparse
import torch
from os.path import normpath, basename
from dataset import DetectionDataset, class2uni
import torchvision.transforms as T
from tqdm.notebook import tqdm
import pandas as pd
from det_class_combine import get_all_outputs


parser = argparse.ArgumentParser()
parser.add_argument("--use-cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--cuda-id", type=int, default=0, help="which cuda to use")
parser.add_argument("--max-label", type=int, default=300, help="interval between saving model weights")
parser.add_argument("--stride", type=int, default=208, help="")
parser.add_argument("--weights-path", type=str, default="", help="path to weights file")
parser.add_argument("--pred-path", type=str, default="answer.csv", help="path to weights file")
parser.add_argument("--conf-thres", type=float, default=0.9, help="object confidence threshold")
parser.add_argument("--nms-thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
opt = parser.parse_args()

device = f'cuda:{opt.cuda_id}' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    model = Darknet('yolo/yolo-tiny.cfg')
    model.load_weights(opt.weights_path, device)
    model.eval()
    model = model.to(device)

    stride = opt.stride
    yolo_img_size = 416

    train_ds = DetectionDataset('train', max_size=1024, transforms=T.Compose([T.ToTensor()]),
                                apply_crop=False, threshold=0.1, mode='valid', max_labels=300)

    pred_df = pd.DataFrame()
    for i in tqdm(range(len(train_ds))):
        image_tensor = train_ds[i][0]
        file_name = basename(normpath(train_ds.images[i]['file']))[:-4]
        with torch.no_grad():
            all_outputs = get_all_outputs(model, image_tensor)

        outputs = non_max_suppression(all_outputs, 4781, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
        labels = ''
        output_sorted_ind = outputs[0][:, 4].sort(descending=True)[1]
        for idx, pred in enumerate(outputs[0][output_sorted_ind]):
            if idx >= 1200:
                break
            xo, yo, xe, ye = pred[:4]
            cl = int(pred[-1].item())
            xc, yc = (xo + xe) / 2, (yo + ye) / 2
            xc, yc = xc.item(), yc.item()
            labels += class2uni[cl] + ' ' + str(int(xc)) + ' ' + str(int(yc)) + ' '

        needed_row = pd.DataFrame({'image_id': [file_name], 'labels': [labels[:-1]], 'Useage': ['Public']})
        pred_df = pd.concat([pred_df, needed_row])
    pred_df.to_csv(opt.pred_path)
