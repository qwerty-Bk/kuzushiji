import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from random import randint
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.notebook import tqdm


np.random.seed(3407)

uni2sym_df = pd.read_csv('./data/unicode_translation.csv')
forgotten = [{'Unicode': 'U+770C', 'char': '県'},
             {'Unicode': 'U+4FA1', 'char': '価'},
             {'Unicode': 'U+7A83', 'char': '窃'},
             {'Unicode': 'U+515A', 'char': '党'},
             {'Unicode': 'U+5E81', 'char': '庁'},
             {'Unicode': 'U+5039', 'char': '倹'}]
uni2sym_df = pd.concat([uni2sym_df, pd.DataFrame.from_records(forgotten)])
uni2class = dict(zip(uni2sym_df.Unicode, uni2sym_df.index.tolist()))
class2sym = dict(zip(uni2sym_df.index.tolist(), uni2sym_df.char))
class2uni = dict(zip(uni2sym_df.index.tolist(), uni2sym_df.Unicode))


def load_data(split, path):
    assert split in ('train', 'test')
    csv_split = 'train' if split == 'train' else 'sample_submission'
    df = pd.read_csv(f'{path}/{csv_split}.csv', keep_default_na=False)
    image_path = Path(f'{path}/{split}')
    images = []
    for id, row in tqdm(df.iterrows(), total=len(df)):
        id_name = row['image_id'] + '.jpg'
        image = {
            'file': Path.joinpath(image_path, id_name),
            'bboxes': [],
            'labels': []
        }
        if split == 'train':
            labels = row['labels'].split()
            assert len(labels) % 5 == 0
            n = len(labels) // 5
            for i in range(0, n):
                uni, x, y, w, h = labels[i * 5:(i + 1) * 5]
                image['bboxes'].append(np.array([int(x), int(y), int(w), int(h)]))
                image['labels'].append(uni2class[uni])
        else:
            image['useage'] = row['Useage']
        image['bboxes'] = np.array(image['bboxes'])
        image['labels'] = np.array(image['labels'])
        images.append(image)
    return images


def ourCrop(image, bboxes, labels, w, h, n_w, n_h, thresh=0.33, max_labels=50, to_fill=True):
    if w == n_w or h == n_h:
        return image, bboxes, labels

    x0, y0 = randint(0, w - n_w), randint(0, h - n_h)
    x1, y1 = x0 + n_w, y0 + n_h

    new_bboxes = bboxes
    new_labels = labels
    if to_fill:
        new_bboxes = torch.full((max_labels, 4), 0)
        new_labels = torch.full((max_labels, 1), 0)
    fill_i = 0
    for i, bbox in enumerate(bboxes):
        if fill_i == max_labels:
            break
        intersec = max(0, min(x1, bbox[0] + bbox[2]) - max(x0, bbox[0])) * \
                   max(0, min(y1, bbox[1] + bbox[3]) - max(y0, bbox[1]))
        if intersec / bbox[2] / bbox[3] > thresh:
            new_bboxes[fill_i] = torch.Tensor([max(x0, bbox[0]) - x0, max(y0, bbox[1]) - y0,
                                               min(x1, bbox[0] + bbox[2]) - max(x0, bbox[0]),
                                               min(y1, bbox[1] + bbox[3]) - max(y0, bbox[1])])
            new_labels[fill_i] = labels[i]
            fill_i += 1
    if isinstance(image, torch.Tensor) or isinstance(image, np.ndarray):
        new_image = image[x0:x1, y0:y1]
    elif isinstance(image, Image.Image):
        new_image = image.crop((x0, y0, x1, y1))
    else:
        raise ValueError('unknown image type in crop')
    return new_image, np.array(new_bboxes), np.array(new_labels)


class DetectionDataset(Dataset):
    def __init__(self, images, max_size=None, crop_size=(1024, 1024), transforms=None, threshold=0.5, mode='train',
                 max_labels=300, to_fill=True):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if isinstance(images, str):
            self.images = load_data(images, 'data')
            train, valid = train_test_split(self.images, test_size=0.3, 
                                            train_size=0.7, random_state=3407,
                                            shuffle=False)
            if mode == 'train':
                self.images = train
            elif mode == 'valid':
                self.images = valid
        else:
            self.images = images
        self.max_size = max_size
        self.crop_size = crop_size
        self.transforms = transforms
        self.threshold = threshold
        self.max_labels = max_labels
        self.to_fill = to_fill

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]['file'])
        bboxes = self.images[idx]['bboxes']  # pixels: (x, y, w, h)
        labels = self.images[idx]['labels']
        if self.max_size is not None:
            ratio = self.max_size / max(image.width, image.height)
            bboxes = (bboxes.astype(float) * ratio).astype(int)
            image = image.resize((int(image.width * ratio), int(image.height * ratio)))

        # note that the boxes after this line might go over the borders (if we have to crop them, we have to change the new_boxxes.append line in ourCrop)
        image, bboxes, labels = ourCrop(image, bboxes, labels, image.width, image.height, *self.crop_size,
                                        self.threshold, max_labels=self.max_labels, to_fill=self.to_fill)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, bboxes, torch.LongTensor(labels)