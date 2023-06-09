import mmengine
import mmcv
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def iter_bboxes(labels):
    if not labels:
        return
    labels = labels.split()
    n = len(labels)
    assert n % 5 == 0
    for i in range(0, n, 5):
        ch, x, y, w, h = labels[i:i + 5]
        yield int(x), int(y), int(w), int(h), ch



def prepare_train():
    df = pd.read_csv('./data/train.csv', keep_default_na=False)
    img_dir = Path('./data/train')

    unicode_translation = pd.read_csv('./data/unicode_translation.csv')
    forgotten = [{'Unicode': 'U+770C', 'char': '県'},
                 {'Unicode': 'U+4FA1', 'char': '価'},
                 {'Unicode': 'U+7A83', 'char': '窃'},
                 {'Unicode': 'U+515A', 'char': '党'},
                 {'Unicode': 'U+5E81', 'char': '庁'},
                 {'Unicode': 'U+5039', 'char': '倹'}]
    unicode_translation = pd.concat([unicode_translation, pd.DataFrame.from_records(forgotten)])
    unicode2class = dict(
        zip(unicode_translation['Unicode'], unicode_translation.index.values))

    # Add images to COCO
    images = []
    for img_id, row in tqdm(df.iterrows()):
        filename = row['image_id'] + '.jpg'
        img = Image.open(img_dir / filename)
        image = {
            'filename': filename,
            'width': img.width,
            'height': img.height,
        }
        bboxes = []
        labels = []
        for x, y, w, h, ch in iter_bboxes(row['labels']):
            bboxes.append([x, y, x + w, y + h])
            # labels.append(1)
            labels.append(unicode2class[ch] + 1)
        image['ann'] = {
            'bboxes': np.array(bboxes).astype(np.float32).reshape(-1, 4),
            'labels': np.array(labels).astype(np.int64).reshape(-1),
            'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
            'labels_ignore': np.array([], dtype=np.int64).reshape(-1, )
        }
        images.append(image)

    import random
    random.shuffle(images)
    mmengine.dump([im for im in images if im['filename'].startswith('umgy')], './data/dval.pkl')
    mmengine.dump([im for im in images if not im['filename'].startswith('umgy')], './data/dtrain.pkl')
    mmengine.dump(images, './data/dtrainval.pkl')


def prepare_test():
    df = pd.read_csv('./data/sample_submission.csv', keep_default_na=False)
    img_dir = Path('./data/test')

    images = []
    for img_id, row in tqdm(df.iterrows()):
        filename = row['image_id'] + '.jpg'
        img = Image.open(img_dir / filename)
        images.append({
            'filename': filename,
            'width': img.width,
            'height': img.height,
            'ann': {
                'bboxes': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels': np.array([], dtype=np.int64).reshape(-1, ),
                'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array([], dtype=np.int64).reshape(-1, )
            }
        })
    mmengine.dump(images, './data/dtest.pkl')


if __name__ == "__main__":
    prepare_train()
    prepare_test()