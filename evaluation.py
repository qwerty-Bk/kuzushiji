# official python version
import argparse
import multiprocessing
import numpy as np
import os
import pandas as pd


def define_console_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_path', default='answer.csv', type=str)   # submission
    parser.add_argument('--solution_path', default='', type=str)        # ground truth
    return parser


def score_page(preds, truth):
    """
    Scores a single page.
    Args:
        preds: prediction string of labels and center points.
        truth: ground truth string of labels and bounding boxes.
    Returns:
        True/false positive and false negative counts for the page
    """
    tp = 0
    fp = 0
    fn = 0

    truth_indices = {
        'label': 0,
        'X': 1,
        'Y': 2,
        'Width': 3,
        'Height': 4
    }
    preds_indices = {
        'label': 0,
        'X': 1,
        'Y': 2
    }

    if pd.isna(truth) and pd.isna(preds):
        return {'tp': tp, 'fp': fp, 'fn': fn}

    if pd.isna(truth):
        fp += len(preds.split(' ')) // len(preds_indices)
        return {'tp': tp, 'fp': fp, 'fn': fn}

    if pd.isna(preds):
        fn += len(truth.split(' ')) // len(truth_indices)
        return {'tp': tp, 'fp': fp, 'fn': fn}

    truth = truth.split(' ')
    if len(truth) % len(truth_indices) != 0:
        raise ValueError('Malformed solution string')
    truth_label = np.array(truth[truth_indices['label']::len(truth_indices)])
    truth_xmin = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
    truth_ymin = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
    truth_xmax = truth_xmin + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
    truth_ymax = truth_ymin + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

    preds = preds.split(' ')
    if len(preds) % len(preds_indices) != 0:
        raise ValueError('Malformed prediction string')
    preds_label = np.array(preds[preds_indices['label']::len(preds_indices)])
    preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
    preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)
    preds_unused = np.ones(len(preds_label)).astype(bool)

    for xmin, xmax, ymin, ymax, label in zip(truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label):
        # Matching = point inside box & character same & prediction not already used
        matching = (xmin < preds_x) & (xmax > preds_x) & (ymin < preds_y) & (ymax > preds_y) & (preds_label == label) & preds_unused
        if matching.sum() == 0:
            fn += 1
        else:
            tp += 1
            preds_unused[np.argmax(matching)] = False
    fp += preds_unused.sum()
    return {'tp': tp, 'fp': fp, 'fn': fn}


def kuzushiji_f1(sub, solution):
    """
    Calculates the competition metric.
    Args:
        sub: submissions, as a Pandas dataframe
        solution: solution, as a Pandas dataframe
    Returns:
        f1 score
    """
    if not all(sub['image_id'].values == solution['image_id'].values):
        raise ValueError("Submission image id codes don't match solution")

    pool = multiprocessing.Pool()
    results = pool.starmap(score_page, zip(sub['labels'].values, solution['labels'].values))
    pool.close()
    pool.join()

    tp = sum([x['tp'] for x in results])
    fp = sum([x['fp'] for x in results])
    fn = sum([x['fn'] for x in results])

    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision > 0 and recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


if __name__ == '__main__':
    # both dfs have 2 columns: image_id and labels
    parser = define_console_parser()
    shell_args = parser.parse_args()
    sub = pd.read_csv(shell_args.sub_path)
    path, split = 'data', 'train'
    if len(shell_args.solution_path) == 0:
        df = pd.read_csv(f'{path}/{split}.csv', keep_default_na=False)
        solution = pd.DataFrame()
        for id, row in sub.iterrows():
            img_id = row['image_id']
            needed_row = df.loc[df['image_id'] == img_id]
            rest = needed_row['labels'].tolist()[0]
            rest = rest.split()
            labels = ""
            for n in range(len(rest) // 5):
                labels += rest[n * 5] + " " + rest[n * 5 + 1] + " " + rest[n * 5 + 2] + " "
                labels += str(int(rest[n * 5 + 1]) + int(rest[n * 5 + 3])) + " "
                labels += str(int(rest[n * 5 + 2]) + int(rest[n * 5 + 4])) + " "
            solution = pd.concat([solution, pd.DataFrame({'image_id': [img_id], 'labels': [labels[:-1]]})])
    else:
        solution = pd.read_csv(shell_args.solution_path)
    score = kuzushiji_f1(sub, solution)
    print('sub', sub)
    print('solution', solution)
    print('F1 score of: {0}'.format(score))
