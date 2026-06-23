# Copyright (c) OpenMMLab. All rights reserved.
# 
# Modified by Felix Benjamin Mueller, 2025

from collections import defaultdict
import multiprocessing
import time

import numpy as np
from privi.eval.mmaction import metrics
from privi.eval.mmaction.ava_utils import read_exclusions, print_time, tpfp_single


def ava_map(scores_avg, scores2idxs, annot_join, labels_ava, gt, exclude_file, agg_metrics={}, verbose=False, ignore_empty_frames=False, info={}, parallelize="none", sort_pred=False):
    """"

    Calculate mAP for AVA dataset.
    
    Args:
    scores_avg np.array(samples, classes): raw logits for each frame
    scores2idxs np.array(samples): indices of the bounding boxes corresponding to the logits
    annot_join pd.DataFrame: bounding boxes for each prediction
    gt pd.DataFrame: ground truth bounding boxes with labels in AVA format
    exclude_file str: path to the exclusion file
    agg_metrics dict[str, list[str]]: which labels to aggregate to which categories
    verbose bool: whether to print detailed information
    ignore_empty_frames bool: whether to ignore empty frames
    info dict: additional information to include in the result dict
    
    Returns:
    dict: mAP results for each class and overall mAP
    """

    pred_boxes = defaultdict(list)
    pred_scores = defaultdict(list)
    pred_labels = defaultdict(list)

    categories = labels_ava

    if annot_join is None:
        raise ValueError("Need annot_join")

    labels = list(sorted(c['id'] for c in categories))

    for idx, scores in zip(scores2idxs, scores_avg):

        try:
            row = annot_join.loc[idx]
        except KeyError:
            continue

        entry = f"test/images/{row['filename']},{row['frame']:04d}"

        for i, score in enumerate(scores):
            pred_boxes[entry].append(row[['x_min', 'y_min', 'x_max', 'y_max']].to_list())
            pred_scores[entry].append(score * row["confidence"])
            pred_labels[entry].append(labels[i])

    if sort_pred:
        entries = list(pred_scores.keys())
        for entry in entries:
            scores = pred_scores[entry]
            order = np.argsort(-np.array(scores))
            pred_boxes[entry] = [pred_boxes[entry][i] for i in order]
            pred_scores[entry] = [pred_scores[entry][i] for i in order]
            pred_labels[entry] = [pred_labels[entry][i] for i in order]

    gt_bboxes = defaultdict(list)
    gt_labels = defaultdict(list)

    for idx, row in gt.iterrows():
        entry = f"test/images/{row['filename']},{row['frame']:04d}"
        gt_bboxes[entry].append(row[['x_min', 'y_min', 'x_max', 'y_max']].to_list())
        gt_labels[entry].append(row['label'])

    boxes = pred_boxes
    labels = pred_labels
    scores = pred_scores

    if agg_metrics is None:
        agg_metrics = defaultdict(list)
        for cat in categories:
            if 'label_type' in cat:
                agg_metrics[cat['label_type']].append(cat['name'])

    class_whitelist = set(cat['id'] for cat in categories)

    with open(exclude_file) as f:
        excluded_keys = read_exclusions(f)

    start = time.time()
    all_gt_labels = np.concatenate(list(gt_labels.values()))
    gt_count = {k: np.sum(all_gt_labels == k) for k in class_whitelist}

    if verbose:
        print(gt_count)

    if ignore_empty_frames:
        tups = [(gt_bboxes[k], gt_labels[k], boxes[k], labels[k], scores[k])
                for k in gt_bboxes if k not in excluded_keys]
    else:
        tups = [(gt_bboxes.get(k, np.zeros((0, 4), dtype=np.float32)),
                    gt_labels.get(k, []), boxes[k], labels[k], scores[k])
                for k in boxes if k not in excluded_keys]
    if parallelize == "none":
        rets = [tpfp_single(tup) for tup in tups]
    elif parallelize == "multiprocessing":
        pool = multiprocessing.Pool(32)
        rets = pool.map(tpfp_single, tups)
    else:
        raise ValueError(f"Unknown parallelize method: {parallelize}")

    if verbose:
        print_time('Calculating TP/FP', start)

    start = time.time()
    scores, tpfps = defaultdict(list), defaultdict(list)
    for score, tpfp in rets:
        for k in score:
            scores[k].append(score[k])
            tpfps[k].append(tpfp[k])

    cls_AP = []
    for k in scores:
        scores[k] = np.concatenate(scores[k])
        tpfps[k] = np.concatenate(tpfps[k])
        precision, recall = metrics.compute_precision_recall(
            scores[k], tpfps[k], gt_count[k])
        ap = metrics.compute_average_precision(precision, recall)
        class_name = [x['name'] for x in categories if x['id'] == k]
        assert len(class_name) == 1
        class_name = class_name[0]
        cls_AP.append((k, class_name, ap))

    
    if verbose:
        print_time('Run Evaluator', start)

        print('Per-class results: ', flush=True)
        for k, class_name, ap in cls_AP:
            print(f'Index: {k}, Action: {class_name}: AP: {ap:.4f};', flush=True)

    overall = np.nanmean([x[2] for x in cls_AP])
    # Treat NaN as 0 for weighted mAP calculation
    weighted = np.nansum([ap * gt_count[k] for k, _, ap in cls_AP]) / sum(gt_count.values())

    if "notail" in agg_metrics:
        notail_weighted = np.nansum([ap * gt_count[k] for k, cls, ap in cls_AP if cls in agg_metrics["notail"]]) / sum(gt_count[k] for k, cls, _ in cls_AP if cls in agg_metrics["notail"])
        if verbose:
            print(f'Notail uAP: {notail_weighted:.4f}')

    if verbose:
        print('Overall Results: ', flush=True)
        print(f'Overall mAP: {overall:.4f}', flush=True)
        print(f'Overall uAP: {weighted:.4f}', flush=True)

    result_dict = {x[1]: x[2] for x in cls_AP}

    result = {**info}
    result["_overall"] = overall
    result["_weighted"] = weighted
    if "notail" in agg_metrics:
        result["notail_weighted"] = notail_weighted

    for agg_name, labels in agg_metrics.items():
        result[agg_name] = np.nanmean([result_dict[label] for label in labels])

    result.update(result_dict)

    return result