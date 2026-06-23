
import numpy as np
import torch
import torchmetrics
from sklearn.metrics import top_k_accuracy_score

from privi.datasets.action.video_dataset import VideoDataset
from privi.datasets.generic import Evaluator

def top_k_macro_accuracy(y_true, y_score, classes, k=5):
    per_class_acc = []
    for c in classes:
        mask = (y_true == c)
        if mask.sum() == 0:
            continue
        acc_c = top_k_accuracy_score(y_true[mask], y_score[mask], k=k, labels=classes)
        per_class_acc.append(acc_c)
    return np.mean(per_class_acc), per_class_acc


class ChimpBehaveEvaluator(Evaluator):

    def __init__(self, dataset):
        self.dataset = dataset

        self.scores = None
        self.indices = None
        self.idx = 0

    def add_batch(self, labels, logits):

        if self.scores is None:
            self.scores = np.zeros((self.dataset.num_crops(), *logits.shape[1:]), dtype=np.float32)
            self.indices = np.zeros((self.dataset.num_crops(),), dtype=np.int32)

        self.scores[self.idx : self.idx + len(logits)] = logits.detach().cpu().numpy()

        indices = labels["index"][labels["present_crops"].bool()].detach().cpu().numpy()
        self.indices[self.idx : self.idx + len(logits)] = indices
        self.idx += len(logits)

    def metrics(self):

        scores = self.scores
        if len(scores.shape) > 2:
            scores = np.mean(scores, axis=1)
        scores2idxs = self.indices

        # GT labels from the dataset, sorted by the indices (1-D integer array)
        data_sorted = np.array(self.dataset.labels, dtype=int)[scores2idxs]

        n_classes = self.dataset.num_classes
        label_names = self.dataset.label_names
        int_classes = list(range(n_classes))

        results = {}
        map_macro = torchmetrics.functional.average_precision(torch.tensor(scores), torch.tensor(data_sorted), task="multiclass", num_classes=n_classes, num_labels=None, average='macro')
        map_weighted = torchmetrics.functional.average_precision(torch.tensor(scores), torch.tensor(data_sorted), task="multiclass", num_classes=n_classes, num_labels=None, average='weighted')
        results["map_macro"] = map_macro.item()
        results["map_weighted"] = map_weighted.item()

        for k in [1, 3, 5]:
            results[f"acc_top{k}_micro"] = top_k_accuracy_score(data_sorted, scores, k=k, labels=int_classes)
            results[f"acc_top{k}_macro"], per_class = top_k_macro_accuracy(data_sorted, scores, int_classes, k=k)
            for label, result in zip(label_names, per_class):
                results[f"per_class_acc_top{k}_{label}"] = result

        return results

    def wandb_log_metrics(self, results, split, log_epoch, classifier_id):
        head_str = f"_h{classifier_id}" if classifier_id is not None else ""
        metrics = {f"{k}{head_str}": v for k, v in results.items() if k.startswith("acc")}
        if log_epoch is not None:
            metrics[f"{split}/epoch{head_str}"] = log_epoch
        return metrics


class ChimpBehave(VideoDataset):

    multi_label = False
    num_classes = 7

    label2number = {"Sitting": 0,
                    "Standing": 1,
                    "Walking": 2,
                    "Hanging": 3,
                    "Climbing Up": 4,
                    "Climbing Down": 5,
                    "Running": 6}

    label_names = list(label2number.keys())

    def __init__(self, label_path, **kwargs):
        super().__init__(label_path=label_path, dataset_type="csv", **kwargs)

    def evaluator(self):
        return ChimpBehaveEvaluator(self)
