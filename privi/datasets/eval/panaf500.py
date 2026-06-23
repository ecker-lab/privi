
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix

from privi.datasets.action.video_dataset import VideoDataset
from privi.datasets.generic import Evaluator

class PanAf500Evaluator(Evaluator):

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

        label_names = self.dataset.label_names

        pred_labels = np.argmax(scores, axis=1)  # (samples,)

        acc = accuracy_score(data_sorted, pred_labels)

        C = confusion_matrix(data_sorted, pred_labels)

        per_class_accs = np.diag(C) / C.sum(axis=1)

        class_avg_acc = np.mean(per_class_accs)

        results = {
            name: float(acc_i) for name, acc_i in zip(label_names, per_class_accs)
        }
        results["acc"] = float(acc)
        results["balanced_acc"] = float(class_avg_acc)

        return results

    def wandb_log_metrics(self, results, split, log_epoch, classifier_id):
        head_str = f"_h{classifier_id}" if classifier_id is not None else ""
        metrics = {
            f"{split}/acc{head_str}": results["acc"],
            f"{split}/balanced_acc{head_str}": results["balanced_acc"],
        }
        if log_epoch is not None:
            metrics[f"{split}/epoch{head_str}"] = log_epoch
        return metrics


class PanAf500(VideoDataset):

    multi_label = False
    num_classes = 9

    label_names = [
        'walking',
        'standing',
        'sitting',
        'climbing_up',
        'hanging',
        'climbing_down',
        'running',
        'camera_interaction',
        'sitting_on_back',
    ]

    label2number = {name: i for i, name in enumerate(label_names)}

    def __init__(self, label_path, pred_bboxes_path=None, **kwargs):
        super().__init__(label_path=label_path, dataset_type="csv", **kwargs)

    def evaluator(self):
        return PanAf500Evaluator(self)
