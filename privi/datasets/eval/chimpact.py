import numpy as np
from pathlib import Path
from collections import defaultdict

from privi.datasets.action.video_dataset import VideoDataset, load_ava_dataset
from privi.datasets.generic import Evaluator
from privi.eval.ava_map import ava_map

class ChimpACTEvaluator(Evaluator):

    def __init__(self, dataset: VideoDataset):
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


        tail_classes = ["displaying", "being begged from", "begging", "taking object", "losing object", "aggressing", "erection"]
        mid_classes = ["being carried", "carrying", "manipulating object", "being nursed", "nursing", "climbing"]
        head_classes = ["being groomed", "eating", "grooming", "moving", "playing", "embracing", "sleeping", "touching"]
        supercommon_classes = ["solitary object playing", "resting"]
        notail_classes = mid_classes + head_classes + supercommon_classes
        sf_overall = mid_classes + head_classes + supercommon_classes + ["displaying", "aggressing"]
        sf_social = ["grooming", "being groomed", "aggressing", "embracing", "carrying", "being carried", "nursing", "being nursed", "playing", "touching", "displaying"]
        locomotion = ["moving", "climbing", "resting", "sleeping"]
        object_interaction = ["solitary object playing", "eating", "manipulating object"]

        agg_metrics = {
            "tail": tail_classes,
            "mid": mid_classes,
            "head": head_classes,
            "supercommon": supercommon_classes,
            "notail": notail_classes,
            "overall_exclude_extreme_tail": sf_overall,
            "social_exclude_extreme_tail": sf_social,
            "locomotion": locomotion,
            "object_interaction": object_interaction,
        }

        scores = self.scores
        if len(scores.shape) > 2:
            if self.dataset.view_agg_fn == "mean":
                scores = np.mean(scores, axis=1) # (samples, classes)
            elif self.dataset.view_agg_fn == "add":
                scores = np.sum(scores, axis=1) # (samples, classes)
            else:
                raise ValueError(f"Unknown view agg func {self.dataset.view_agg_fn}")

        results = ava_map(
            scores_avg=scores,
            scores2idxs=self.indices,
            annot_join=self.dataset.evaluator_annot_joint,
            gt=self.dataset.evaluator_gt,
            labels_ava=self.dataset.labels_ava,
            exclude_file=self.dataset.excluded_timestamps_path,
            agg_metrics=agg_metrics,
            sort_pred=self.dataset.sort_pred,
        )

        return results

    def wandb_log_metrics(self, results, split, log_epoch, classifier_id):
        head_str = f"_h{classifier_id}" if classifier_id is not None else ""
        metrics = {
            f"{split}_mAP/macro{head_str}": results["_overall"],
            f"{split}_mAP/weighted{head_str}": results["_weighted"],
            f"{split}_mAP/head{head_str}": results["head"],
            f"{split}_mAP/mid{head_str}": results["mid"],
            f"{split}_mAP/tail{head_str}": results["tail"],
            f"{split}_mAP/supercommon{head_str}": results["supercommon"],
            f"{split}_mAP/notail{head_str}": results["notail"],
        }
        if log_epoch is not None:
            metrics[f"{split}_mAP/epoch{head_str}"] = log_epoch
        return metrics



class ChimpACT(VideoDataset):

    multi_label = True
    num_classes = 23

    # We are doing classes starting from 0 here, unlike ChimpACT annot files which start with 1
    label_names = [
        'moving',
        'climbing',
        'resting',
        'sleeping',
        'solitary object playing',
        'eating',
        'manipulating object',
        'grooming',
        'being groomed',
        'aggressing',
        'embracing',
        'begging',
        'being begged from',
        'taking object',
        'losing object',
        'carrying',
        'being carried',
        'nursing',
        'being nursed',
        'playing',
        'touching',
        'erection',
        'displaying'
    ]

    idx2label_name = {idx: label_name for idx, label_name in enumerate(label_names)}

    # Labels with categories, used for ava_map
    labels_ava = [
        { "name": "moving", "id": 1, "label_type": "locomotion" },
        { "name": "climbing", "id": 2, "label_type": "locomotion" },
        { "name": "resting", "id": 3, "label_type": "locomotion" },
        { "name": "sleeping", "id": 4, "label_type": "locomotion" },
        { "name": "solitary object playing", "id": 5, "label_type": "object" },
        { "name": "eating", "id": 6, "label_type": "object" },
        { "name": "manipulating object", "id": 7, "label_type": "object" },
        { "name": "grooming", "id": 8, "label_type": "social" },
        { "name": "being groomed", "id": 9, "label_type": "social" },
        { "name": "aggressing", "id": 10, "label_type": "social" },
        { "name": "embracing", "id": 11, "label_type": "social" },
        { "name": "begging", "id": 12, "label_type": "social" },
        { "name": "being begged from", "id": 13, "label_type": "social" },
        { "name": "taking object", "id": 14, "label_type": "social" },
        { "name": "losing object", "id": 15, "label_type": "social" },
        { "name": "carrying", "id": 16, "label_type": "social" },
        { "name": "being carried", "id": 17, "label_type": "social" },
        { "name": "nursing", "id": 18, "label_type": "social" },
        { "name": "being nursed", "id": 19, "label_type": "social" },
        { "name": "playing", "id": 20, "label_type": "social" },
        { "name": "touching", "id": 21, "label_type": "social" },
        { "name": "erection", "id": 22, "label_type": "other" },
        { "name": "displaying", "id": 23, "label_type": "other" }
    ]

    def __init__(self, 
        label_path,
        pred_bboxes_path=None,
        view_agg_fn="mean",
        scale_with_bbox_confidence=False,
        sort_pred=True,
        **kwargs):

        label_path = Path(label_path)

        if pred_bboxes_path is not None:
            super().__init__(dataset_type="ava", label_path=pred_bboxes_path, **kwargs)

            dic = load_ava_dataset(label_path, 
                temporal_crop_frames=1) # temporal_crop_frames has no effect here
            self.evaluator_gt = dic["data"]

        else:
            super().__init__(dataset_type="ava", label_path=label_path, **kwargs)
            self.evaluator_gt = self.label_parser_ret["data"]

        self.evaluator_annot_joint = self.label_parser_ret["annot_joint"]

        if not scale_with_bbox_confidence:
            # use confidence scores from predicted bboxes
            self.evaluator_annot_joint["confidence"] = 1.0

        self.view_agg_fn = view_agg_fn
        self.sort_pred = sort_pred
        self.excluded_timestamps_path = label_path.parent / f"{label_path.stem}_excluded_timestamps.csv"

    def evaluator(self):
        return ChimpACTEvaluator(self)

