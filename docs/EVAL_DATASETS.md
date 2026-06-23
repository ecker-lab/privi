
# Primate Behavior Evaluation

For most primate behavior datasets, there are currently no reference implementations for evaluation code available, which hinders community adaption. To mitigate this issue, we provide reference implementations for ChimpACT and PanAf500 in this repo.

Usage: 

```python
from privi.datasets.generic import make_videodataset
from pprint import pprint
dataset, data_loader, _ = make_videodataset(
    dataset_type='chimpact', # or panaf500
    label_path="PATH_TO_LABEL_FILE", # paths in label files are relative to video_base_path
    video_base_path="PATH_TO_VIDEO_FILES", 
    batch_size=16,
)
model = MODEL_TO_EVALUATE()
evaluator = dataset.evaluator()
for batch in data_loader:
    data, labels, indices, metadata = batch
    logits = model(data)
    evaluator.add_batch(labels, logits)
pprint(evaluator.metrics()) # metrics() returns a dict of all metrics relevant for a specific dataset
```

## ChimpACT

### Preprocessing

Run ChimpACT dataset preprocessing as described in the [ChimpACT repository](https://github.com/ShirleyMaxx/ChimpACT). For the test split, download the fixed annotation file as described [here](https://github.com/ShirleyMaxx/AlphaChimp#eval-tracking) under 2 and convert it to CSV. Then run in folder `ChimpACT_processed/annotations/action/`

```bash
for $split in train test_fix val; do
    sed -E 's#^val/images/([^,]*)#\1.mp4#' ${split}_action.csv > ${split}_action.ava.csv
```

to convert the path format from `val/images/Azibo_ObsChimp_2015_11_25_d_clip_23000_24000,...` to `Azibo_ObsChimp_2015_11_25_d_clip_23000_24000.mp4,...`.

For val, I would advise downsampling `val_action.ava.csv` to every 10th frame. It speeds up evaluation and the results are near-identical.

Then use 

```python
dataset, data_loader, _ = make_videodataset(
    dataset_type='chimpact',
    label_path="ChimpACT_processed/annotations/action/val_action.ava.csv",
    video_base_path="ChimpACT_release_v1/videos_full/", 
    batch_size=16,
)
```

### Evaluating

For ChimpACT, `mAP/macro` (mAP) and `mAP/weighted` (mAP_w) are the metrics we used in the paper. `mAP/overall_exclude_extreme_tail` corresponds to the mAP metric used by AlphaChimp and some other baselines where hard-to-detect tail classes are excluded.

## PanAf500

### Preprocessing

Download the [PanAf dataset](https://obrookes.github.io/panaf.github.io/) and run 

```
python privi/preprocessing/panaf500.py --dataset_path data/panaf/panaf500/ --output_dir data/panaf500_ar/
```

Then use 

```python
dataset, data_loader, _ = make_videodataset(
    dataset_type='panaf500',
    label_path="data/val.csv",
    video_base_path="data/panaf500_ar/videos/", 
    batch_size=16,
)
```