
from privi.datasets.action.video_dataset import VideoDataset

class PretrainDataset(VideoDataset):
    
    def __init__(self, **kwargs):

        super().__init__(dataset_type="csv", **kwargs)