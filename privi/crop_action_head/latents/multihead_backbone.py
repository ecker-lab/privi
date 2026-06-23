
from privi.crop_action_head.latents.backbone import BackboneLatents
from privi.utils.multi_shuffle_iter import MultiShuffleIter

class MultiheadBackboneLatents(BackboneLatents):
    """
    Produce multiple random orderings of the same backbone batches with a constant memory 
    footprint. This is used to train several classifiers in parallel while only needing to have one
    backbone in memory.

    For eval splits this behaves the same as BackboneLatents. For train split, this utilizes 
    MultiShuffleIter. The batches for the different heads are yielded round robin during train, i.e. 
    [(head1,elem1),(head2,elem1),...,(head1,elem2),...].

    """
    def __init__(self, n_classifiers, buffer_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_classifiers = n_classifiers
        self.buffer_size = buffer_size
        
        assert self.backbone_training_mode == "frozen", "Multi-classifier buffer training requires a frozen backbone."

    def get_iter(self, split):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        if split in ["val", "test"]:
            return super().get_iter(split)
        else:
            iterable = super().get_iter(split)
            iter = MultiShuffleIter(iterable, self.n_classifiers, self.buffer_size)
            iter.num_crops = lambda self: iterable.num_crops()
            return iter

