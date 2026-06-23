class TransformNTimes:
    """
    Wrap another torchvision transform and apply it N times,
    returning a list of transformed samples.
    """
    def __init__(self, transform, n, flatten_lists=True):
        """
        Args:
            transform: a callable (e.g. a torchvision.transforms.Compose) 
                       that takes an input and returns a transformed output.
            n (int): number of times to apply it (with fresh randomness each time).
        """
        self.transform = transform
        self.n = n
        self.flatten_lists = flatten_lists

    def __call__(self, x):
        """
        Args:
            x: input sample (PIL Image, Tensor, etc.)
        Returns:
            List of length n, each entry is transform(x) with independent randomness.
        """
        ret = [ self.transform(x) for _ in range(self.n) ]
        if ret[0] is not None and isinstance(ret[0], (list, tuple)) and self.flatten_lists:
            # Flatten lists of lists into a single list
            ret = [item for sublist in ret for item in sublist]
        return ret