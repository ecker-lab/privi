import logging
import random


logger = logging.getLogger(__name__)

class _MultiShuffleIter:
    current_classifier : int = 0

    def __init__(self, iterable, n_classifiers : int, buffer_size: int):
        self.iter = iter(iterable)
        self.n_classifiers = n_classifiers
        self.buffer_size = buffer_size

        self.total = len(iterable)
        self.produced = 0

        # buffer of precomputed batches
        self.buffer = []  # list of dict entries: {"local": sample, "global": sample|None, "seen": [bool]*n,}

        # fill buffer
        self.fill_buffer()

    def fill_buffer(self):
        while (len(self.buffer) < self.buffer_size):
            if not self.produce_one():
                break

    def produce_one(self):
        try:
            data = next(self.iter)
        except StopIteration:
            return False
        
        entry = {**data, "seen": [False] * self.n_classifiers}
        self.buffer.append(entry)
        self.produced += 1

        return True

    def __iter__(self):
        return self
    
    def __next__(self):

        self.fill_buffer()

        if not self.buffer:
            raise StopIteration

        candidates = [i for i in range(len(self.buffer)) if not self.buffer[i]["seen"][self.current_classifier]]

        if not candidates:
            # temporary overfill if possible
            if self.produce_one():
                logger.warning("Buffer temporarily overfilled to provide new batch for classifier %d", self.current_classifier)
                # newly added entry is unseen for all
                candidates = [len(self.buffer) - 1]
            else:
                logger.warning("buffer not empty, but no more data available to produce a new batch for classifier %d", self.current_classifier)
                if len(self.buffer) > 0:
                    logger.error("First head has no training data anymore, but there are still batches that have not been seen by every head, head %d", self.current_classifier)
                raise StopIteration

        sel_idx = random.randrange(0, len(candidates))
        buf_idx = candidates[sel_idx]

        #print(f"\n[Classifier {h} Step {step}] Selected batch {buf_idx}", end="", flush=True)

        entry = self.buffer[buf_idx]

        entry["seen"][self.current_classifier] = True
        if all(entry["seen"]):
            self.buffer.pop(buf_idx)
        self.current_classifier = (self.current_classifier + 1) % self.n_classifiers

        return {k : entry[k] for k in ["local", "global"]}
    
class MultiShuffleIter:

    def __init__(self, iterable, n_classifiers : int, buffer_size: int):
        """"
        Create a special iterator on top of an iterable which must support len().

        This is designed to create `n_classifier` different random orderings of the elements in 
        `iterable` (e.g. for training different heads on top of a shared backbone). To do so it       
        will fill a buffer of `buffer_size` with entries from `iterable`. The `n_classifier`
        different random orderings are returned round robin, i.e. 
        [(head1,elem1),(head2,elem1),...,(head1,elem2),...].

        Note that len(self) returns the length of iterable. iter(self) will yield
        n_classifiers*len(self) elements.
        """
        self.iterable = iterable
        self.n_classifiers = n_classifiers
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.iterable)
    
    def __iter__(self):
        return _MultiShuffleIter(self.iterable, self.n_classifiers, self.buffer_size)