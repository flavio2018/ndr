from .AbstractGenerator import _PAD
import framework
import torch


class GeneratorWrapper(torch.utils.data.IterableDataset):

    def construct_vocab(self):
        if self.in_vocabulary is None:
            self.in_vocabulary = framework.data_structures.WordVocabulary([c for c in self.generator.x_vocab.vocab.itos_])
            self.out_vocabulary = framework.data_structures.WordVocabulary([c for c in self.generator.y_vocab.vocab.itos_])

    def __init__(self, generator, kwargs):
        self.generator = generator
        self.kwargs = kwargs
        self.in_vocabulary = None
        self.out_vocabulary = None
        self.construct_vocab()

    def __iter__(self):
        return self._generate_dict()

    def _generate_dict(self):
        def _sample_len(batch):
            pad_idx = self.generator.y_vocab[_PAD]
            return (batch != pad_idx).sum(-1)

        self.max_iter_eval = 10
        self.curr_iter = 0
        
        def _continue():
            if self.kwargs['split'] == 'train':
                return True
            else:
                self.curr_iter += 1
                return self.curr_iter <= self.max_iter_eval

        while _continue():
            X, Y = self.generator.generate_batch(**self.kwargs)
            token_X, token_Y = X.argmax(-1), Y.argmax(-1)
            yield {
                "in": token_X,
                "out": token_Y,
                "in_len": _sample_len(token_X),
                "out_len": _sample_len(token_Y),
            }
