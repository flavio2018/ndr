import torch
import pandas as pd
import framework
from dataset.sequence import TextClassifierTestState
from itersolv.vocabulary import Vocabulary


class ItersolvDataset(torch.utils.data.IterableDataset):

    def __init__(self, dataset_name, split, train_batch_size, eval_batch_size, device, sos, eos, specials_in_x=False):
        self.in_vocabulary = None
        self.out_vocabulary = None
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.split = split
        self.device = device
        self.specials_in_x = specials_in_x
        self.sos = sos
        self.eos = eos

        self.df = pd.read_csv(f'datasets/{dataset_name}/{split}.csv')
        print(f"{len(self.df)} total samples in {split} split.")
        self._build_vocabulary()
        self._build_ndr_vocab()

    def __iter__(self):
        return self._generate_dict()

    def __len__(self):
        return len(self.df)

    def _build_vocabulary(self):
        x_vocab_chars, y_vocab_chars = self._get_vocabs_chars()
        self.vocabulary = Vocabulary(x_vocab_chars, y_vocab_chars, self.device, self.sos, self.eos, self.specials_in_x)

    def _build_ndr_vocab(self):
        if self.in_vocabulary is None:
            self.in_vocabulary = framework.data_structures.WordVocabulary([c for c in self.vocabulary.x_vocab.vocab.itos_])
            self.out_vocabulary = framework.data_structures.WordVocabulary([c for c in self.vocabulary.y_vocab.vocab.itos_])
    
    def _get_vocabs_chars(self):
        x_chars_sets = self.df['X'].apply(lambda s: set(s))
        y_chars_sets = self.df['Y'].apply(lambda s: set(s))

        x_vocab_chars = set()
        for char_set in x_chars_sets:
            x_vocab_chars |= char_set

        y_vocab_chars = set()
        for char_set in y_chars_sets:
            y_vocab_chars |= char_set

        x_vocab_chars = sorted(list(x_vocab_chars))
        y_vocab_chars = sorted(list(y_vocab_chars))

        return x_vocab_chars, y_vocab_chars

    @property
    def batch_size(self):
        if self.split == 'train':
            return self.train_batch_size
        else:
            return self.eval_batch_size

    def _generate_dict(self):
        def _sample_len(batch):
            pad_idx = self.vocabulary.y_vocab[_PAD]
            return (batch != pad_idx).sum(-1)

        def _continue():
            if self.split == 'train':
                return True
            else:
                self.curr_iter += 1
                return self.curr_iter <= len(self.df) // self.eval_batch_size

        self.curr_iter = 0
        
        while _continue():
            batch_df = self.df.sample(n=self.batch_size)
            X, Y = batch_df['X'].astype(str).tolist(), batch_df['Y'].astype(str).tolist()
            token_X, token_Y = self.vocabulary.str_to_batch(X), self.vocabulary.str_to_batch(Y, x=False)
            yield {
                "in": token_X.T,
                "out": token_Y,
                "in_len": _sample_len(token_X),
                "out_len": _sample_len(token_Y),
            }

    def start_test(self) -> TextClassifierTestState:
        return TextClassifierTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                       lambda x: "".join(self.out_vocabulary(x)), max_bad_samples=100)
