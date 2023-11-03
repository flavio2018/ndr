import framework
import torch
from dataset.sequence import TextClassifierTestState
from glob import glob
import pandas as pd
from .AbstractGenerator import _PAD


class ItersolvDataset(torch.utils.data.IterableDataset):

    def construct_vocab(self):
        if self.in_vocabulary is None:
            self.in_vocabulary = framework.data_structures.WordVocabulary([c for c in self.generator.x_vocab.vocab.itos_])
            self.out_vocabulary = framework.data_structures.WordVocabulary([c for c in self.generator.y_vocab.vocab.itos_])

    def __init__(self, generator, task_name, split, batch_size):
        self.generator = generator
        self.in_vocabulary = None
        self.out_vocabulary = None
        self.construct_vocab()
        self.batch_size = batch_size if split == 'train' else batch_size // 32
        self.split = split

        files_glob = glob(f'dataset/itersolv/{task_name}/{task_name}_*_{split}.csv')
        self.df = pd.concat([pd.read_csv(f) for f in files_glob])
        print(f"Loaded {len(files_glob)} files.")
        print(f"{len(self.df)} total samples in {split} split.")
    
    def __iter__(self):
        return self._generate_dict()

    def __len__(self):
        # used in test
        return len(self.df)

    def _generate_dict(self):
        def _sample_len(batch):
            pad_idx = self.generator.y_vocab[_PAD]
            return (batch != pad_idx).sum(-1)

        
        def _continue():
            if self.split == 'train':
                return True
            else:
                self.curr_iter += 1
                return self.curr_iter <= len(self.df) // self.batch_size

        self.curr_iter = 0
        
        while _continue():
            batch_df = self.df.sample(n=self.batch_size)
            X, Y = batch_df['X'].astype(str).tolist(), batch_df['Y'].astype(str).tolist()
            batch_X, batch_Y = self.generator.str_to_batch(X), self.generator.str_to_batch(Y, x=False)
            token_X, token_Y = batch_X.argmax(-1), batch_Y.argmax(-1)
            yield {
                "in": token_X.T,
                "out": token_Y,
                "in_len": _sample_len(token_X),
                "out_len": _sample_len(token_Y),
            }

    def start_test(self) -> TextClassifierTestState:
        return TextClassifierTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                       lambda x: "".join(self.out_vocabulary(x)), max_bad_samples=100)
