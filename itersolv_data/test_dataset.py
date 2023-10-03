from .AbstractGenerator import _PAD
from .algebra import AlgebraicExpressionGenerator
from .arithmetic import ArithmeticExpressionGenerator
from .listops import ListOpsGenerator
import torch
import framework
import pandas as pd
from dataset.sequence import TextClassifierTestState


class TestDataset(torch.utils.data.IterableDataset):

    def construct_vocab(self):
        if self.in_vocabulary is None:
            self.in_vocabulary = framework.data_structures.WordVocabulary([c for c in self.generator.x_vocab.vocab.itos_])
            self.out_vocabulary = framework.data_structures.WordVocabulary([c for c in self.generator.y_vocab.vocab.itos_])

    def build_input_target(self):
        df = pd.read_csv(f'dataset/itersolv/{self.task_name}/test/nesting-{self.kwargs["nesting"]}_num-operands-{self.kwargs["num_operands"]}.csv')
        inputs = df['input'].tolist()[:self.kwargs['batch_size']]
        target = df['target'].tolist()[:self.kwargs['batch_size']]
        self.X = self.generator.str_to_batch(inputs)
        self.target = self.generator.str_to_batch(target, x=False)

    @property
    def task_name(self):
        if isinstance(self.generator, AlgebraicExpressionGenerator):
            return 'algebra'
        elif isinstance(self.generator, ArithmeticExpressionGenerator):
            return 'arithmetic'
        elif isinstance(self.generator, ListOpsGenerator):
            return 'listops'
    
    def __init__(self, generator, kwargs):
        self.generator = generator
        self.kwargs = kwargs
        self.in_vocabulary = None
        self.out_vocabulary = None
        self.construct_vocab()
        self.build_input_target()

    def __len__(self):
        return self.X.size(0)

    def __iter__(self):
        return self._generate_dict()

    def _generate_dict(self):
        def _sample_len(batch):
            pad_idx = self.generator.y_vocab[_PAD]
            return (batch != pad_idx).sum(-1)

        token_X, token_Y = self.X.argmax(-1), self.target.argmax(-1)
        yield {
                "in": token_X.T,
                "out": token_Y,
                "in_len": _sample_len(token_X),
                "out_len": _sample_len(token_Y),
            }

    def start_test(self) -> TextClassifierTestState:
        return TextClassifierTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                       lambda x: "".join(self.out_vocabulary(x)), max_bad_samples=100)
