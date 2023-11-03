from itersolv.arithmetic import ArithmeticExpressionGenerator
from itersolv.wrapper import GeneratorWrapper


class IterSolvArithmeticTestMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = ArithmeticExpressionGenerator('cuda', specials_in_x=True)
        
        train_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "nesting": 2,
            "num_operands": 3,
            "split": 'train',
            "s2e_baseline": True,
        }
        self.train_set = GeneratorWrapper(generator, train_kwargs)