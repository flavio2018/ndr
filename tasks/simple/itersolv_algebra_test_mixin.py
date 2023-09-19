from itersolv_data.algebra import AlgebraicExpressionGenerator
from itersolv_data.wrapper import GeneratorWrapper


class IterSolvAlgebraTestMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = AlgebraicExpressionGenerator('cuda', specials_in_x=True,
                                                 variables='xy',
                                                 coeff_variables='ab')
        train_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "nesting": 2,
            "num_operands": 3,
            "split": 'train',
            "s2e_baseline": True,
        }
        self.train_set = GeneratorWrapper(generator, train_kwargs)
