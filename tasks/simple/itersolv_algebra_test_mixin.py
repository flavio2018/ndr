from itersolv_data.algebra import AlgebraicExpressionGenerator
from itersolv_data.wrapper import GeneratorWrapper


class IterSolvAlgebraTestMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = AlgebraicExpressionGenerator('cuda', specials_in_x=True,
                                                 variables='xy',
                                                 coeff_variables='ab')
        
        self.train_set = ItersolvDataset(generator, 'algebra', 'train', self.helper.args.batch_size)
