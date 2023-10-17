from itersolv_data.algebra import AlgebraicExpressionGenerator
from itersolv_data.ItersolvDataset import ItersolvDataset


class IterSolvAlgebraMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = AlgebraicExpressionGenerator('cuda', specials_in_x=True,
                                                 variables='xy',
                                                 coeff_variables='ab')
        
        self.train_set = ItersolvDataset(generator, 'algebra', 'train', self.helper.args.batch_size)

        self.valid_sets.iid = ItersolvDataset(generator, 'algebra', 'valid_iid', self.helper.args.batch_size)
        
        self.valid_sets.ood = ItersolvDataset(generator, 'algebra', 'valid_ood', self.helper.args.batch_size)
