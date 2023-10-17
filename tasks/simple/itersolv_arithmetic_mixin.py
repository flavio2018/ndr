from itersolv_data.arithmetic import ArithmeticExpressionGenerator
from itersolv_data.ItersolvDataset import ItersolvDataset


class IterSolvArithmeticMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = ArithmeticExpressionGenerator('cuda', specials_in_x=True)
        
        self.train_set = ItersolvDataset(generator, 'arithmetic', 'train', self.helper.args.batch_size)

        self.valid_sets.iid = ItersolvDataset(generator, 'arithmetic', 'valid_iid', self.helper.args.batch_size)
        
        self.valid_sets.ood = ItersolvDataset(generator, 'arithmetic', 'valid_ood', self.helper.args.batch_size)
