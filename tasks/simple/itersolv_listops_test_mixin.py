from itersolv.listops import ListOpsGenerator
from itersolv.ItersolvDataset import ItersolvDataset


class IterSolvListopsTestMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = ListOpsGenerator('cuda', specials_in_x=True, ops='ias')
        
        self.train_set = ItersolvDataset(generator, 'listops', 'train', self.helper.args.batch_size)
