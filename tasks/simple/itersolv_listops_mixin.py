from itersolv_data.listops import ListOpsGenerator
from itersolv_data.ItersolvDataset import ItersolvDataset


class IterSolvListopsMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = ListOpsGenerator('cuda', specials_in_x=True, ops='ias')
        
        self.train_set = ItersolvDataset(generator, 'listops', 'train', self.helper.args.batch_size)

        self.valid_sets.iid = ItersolvDataset(generator, 'listops', 'valid_iid', self.helper.args.batch_size)
        
        self.valid_sets.ood = ItersolvDataset(generator, 'listops', 'valid_ood', self.helper.args.batch_size)
        