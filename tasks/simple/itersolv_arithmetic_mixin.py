from itersolv.ndr_dataset import ItersolvDataset


class IterSolvArithmeticMixin:

    def create_datasets(self):
        self.batch_dim = 1
        
        self.train_set = ItersolvDataset(
            'arithmetic',
            'train',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)

        self.valid_sets.iid = ItersolvDataset(
            'arithmetic',
            'valid_iid',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)
        
        self.valid_sets.ood = ItersolvDataset(
            'arithmetic',
            'valid_ood',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)
        
        self.valid_sets.all = ItersolvDataset(
            'arithmetic',
            'valid',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)
