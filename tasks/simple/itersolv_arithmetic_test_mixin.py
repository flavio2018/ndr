from itersolv.ndr_dataset import ItersolvDataset


class IterSolvArithmeticTestMixin:

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