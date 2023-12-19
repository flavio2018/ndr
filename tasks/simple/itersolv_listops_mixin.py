from itersolv.ndr_dataset import ItersolvDataset


class IterSolvListopsMixin:

    def create_datasets(self):
        self.batch_dim = 1

        self.train_set = ItersolvDataset(
            'listops',
            'train',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)

        self.valid_sets.iid = ItersolvDataset(
            'listops',
            'valid_iid',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)
        
        self.valid_sets.ood = ItersolvDataset(
            'listops',
            'valid_ood',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)
        
        self.valid_sets.all = ItersolvDataset(
            'listops',
            'valid',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)
        