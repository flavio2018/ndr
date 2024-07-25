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

        self.valid_sets.iid.vocabulary = self.train_set.vocabulary
        self.valid_sets.iid._build_ndr_vocab()
        self.valid_sets.ood.vocabulary = self.train_set.vocabulary
        self.valid_sets.ood._build_ndr_vocab()
        self.valid_sets.all.vocabulary = self.train_set.vocabulary
        self.valid_sets.all._build_ndr_vocab()