from itersolv.ndr_dataset import ItersolvDataset


class IterSolvLogicMixin:

    def create_datasets(self):
        self.batch_dim = 1

        self.train_set = ItersolvDataset(
            'logic',
            'train',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)

        self.valid_sets.iid = ItersolvDataset(
            'logic',
            'valid_iid',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)

        self.valid_sets.ood = ItersolvDataset(
            'logic',
            'valid_ood',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)

        self.valid_sets.all = ItersolvDataset(
            'logic',
            'valid',
            self.helper.args.batch_size,
            self.helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False)
