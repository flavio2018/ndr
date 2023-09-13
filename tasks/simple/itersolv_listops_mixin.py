from itersolv_data.listops import ListOpsGenerator
from itersolv_data.wrapper import GeneratorWrapper


class IterSolvListopsMixin:

    def create_dataset(self):
        self.batch_dim = 1
        generator = ListOpsGenerator('cuda', specials_in_x=True)
        train_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "max_depth": self.helper.args.itersolv_listops.iid_nesting,
            "max_args": self.helper.args.itersolv_listops.iid_num_operands,
            "split": 'train',
            "s2e_baseline": True,
        }
        self.train_set = GeneratorWrapper(generator, train_kwargs)

        valid_iid_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "max_depth": self.helper.args.itersolv_listops.iid_nesting,
            "max_args": self.helper.args.itersolv_listops.iid_num_operands,
            "split": 'valid',
            "s2e_baseline": True,
        }
        self.valid_sets.iid = GeneratorWrapper(generator, valid_iid_kwargs)

        valid_ood_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "max_depth": self.helper.args.itersolv_listops.ood_nesting,
            "max_args": self.helper.args.itersolv_listops.ood_num_operands,
            "split": 'valid',
            "s2e_baseline": True,
        }
        self.valid_sets.ood = GeneratorWrapper(generator, valid_ood_kwargs)
