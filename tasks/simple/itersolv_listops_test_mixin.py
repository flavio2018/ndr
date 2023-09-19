from itersolv_data.listops import ListOpsGenerator
from itersolv_data.wrapper import GeneratorWrapper


class IterSolvListopsTestMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = ListOpsGenerator('cuda', specials_in_x=True, ops='ias')
        
        train_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "nesting": 2,
            "n_operands": 3,
            "split": 'train',
            "s2e_baseline": True,
        }
        self.train_set = GeneratorWrapper(generator, train_kwargs)
