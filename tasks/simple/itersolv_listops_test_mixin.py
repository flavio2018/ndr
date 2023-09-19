from itersolv_data.listops import ListOpsGenerator
from itersolv_data.wrapper import GeneratorWrapper

class IterSolvListopsTestMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = ListOpsGenerator('cuda', specials_in_x=True, ops='ias')
        
        train_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "max_depth": 2,
            "max_args": 3,
            "split": 'train',
            "s2e_baseline": True,
        }
        self.train_set = GeneratorWrapper(generator, train_kwargs)


        valid_ood_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "max_depth": 2,
            "max_args": 2,
            "split": 'test',
            "s2e_baseline": True,
            "exact": True,
        }
        self.valid_sets.ood_2_2 = GeneratorWrapper(generator, valid_ood_kwargs)
        # valid_ood_kwargs['max_args'] = 3
        # self.valid_sets.ood_2_3 = GeneratorWrapper(generator, valid_ood_kwargs)
        # valid_ood_kwargs['max_args'] = 4
        # self.valid_sets.ood_2_4 = GeneratorWrapper(generator, valid_ood_kwargs)
        
        # valid_ood_kwargs['max_depth'] = 3
        # valid_ood_kwargs['max_args'] = 2
        # self.valid_sets.ood_3_2 = GeneratorWrapper(generator, valid_ood_kwargs)
        # valid_ood_kwargs['max_args'] = 3
        # self.valid_sets.ood_3_3 = GeneratorWrapper(generator, valid_ood_kwargs)
        # valid_ood_kwargs['max_args'] = 4
        # self.valid_sets.ood_3_4 = GeneratorWrapper(generator, valid_ood_kwargs)

        # valid_ood_kwargs['max_depth'] = 4
        # valid_ood_kwargs['max_args'] = 2
        # self.valid_sets.ood_4_2 = GeneratorWrapper(generator, valid_ood_kwargs)
        # valid_ood_kwargs['max_args'] = 3
        # self.valid_sets.ood_4_3 = GeneratorWrapper(generator, valid_ood_kwargs)
        # valid_ood_kwargs['max_args'] = 4
        # self.valid_sets.ood_4_4 = GeneratorWrapper(generator, valid_ood_kwargs)
        