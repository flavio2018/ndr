from itersolv_data.algebra import AlgebraicExpressionGenerator
from itersolv_data.wrapper import GeneratorWrapper


class IterSolvAlgebraTestMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = AlgebraicExpressionGenerator('cuda', specials_in_x=True,
                                                 variables='xy',
                                                 coeff_variables='ab')
        valid_ood_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "nesting": 2,
            "num_operands": 2,
            "split": 'test',
            "s2e_baseline": True,
            "exact": True
        }
        self.valid_sets.ood_2_2 = GeneratorWrapper(generator, valid_ood_kwargs)
        valid_ood_kwargs['num_operands'] = 3
        self.valid_sets.ood_2_3 = GeneratorWrapper(generator, valid_ood_kwargs)
        valid_ood_kwargs['num_operands'] = 4
        self.valid_sets.ood_2_4 = GeneratorWrapper(generator, valid_ood_kwargs)
        
        valid_ood_kwargs['nesting'] = 3
        valid_ood_kwargs['num_operands'] = 2
        self.valid_sets.ood_3_2 = GeneratorWrapper(generator, valid_ood_kwargs)
        valid_ood_kwargs['num_operands'] = 3
        self.valid_sets.ood_3_3 = GeneratorWrapper(generator, valid_ood_kwargs)
        valid_ood_kwargs['num_operands'] = 4
        self.valid_sets.ood_3_4 = GeneratorWrapper(generator, valid_ood_kwargs)

        valid_ood_kwargs['nesting'] = 4
        valid_ood_kwargs['num_operands'] = 2
        self.valid_sets.ood_4_2 = GeneratorWrapper(generator, valid_ood_kwargs)
        valid_ood_kwargs['num_operands'] = 3
        self.valid_sets.ood_4_3 = GeneratorWrapper(generator, valid_ood_kwargs)
        valid_ood_kwargs['num_operands'] = 4
        self.valid_sets.ood_4_4 = GeneratorWrapper(generator, valid_ood_kwargs)