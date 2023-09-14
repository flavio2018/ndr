from itersolv_data.algebra import AlgebraicExpressionGenerator
from itersolv_data.wrapper import GeneratorWrapper


class IterSolvAlgebraMixin:

    def create_datasets(self):
        self.batch_dim = 1
        generator = AlgebraicExpressionGenerator('cuda', specials_in_x=True)
        train_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "nesting": self.helper.args.itersolv_arithmetics.iid_nesting,
            "num_operands": self.helper.args.itersolv_arithmetics.iid_num_operands,
            "split": 'train',
            "s2e_baseline": True,
        }
        self.train_set = GeneratorWrapper(generator, train_kwargs)

        valid_iid_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "nesting": self.helper.args.itersolv_arithmetics.iid_nesting,
            "num_operands": self.helper.args.itersolv_arithmetics.iid_num_operands,
            "split": 'valid',
            "s2e_baseline": True,
        }
        self.valid_sets.iid = GeneratorWrapper(generator, valid_iid_kwargs)

        valid_ood_kwargs = {
            "batch_size": self.helper.args.batch_size,
            "nesting": self.helper.args.itersolv_arithmetics.ood_nesting,
            "num_operands": self.helper.args.itersolv_arithmetics.ood_num_operands,
            "split": 'valid',
            "s2e_baseline": True,
        }
        self.valid_sets.ood = GeneratorWrapper(generator, valid_ood_kwargs)