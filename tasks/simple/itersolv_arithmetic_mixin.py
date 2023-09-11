from itersolv_data.arithmetic import ArithmeticExpressionGenerator
from itersolv_data.wrapper import GeneratorWrapper


class IterSolvArithmeticMixin:

	def create_datasets(self):
		generator = ArithmeticExpressionGenerator('cuda', specials_in_x=True)
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