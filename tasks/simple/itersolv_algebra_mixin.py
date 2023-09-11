from itersolv_data.algebra import AlgebraicExpressionGenerator
from itersolv_data.wrapper import GeneratorWrapper


class IterSolvArithmeticMixin:

	def create_dataset(self):
		generator = AlgebraicExpressionGenerator('cuda', specials_in_x=True)
		train_kwargs = {
		    "batch_size": 4,
		    "nesting": 2,
		    "num_operands": 2,
		}
		self.train_set = GeneratorWrapper(gen, kwargs)

		valid_iid_kwargs = {
		    "batch_size": 4,
		    "nesting": 2,
		    "num_operands": 2,
		    "split": 'valid',
		}
		self.valid_sets.iid = GeneratorWrapper(gen, kwargs)

		valid_ood_kwargs = {
		    "batch_size": 4,
		    "nesting": 4,
		    "num_operands": 4,
		    "split": 'valid',
		}
		self.valid_sets.ood = GeneratorWrapper(gen, kwargs)