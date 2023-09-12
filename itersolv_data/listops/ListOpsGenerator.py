import random
from ..AbstractGenerator import AbstractGenerator, _SOS, _EOS, _SEP, _HAL
from ..listops.ListOpsTree import ListOpsTree, listops_tokens


class ListOpsGenerator(AbstractGenerator):

	def __init__(self, device, specials_in_x, simplify_last=False, mini_steps=False, ops='iaes'):
		super().__init__(listops_tokens(ops), listops_tokens(ops), device, specials_in_x)
		self.simplify_last = simplify_last
		self.mini_steps = mini_steps
		self.ops = ops
		self.process_ops = lambda s: (s.replace('MAX', 'A')
									   .replace('MIN', 'I')
									   .replace('MED', 'E')
									   .replace('SM', 'M')
									   .replace('FIRST', 'F')
									   .replace('LAST', 'L')
									   .replace('FLSUM', 'U')
									   .replace('PM', 'P'))


	def _tokenize_sample(self, sample: str) -> list:
		"""Override method from AbstractGenerator to customize tokenization."""
		sample = self.process_ops(sample)
		tokenized_sample = [c for c in sample]
		output = []
		
		for token in tokenized_sample:
			
			if token == 'A':
				output.append('MAX')
			
			elif token == 'I':
				output.append('MIN')
			
			elif token == 'E':
				output.append('MED')
			
			elif token == 'M':
				output.append('SM')
			
			elif token == 'F':
				output.append('FIRST')
			
			elif token == 'L':
				output.append('LAST')
			
			elif token == 'U':
				output.append('FLSUM')
			
			elif token == 'P':
				output.append('PM')
			
			else:
				output.append(token)

		return output


	def generate_batch(self, batch_size, max_depth, max_args, split='train', exact=False, combiner=False, s2e=False, s2e_baseline=False, simplify=False):
		samples = [self._generate_sample_in_split(max_depth, max_args, split, exact) for _ in range(batch_size)]
		self.subexpressions_positions = [sample.get_start_end_subexpression() for sample in samples]
		X_str, Y_str = self._build_simplify_w_value(samples)

		if combiner:
			Y_com_str = self._build_combiner_target(samples)
			return self.str_to_batch(X_str), (self.str_to_batch(Y_str, x=False), self.str_to_batch(Y_com_str, x=False)) 
		
		elif s2e:
			Y_str = self._build_s2e_target(samples)
			return self.str_to_batch(X_str), self.str_to_batch(Y_str, x=False)

		elif s2e_baseline:
			Y_str = self._build_s2e_baseline_target(samples)
			return self.str_to_batch(X_str), self.str_to_batch(Y_str, x=False)

		elif simplify:
			Y_com_str = self._build_simplify_target(samples)
			return self.str_to_batch(X_str), self.str_to_batch(Y_com_str, x=False)
		
		else:
			return self.str_to_batch(X_str), self.str_to_batch(Y_str, x=False)


	def _generate_sample_in_split(self, max_depth, max_args, split, exact):
		tree = ListOpsTree(
			max_depth=max_depth,
			max_args=max_args,
			simplify_last=self.simplify_last,
			mini_steps=self.mini_steps,
			ops=self.ops)
		current_split = ''

		while current_split != split:

			if exact:
				tree.generate_tree(max_depth)

			else:
				tree.generate_tree()

			if isinstance(tree.tree, int):  # case halting
				current_split = 'train'

			else:
				sample_hash = hash(tree.to_string())

				if sample_hash % 3 == 0:
					current_split = 'train'

				elif sample_hash % 3 == 1:
					current_split = 'valid'

				else:
					current_split = 'test'

		return tree


	def _build_simplify_w_value(self, samples):
		X_str = []
		Y_str = []

		for sample in samples:
			X_str.append(sample.steps[0])

			if sample.values[0] == _HAL:
				Y_str.append(f"{_SOS}{_HAL}{_EOS}")
			
			else:	
				Y_str.append(f"{_SOS}{sample.sub_expr[0]}{_SEP}{sample.values[0]}{_EOS}")
		
		return X_str, Y_str


	def _build_s2e_target(self, samples):
		return [sample.steps[-1] for sample in samples]


	def _build_s2e_baseline_target(self, samples):
		s2e_target = self._build_s2e_target(samples)
		return [f"{sample}" for sample in s2e_target]

	def _build_combiner_target(self, samples):
		return [sample.steps[1] for sample in samples]

	def _build_simplify_target(self, samples):
		combiner_target = self._build_combiner_target(samples)
		return [f"{_SOS}{sample}{_EOS}" for sample in combiner_target]
