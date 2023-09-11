import pdb
import random
import re
import numpy as np
import string
from ..AbstractGenerator import _HAL


MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
FL_SUM_MOD = "[FLSUM"
PROD_MOD = "[PM"
END = "]"
OPERATORS = {
		"i": MIN,
		"a": MAX,
		"e": MED,
		"s": SUM_MOD,
		"f": FIRST,
		"l": LAST,
		"u": FL_SUM_MOD,
		"p": PROD_MOD
	}
OPERATORS_TOK = {
		"i": "MIN",
		"a": "MAX",
		"e": "MED",
		"s": "SM",
		"f": "FIRST",
		"l": "LAST",
		"u": "FLSUM",
		"p": "PM",
	}


def listops_tokens(ops):
	used_operators = [OPERATORS_TOK[op] for op in ops]
	return used_operators + list(string.digits + '/[] ')    


class ListOpsTree:

	def __init__(self, max_depth, max_args, simplify_last=True, mini_steps=False, ops='iaes'):
		self.max_depth = max_depth
		self.max_args = max_args
		self.value_p = .25
		self.tree = ()
		self.steps = None
		self.sub_expr = None
		self.values = None
		self.depth = None
		self.simplify_last = simplify_last
		self.mini_steps = mini_steps
		self.OPERATORS = [OPERATORS[op] for op in ops]
	
	def _generate_tree(self, depth):
		# set probability with which value will be tree or number
		if depth == 1:
			r = 0
		elif depth <= self.max_depth:
			r = random.random()
		else:
			r = 1

		# pick a value at random
		if r > self.value_p:
			value = random.choice(range(10))
			return value

		# choose number of values of expression and recursively 
		# generate values using the same procedure
		else:

			if random.random() > 0.9:
				return random.choice(range(10))

			num_values = random.randint(1, self.max_args)
			
			values = []
			for _ in range(num_values):
				values.append(self._generate_tree(depth + 1))

			# randomly choose op to apply on values
			op = random.choice(self.OPERATORS)
			# build subtree and return to recurse
			t = (op, values[0])
			for value in values[1:]:
				t = (t, value)
			t = (t, END)
		return t
	
	def _generate_depth_k_tree(self, k):
		assert k <= self.max_depth, f'Depth can be at most {self.max_depth}'
		
		if k == 0:
			return random.choice(range(10))
		
		else:

			if self.max_args == 0:
				assert k == 0, "Cannot build expression with 0 args and depth > 0"
				return random.choice(range(10))

			op = random.choice(self.OPERATORS)
			num_values = self.max_args
			num_depth_k_branches = np.random.randint(1, num_values+1)
			pos_depth_k_branches = set(np.random.choice(range(num_values), num_depth_k_branches, replace=False))

			values = []
			for value_pos in range(num_values):
				if value_pos in pos_depth_k_branches:
					values.append(self._generate_depth_k_tree(k-1))
				else:
					values.append(self._generate_depth_k_tree(0))
			
			t = (op, values[0])
			for value in values[1:]:
				t = (t, value)
			t = (t, END)
			return t
			
	def generate_tree(self, depth=None):
		
		if self.max_args == 0:  # halting case
			depth = 0

		if depth is None:
			self.tree = self._generate_tree(1)
		else:
			self.tree = self._generate_depth_k_tree(depth)
		self.depth = self._compute_depth()
		self.steps, self.sub_expr, self.values = self._compute_steps()
   

	@property
	def solution(self):
		if self.steps:
			return self.steps[-1]
		elif self.tree:
			return self.to_string()
		else:
			return None        
	
	def _compute_depth(self):
		if self.tree:
			tree_string = self.to_string()
			depth, max_depth = 0, 0
			for c in tree_string:
				if c == '[':
					depth += 1
				elif c == ']':
					depth -= 1
				if depth > max_depth:
					max_depth = depth
			return max_depth
		else:
			return None
	
	def _to_string(self, t, parens=True):
		if isinstance(t, str):
			return t
		elif isinstance(t, int):
			return str(t)
		else:
			if parens:
				return '( ' + self._to_string(t[0], parens=parens) + ' ' + self._to_string(t[1], parens=parens) + ' )'
			else:
				return self._to_string(t[0], parens=parens) + self._to_string(t[1], parens=parens)
	
	def to_string(self):
		return self._to_string(self.tree, parens=False)
	

	def _compute_op(self, op, args):
		if op == 'MIN':
			sub_expr_value = min(args)
		elif op == 'MAX':
			sub_expr_value = max(args)
		elif op == 'MED':
			sub_expr_value = int(np.median(args))
		elif op == 'SM':
			sub_expr_value = np.sum(args) % 10
		elif op == 'FIRST':
			sub_expr_value = args[0]
		elif op == 'LAST':
			sub_expr_value = args[-1]
		elif op == 'FLSUM':
			sub_expr_value = sum([args[0], args[-1]]) % 10
		elif op == 'PM':
			sub_expr_value = np.prod(args) % 10
		else:
			print(f"Operation {op} not allowed.")
		return sub_expr_value
			

	def simplify_one_lvl(self, sample):
		"""Create a simplified version of the expression substituting
		one of the sub-expressions with its result.

		:param sample: the original expression in string format.
		"""
		simplified_sample = sample
		target = sample
		sub_expr_value = sample
		
		sub_expression_re = re.compile(r'\[[A-Z]+[\d{1}]+\]')
		sub_expressions = sub_expression_re.findall(sample)

		if sub_expressions != []:
			if self.simplify_last:
				sub_expression = sub_expressions[-1]
			else:
				sub_expression = sub_expressions[0]

			sub_expression_no_parens = sub_expression[1:-1]
			op = re.findall(r'[A-Z]+', sub_expression_no_parens)[0]
			args = [int(v) for v in sub_expression_no_parens[len(op):]]

			if self.mini_steps and (len(args) > 1):
				target = sub_expression_no_parens[:len(op)+2]
				mini_step_value = self._compute_op(op, args[:2])
				replacement = sub_expression_no_parens[:len(op)] + str(mini_step_value)
				# pdb.set_trace()
				sub_expr_value = replacement
			else:
				target = sub_expression
				sub_expr_value = self._compute_op(op, args)
				replacement = str(sub_expr_value)

			simplified_sample = sample.replace(target, replacement)
		return simplified_sample, target, str(sub_expr_value)


	def _compute_steps(self):
		sample = self.to_string()
		steps = [sample]
		values = []
		sub_expr = []
		simplified_sample, sub_expression, sub_expr_value = self.simplify_one_lvl(sample)

		while (sample != simplified_sample):
			steps.append(simplified_sample)
			values.append(sub_expr_value)
			sub_expr.append(sub_expression)
			sample = simplified_sample
			simplified_sample, sub_expression, sub_expr_value = self.simplify_one_lvl(sample)

		if (len(steps) == 1) and (len(sample) == 1) and (str.isdigit(sample)):  # case halting
			values.append(_HAL)
		
		else:
			values.append(sample)
		
		sub_expr.append(sample)
		return steps, sub_expr, values

	def get_start_end_subexpression(self):
	    as_tokenized = lambda s: (s.replace('MAX', 'A')
	    						   .replace('MIN', 'I')
	    						   .replace('MED', 'E')
	    						   .replace('SM', 'M')
	    						   .replace('FIRST', 'F')
	    						   .replace('LAST', 'L')
	    						   .replace('FLSUM', 'U')
	    						   .replace('PM', 'P'))
	    
	    pattern = as_tokenized(self.sub_expr[0].replace('[', '\[').replace(']', '\]'))
	    expr = as_tokenized(self.to_string())
	    match = re.search(pattern, expr)
	    return match.start(), match.end()
