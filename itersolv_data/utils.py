# import hydra
import logging
from .listops.ListOpsGenerator import ListOpsGenerator
# from data.old_arithmetic.ArithmeticGenerator import ArithmeticGenerator
from .arithmetic import ArithmeticExpressionGenerator
from .algebra import AlgebraicExpressionGenerator


def build_generator(cfg):
	
	logging.info('Building data generator...')
	
	if cfg.data.name == 'arithmetic':
		generator = ArithmeticGenerator(
			cfg.device,
			cfg.data.hash_split,
			cfg.data.specials_in_x,
			cfg.data.always_branch_left,
			cfg.data.num_operands,
			cfg.data.max_branches)
		generator.load_sample2split(hydra.utils.get_original_cwd())

	elif cfg.data.name == 'new_arithmetic':
		generator = ArithmeticExpressionGenerator(
			cfg.device,
			cfg.data.specials_in_x,
			cfg.data.min_operand_value,
			cfg.data.max_operand_value,
			cfg.data.modulo,
			cfg.data.operators,
			cfg.data.mini_steps)
	
	elif cfg.data.name == 'listops':
		generator = ListOpsGenerator(
			cfg.device,
			cfg.data.specials_in_x,
			cfg.data.simplify_last,
			cfg.data.mini_steps,
			cfg.data.ops)

	elif cfg.data.name == 'algebra':
		generator = AlgebraicExpressionGenerator(
			cfg.device,
			cfg.data.specials_in_x,
			cfg.data.modulo,
			cfg.data.mini_steps,
			cfg.data.variables,
			cfg.data.coeff_variables)
	
	logging.info('Done.')
	
	return generator


def build_generator_kwargs(cfg):
	
	if cfg.data.name == 'arithmetic':
		generator_kwargs = {
			"batch_size": cfg.data.bs,
			"filtered_swv": cfg.data.filtered_swv,
			"filtered_s2e": cfg.data.filtered_s2e,
			"combiner": cfg.data.combiner,
			"substitute": cfg.data.substitute,
			"split": "train",
			"ops": cfg.data.ops,
		}

	elif cfg.data.name == 'new_arithmetic':
		generator_kwargs = {
			"batch_size": cfg.data.bs, 
			"nesting": cfg.data.nesting, 
			"num_operands": cfg.data.num_operands, 
			"split": 'train',
			"exact": cfg.data.exact, 
			"combiner": cfg.data.combiner, 
			"s2e": cfg.data.s2e,
			"s2e_baseline": cfg.data.s2e_baseline,
		}
	
	elif cfg.data.name == 'listops':
		generator_kwargs = {
			"batch_size": cfg.data.bs,
			"split": "train",
			"exact": cfg.data.exact,
			"max_depth": cfg.data.max_depth,
			"max_args": cfg.data.max_args,
			"combiner": cfg.data.combiner,
			"s2e": cfg.data.filtered_s2e,
			"s2e_baseline": cfg.data.s2e_baseline,
		}

	elif cfg.data.name == 'algebra':
		generator_kwargs = {
			"batch_size": cfg.data.bs,
			"split": "train",
			"exact": cfg.data.exact,
			"nesting": cfg.data.nesting,
			"num_operands": cfg.data.num_operands,
			"combiner": cfg.data.combiner,
			"s2e": cfg.data.s2e,
			"s2e_baseline": cfg.data.s2e_baseline,
		}

	return generator_kwargs
