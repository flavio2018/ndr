import pdb
import copy
import sympy
import numpy as np
import string
import logging

from .AbstractGenerator import AbstractGenerator, _SEP, _SOS, _EOS, _HAL


class Monomial:
    
    def __init__(self, coefficient=0, string_variables="", modulo=100):
        self.coefficient = coefficient
        self.string_variables = string_variables
        self.is_instantiated = False
        self.modulo = modulo
        
    def expand(self, num_terms):
        logging.debug(f"Expanding {self}")
        return self._expand_arithmetics(num_terms)
    
    def instantiate(self, coefficient=0, string_variables=""):        
        logging.debug(f"Instantiating {self}")
        
        self._add_variables(string_variables)
        self._add_coefficient(coefficient)
        if self.string_variables.strip() != "":
            self.variables = sympy.symbols(self.string_variables)
        else:
            self.variables = []
        self._build()
        self.is_instantiated = True

    def _add_coefficient(self, coefficient):
        self.coefficient += coefficient

    def _add_variables(self, string_variables):
        self.string_variables = self.string_variables + ' ' + string_variables
       
    def _reduce_coeff_modulo(self):
        self.coefficient = 0 if self.coefficient == 0 else self.coefficient % (np.sign(self.coefficient) * self.modulo)

    def _build(self):        
        logging.debug(f"Building {self}")
        
        self._reduce_coeff_modulo()
        self.value = self.coefficient

        if isinstance(self.variables, sympy.Symbol):
            self.value *= self.variables
        
        else:
            for v in self.variables:
                self.value *= v
        
    def from_sympy(self, sympy_monomial):
        string_variables = ''

        coeff = sympy_monomial
        for v in sympy_monomial.free_symbols:
            coeff = coeff.coeff(v)
            string_variables += f'{str(v)} '
            
        self.instantiate(coeff, string_variables)
        return self
          
    def _expand_arithmetics(self, num_terms):
        assert num_terms > 1, f"Arithmetic operation should involve at least 2 operands, asked {num_terms}"

        kwargs = {
            "string_variables": self.string_variables
        }
            
        return [Monomial(**kwargs) for _ in range(num_terms)]

    def same_variables_as(self, other):
        assert isinstance(other, Monomial), "Other is not a Monomial."
        return set(self.string_variables.strip().split()) == set(other.string_variables.strip().split())
    
    def __repr__(self):
        if not self.is_instantiated:
            return f'Monomial({str(self.coefficient)}, {self.string_variables})'
        else:
            sign = '+' if self.coefficient >= 0 else ''
            if self.value == 0:
                return sign + '*'.join(['0'] + self.string_variables.split())
            else:
                return sign + str(self.value)
        
    def __add__(self, addvalue):
        
        if self.same_variables_as(addvalue):
            sympy_res = self.value + addvalue.value
            
            if sympy_res != 0:
                return Monomial().from_sympy(sympy_res)

            else:
                zero_monomial = Monomial(coefficient=0, string_variables=self.string_variables)
                zero_monomial.instantiate()
                return zero_monomial
        
        else:
            sympy_res = self.value + addvalue.value
            return Binomial().from_sympy(sympy_res)

    def __eq__(self, other):
        return self.value == other.value


class Binomial:
    
    def __init__(self, coefficient=1, first_term=None, second_term=None, modulo=100):
        self.coefficient = coefficient
        self.first_term = first_term
        self.second_term = second_term
        self.is_instantiated = False
        self.modulo = modulo

        if ((coefficient is not None) and
            (first_term is not None) and (first_term.is_instantiated) and
            (second_term is not None) and (second_term.is_instantiated)):
            self.instantiate(coefficient, first_term, second_term)

    def instantiate(self, terms_coefficients=None, first_term=None, second_term=None, coefficient=1):
        logging.debug(f"Instantiating {self}")

        if first_term is not None:
            self.first_term = first_term
        else:
            self.first_term.instantiate(coefficient=terms_coefficients[0])

        if second_term is not None:
            self.second_term = second_term
        else:
            self.second_term.instantiate(coefficient=terms_coefficients[1])
        
        if isinstance(self.coefficient, Monomial):
            self.coefficient.instantiate()
        else:
            if coefficient != 1:
                self.coefficient = coefficient
            
        self._build()
        self.is_instantiated = True

    def _build(self):
        logging.debug(f"Building {self}")
        
        assert self.coefficient is not None
        assert self.first_term is not None
        assert self.second_term is not None

        binomial = self.first_term.value + self.second_term.value
        
        if isinstance(self.coefficient, Monomial):
            self.value = self.coefficient.value * binomial
        else:
            self.value = self.coefficient * binomial
        
        self.value = sympy.factor(self.value)

        if self.value != 0:
            # make sure coeff is in modulo range
            args_modulo = [a for a in self.value.args]
            if not isinstance(args_modulo[0], sympy.core.symbol.Symbol):
                args_modulo[0] = args_modulo[0] % 100 if args_modulo[0] > 0 else args_modulo[0] % -100
            self.value = self.value.func(*args_modulo)

    def from_sympy(self, sympy_binomial):
        first_term, second_term = sympy_binomial.args

        if first_term != 0:
            first_term = Monomial().from_sympy(first_term)

        else:
            first_term = Monomial(coefficient=0, string_variables=first_term.string_variables)

        if second_term != 0:
            second_term = Monomial().from_sympy(second_term)

        else:
            second_term = Monomial(coefficient=0, string_variables=second_term.string_variables)

        self.instantiate(first_term=first_term, second_term=second_term)
        return self
    
    def expand(self, num_operands):
        logging.debug(f"Expanding {self}")
        return self._expand_binomial()
        # return self._expand_arithmetics(num_operands) if np.random.rand() > .5 else self._expand_binomial()
    
    def _expand_binomial(self):
        first_term_copy = copy.deepcopy(self.first_term)
        second_term_copy = copy.deepcopy(self.second_term)
        first_term_copy._add_coefficient(self.coefficient.coefficient)
        first_term_copy._add_variables(self.coefficient.string_variables)
        second_term_copy._add_coefficient(self.coefficient.coefficient)
        second_term_copy._add_variables(self.coefficient.string_variables)
        return [first_term_copy, second_term_copy]
    
    def _expand_arithmetics(self, num_terms):
        assert num_terms > 1, f"Arithmetic operation should involve at least 2 operands, asked {num_terms}"

        return [Binomial(coefficient=self.coefficient, first_term=self.first_term, second_term=self.second_term) for _ in range(num_terms)]

    def __repr__(self):
        if not self.is_instantiated:
            return f'Binomial({str(self.coefficient)}, {str(self.first_term)}, {str(self.second_term)})'
        else:
            if isinstance(self.coefficient, Monomial):
                sign = '+' if self.coefficient.coefficient >= 0 else ''
            else:            
                sign = '+' if self.coefficient >= 0 else ''
            return sign + str(self.value).replace(' ', '')

    def __add__(self, addvalue):
        assert (self.first_term == addvalue.first_term) and (self.second_term == addvalue.second_term), "Binomials should have the same terms to be added."
        return Binomial().from_sympy(self.value + addvalue.value)


class SquareOfBinomial:
    
    def __init__(self):
        pass
    
    def expand(self):
        pass
    
    def _expand_square_of_binomial(self):
        pass
    
    def _expand_arithmetics(self):
        pass

    def __repr__(self):
        pass

    def __add__(self, addvalue):
        pass


class AlgebraicExpression:

    def __init__(self, variables='xyz', coeff_variables='abc', mini_steps=False, modulo=100, simplify_signs=False):
        self.root = None
        self.tree = []
        self.steps = []
        self.variables = variables
        self.coeff_variables = coeff_variables
        self.mini_steps = mini_steps
        self.modulo = modulo
        self.simplify_signs = simplify_signs

    def build(self, nesting, num_operands, exact=False):
        self.expression_string = ''
        self.max_num_operands = num_operands
        self.steps = []
        self.input_nesting = nesting+1

        if num_operands == 1 and nesting == 1:
            # is exact even if not specified by user
            exact = True
        
        monomial_root = np.random.rand() > .5

        if monomial_root:
            root = Monomial()
        else:
            first_term_variables = self._sample_variables(self.variables)
            second_term_variables = self._sample_variables(self.variables, different_from=first_term_variables)
            coefficient_variables = self._sample_variables(self.coeff_variables)
            root = Binomial(
                first_term=Monomial(string_variables=first_term_variables),
                second_term=Monomial(string_variables=second_term_variables),
                coefficient=Monomial(coefficient=np.random.randint(-99, 99), string_variables=coefficient_variables))
        self.tree = self._make_tree([root], current_nesting=nesting+1, exact=exact)[0]
        logging.debug(f"Tree: {self.tree}")
        self._instantiate(self._sample_variables(self.variables))
        self.expression_string = self._build_str_repr()
        self._compute_steps()
        
    def _sample_variables(self, variables, different_from=''):
        return_variables = different_from
 
        while return_variables == different_from:
            num_variables = np.random.randint(1, len(variables)+1)
            which_variables = np.random.choice(list(range(len(variables))), size=num_variables, replace=False)
            which_variables.sort()
            return_variables = ' '.join(variables[v_idx] for v_idx in which_variables)

        return return_variables

    def _make_tree(self, subtree, current_nesting, exact=False):
        if exact:

            if self.max_num_operands == 1:
                assert self.input_nesting == 2, "Cannot request single op expression of nesting > 1."
                return subtree
            num_operands = self.max_num_operands
        
        else:

            # case monomial or factorized binomial, halting condition
            nesting_lvl = np.random.randint(1, current_nesting+1)

            if (current_nesting == self.input_nesting) and (nesting_lvl == 1):
                num_operands = np.random.randint(1, self.max_num_operands+1)

                if num_operands == 1:
                    return subtree

            num_operands = np.random.randint(2, self.max_num_operands+1)
        
        num_nesting_pts = np.random.randint(1, len(subtree)+1)
        
        if exact:
            depth_nesting_pts = [current_nesting]*num_nesting_pts
        
        else:
            min_nes = 2 if len(subtree) == 1 else 1
            depth_nesting_pts = np.random.randint(min_nes, current_nesting+1, num_nesting_pts)
        
        nesting_pts = list(set(np.random.choice(range(len(subtree)), num_nesting_pts, replace=False)))
        nesting_pts_depth = { p: d for (p, d) in zip(nesting_pts, depth_nesting_pts) }
        
        expression = []
        for pos in range(len(subtree)):
            
            copy_of_element = copy.deepcopy(subtree[pos])

            if pos in nesting_pts_depth:
                
                if nesting_pts_depth[pos] == 1:
                    expression.append(copy_of_element)

                else:
                    expanded_pt = copy_of_element.expand(num_operands)
                    expression.append(self._make_tree(expanded_pt, nesting_pts_depth[pos]-1, exact=exact))
            else:
                expression.append(copy_of_element)
                
        return expression
        
    def _instantiate(self, variables, expression=None):
        
        if expression is None:
            expression = self.tree

        if isinstance(expression, list):

            for term in expression:
                        
                if isinstance(term, list):
                    self._instantiate(variables, term)

                else:
                    self._instantiate_term(term, variables)

        else:
            self._instantiate_term(expression, variables)                    

    def _instantiate_term(self, term, variables):
        if isinstance(term, Monomial):
            coeff = np.random.randint(-99, 99)
            
            if term.string_variables != "":  # coming from a binomial
                term.instantiate(coefficient=coeff)
            
            else:
                term.instantiate(coeff, variables)
            
        elif isinstance(term, Binomial):
            coeff = np.random.randint(-99, 99)
            coeff_first_term_binom = np.random.randint(-99, 99)
            coeff_second_term_binom = np.random.randint(-99, 99)
            term.instantiate(coefficient=coeff, terms_coefficients=[coeff_first_term_binom, coeff_second_term_binom])

        elif isinstance(term, SquareOfBinomial):
            raise NotImplementedError()

    def _compute_steps(self, expression=None):
        logging.debug(f"Computing steps for {self}")

        self.steps = []
        expression = copy.deepcopy(self.tree)
        
        reduced_expression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(expression)
        self.steps.append((copy.deepcopy(reduced_expression), subexpression_string, subexpression_value))
        
        while isinstance(reduced_expression, list):
            reduced_expression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(reduced_expression)
            self.steps.append((copy.deepcopy(reduced_expression), subexpression_string, subexpression_value))
            
    def _compute_rightmost_deepmost_step(self, expression=None):

        if expression is None:
            assert self.tree is not None, "Cannot evaluate before building expression"
            expression = copy.deepcopy(self.tree)
        
        expression_types = set([type(v) for v in expression]) if isinstance(expression, list) else [type(expression)]
        
        if list in expression_types:
            
            for value_pos in range(len(expression)-1, -1, -1):
                value = expression[value_pos]
                
                if isinstance(value, list):
                    new_subexpression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(value)
                    reduced_expression = expression
                    reduced_expression[value_pos] = new_subexpression
                    
                    return (reduced_expression, subexpression_string, subexpression_value)
        
        elif Monomial in expression_types:

            # if all monomials have the same variables or there is only one monomial
            if isinstance(expression, Monomial):
                reduced_expression = _HAL
                expression_value = _HAL
                return (reduced_expression, self._build_str_repr(expression), expression_value)

            elif expression[0].same_variables_as(expression[1]) or (len(expression) == 1):
                return self._compute_monomial_arithmetics_step(expression)
            
            else:
                return self._compute_binomial_factorization_step(expression)
        
        elif Binomial in expression_types:

            if isinstance(expression, Binomial):
                reduced_expression = _HAL
                expression_value = _HAL
                return (reduced_expression, self._build_str_repr(expression), expression_value)

            else:
                return self._compute_binomial_arithmetics_step(expression)

        elif SquareOfBinomial in expression_types:
            logging.debug('squared binomial arithmetics')
            raise NotImplementedError

    def _compute_monomial_arithmetics_step(self, expression):
        logging.debug('Simplification step: monomial arithmetics')
                
        if self.mini_steps:
            expression_string = '+'.join(str(v) for v in expression[:2])
            value = expression[0] + expression[1]
            
            if len(expression) > 2:
                reduced_expression = [value] + expression[2:]

                if self.simplify_signs:
                    expression_string = self._simplify_signs(expression_string)
                
                return (reduced_expression, expression_string, value)
            
            else:
                reduced_expression = value
                
                if self.simplify_signs:
                    expression_string = self._simplify_signs(expression_string)
            
                return (reduced_expression, f"({expression_string})", value)
        
        else:
            expression_string = '+'.join(str(v) for v in expression)
            value = expression[0]

            for term in expression[1:]:
                value += term

            if self.simplify_signs:
                expression_string = self._simplify_signs(expression_string)

            return (value, f"({expression_string})", value)

    def _compute_binomial_arithmetics_step(self, expression):
        logging.debug('Simplification step: binomial arithmetics')
        
        if self.mini_steps:
            pass
        
        else:
            expression_string = '+'.join(str(v) for v in expression)
            value = expression[0]

            for binomial_term in expression[1:]:
                value += binomial_term

            if self.simplify_signs:
                expression_string = self._simplify_signs(expression_string)

            return (value, f"({expression_string})", value)

    def _compute_binomial_factorization_step(self, expression):
        logging.debug('Simplification step: binomial factorization')

        expression_string = '+'.join(str(v) for v in expression)

        # if one of the two terms is 0, return a Monomial
        if (expression[0].value == 0) or (expression[1].value == 0):
            sympy_value = expression[0].value + expression[1].value
            value = Monomial().from_sympy(sympy_value)

        else:
            binomial_sympy = expression[0].value + expression[1].value
            factorized_binomial_sympy = sympy.factor(binomial_sympy)
            if isinstance(factorized_binomial_sympy, sympy.core.add.Add):  # no factor
                value = Binomial().from_sympy(factorized_binomial_sympy)

            else:
                coefficient = factorized_binomial_sympy / factorized_binomial_sympy.args[-1]
                minimal_binomial_sympy = factorized_binomial_sympy.args[-1]
                value = Binomial(coefficient=Monomial().from_sympy(coefficient)).from_sympy(minimal_binomial_sympy)

        if self.simplify_signs:
            expression_string = self._simplify_signs(expression_string)

        return (value, f"({expression_string})", value)

    @staticmethod
    def _simplify_signs(string_expr):
        return (string_expr.replace('--', '+')
                           .replace('+-', '-')
                           .replace('-+', '-')
                           .replace('++', '+'))

    def _build_str_repr(self, tree=None, string_expr=''):
        
        if tree is None:
            tree = self.tree

        if isinstance(tree, list):
            
            for value_idx, value in enumerate(tree):
                
                if value_idx != 0:
                    string_expr += '+'
                    
                if isinstance(value, list):
                    string_expr += self._build_str_repr(value)
                
                else:
                    string_expr += str(value)

            if self.simplify_signs:
                string_expr = self._simplify_signs(string_expr)
            
            return f"({string_expr})"

        else:
            string_expr = str(tree)
            
            if self.simplify_signs:
                string_expr = self._simplify_signs(string_expr)
            
            return string_expr

    def print_resolution(self):
        print(self)
        for step in self.steps:
            print(self._build_str_repr(step[0]))
        
    def __repr__(self):
        assert isinstance(self.tree, Binomial) or isinstance(self.tree, Monomial) or (len(self.tree) > 0), "Cannot print before building expression."
        return self.expression_string


class AlgebraicExpressionGenerator(AbstractGenerator):

    def __init__(self,
                 device,
                 specials_in_x=False,
                 modulo=100,
                 mini_steps=False,
                 variables='xyz',
                 coeff_variables='abc'):
        vocab_chars = string.digits + coeff_variables + variables + '()+-*' + _SEP
        super().__init__(vocab_chars, vocab_chars, device, specials_in_x)
        self.modulo = modulo
        self.mini_steps = mini_steps
        self.variables = variables
        self.coeff_variables = coeff_variables


    def generate_batch(self, batch_size, nesting, num_operands, split='train', exact=False, combiner=False, s2e=False, s2e_baseline=False, simplify=False):
        samples = [self._generate_sample_in_split(nesting, num_operands, split, exact) for _ in range(batch_size)]
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

    def _generate_sample_in_split(self, nesting, num_operands, split, exact):
        simplify_signs = np.random.rand() > .2
        expression = AlgebraicExpression(
            mini_steps=self.mini_steps,
            modulo=self.modulo,
            variables=self.variables,
            coeff_variables=self.coeff_variables,
            simplify_signs=simplify_signs)
        current_split = ''

        while current_split != split:
            expression.build(nesting=nesting, num_operands=num_operands, exact=exact)
            sample_hash = hash(str(expression))

            if sample_hash % 3 == 0:
                current_split = 'train'
            
            elif sample_hash % 3 == 1:
                current_split = 'valid'
            
            else:
                current_split = 'test'
        return expression

    def _build_simplify_w_value(self, samples):
        X_str, Y_str = [], []

        for sample in samples:
            X_str.append(str(sample))

            if '++' in X_str[-1]:
                Y_str.append(f"{_SOS}++{_SEP}+{_EOS}")
            
            elif '+-' in X_str[-1]:
                Y_str.append(f"{_SOS}+-{_SEP}-{_EOS}")

            elif '-+' in X_str[-1]:
                Y_str.append(f"{_SOS}-+{_SEP}-{_EOS}")

            elif '--' in X_str[-1]:
                Y_str.append(f"{_SOS}--{_SEP}+{_EOS}")

            elif isinstance(sample.steps[0][2], str) and (sample.steps[0][2] == _HAL):
                Y_str.append(f"{_SOS}{_HAL}{_EOS}")
            
            else:
                Y_str.append(f"{_SOS}{sample.steps[0][1]}{_SEP}{str(sample.steps[0][2])}{_EOS}")

        return X_str, Y_str

    def _build_s2e_target(self, samples):
        return [str(sample.steps[-1][2]) for sample in samples]

    def _build_s2e_baseline_target(self, samples):
        s2e_target = self._build_s2e_target(samples)
        return [f"{_SOS}{sample}{_EOS}" for sample in s2e_target]

    def _build_combiner_target(self, samples):
        Y_str = []

        for sample in samples:
            sample_str = str(sample)
            
            if '+-' in sample_str:
                Y_str.append(sample_str.replace('+-', '-', 1))

            elif '-+' in sample_str:
                Y_str.append(sample_str.replace('-+', '-', 1))

            elif '--' in sample_str:
                Y_str.append(sample_str.replace('--', '+', 1))

            elif '++' in sample_str:
                Y_str.append(sample_str.replace('++', '+', 1))

            else:
                Y_str.append(sample._build_str_repr(sample.steps[0][0]))

        return Y_str

    def _build_simplify_target(self, samples):
        combiner_target = self._build_combiner_target(samples)
        return [f"{_SOS}{sample}{_EOS}" for sample in combiner_target]
