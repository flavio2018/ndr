import pdb
import copy
import string
import numpy as np
from .AbstractGenerator import AbstractGenerator, _SEP, _SOS, _EOS, _HAL


class ArithmeticExpression:
    
    def __init__(self, min_operand_value=-99, max_operand_value=99, modulo=100, operators='+-*', mini_steps=False, simplify_signs=False):
        self.min_operand_value = min_operand_value
        self.max_operand_value = max_operand_value
        self.operators = operators
        self.steps = []
        self.modulo = modulo
        self.mini_steps = mini_steps
        self.simplify_signs = simplify_signs
    
    def _make_structure(self, current_nesting, exact=False):
        
        if exact:

            if self.max_num_operands == 1:  # case single number, halting condition
                assert self.input_nesting == 1, "Cannot request single op expression of nesting > 1."
                return [self.input_nesting]

            if current_nesting == 1:  # enable multiplication for deeper formulas
                num_operands = np.random.randint(2, self.max_num_operands+1)
            
            else:
                num_operands = self.max_num_operands

            num_nesting_pts = np.random.randint(1, num_operands+1)
            depth_nesting_pts = [current_nesting]*num_nesting_pts
        
        else:

            # case single number, halting condition
            nesting_lvl = np.random.randint(1, current_nesting+1)

            if (current_nesting == self.input_nesting) and (nesting_lvl == 1):
                num_operands = np.random.randint(1, self.max_num_operands+1)
                
                if num_operands == 1:
                    return [current_nesting]

            num_operands = np.random.randint(2, self.max_num_operands+1)
            num_nesting_pts = np.random.randint(1, num_operands+1)
            depth_nesting_pts = np.random.randint(1, current_nesting+1, num_nesting_pts)
        
        nesting_pts = list(set(np.random.choice(range(num_operands), num_nesting_pts, replace=False)))
        nesting_pts_depth = { p: d for (p, d) in zip(nesting_pts, depth_nesting_pts) }
        
        structure = []
        
        for pos in range(num_operands):
            
            if pos in nesting_pts_depth:
                
                if nesting_pts_depth[pos] == 1:
                    structure.append(current_nesting)

                else:
                    structure.append(self._make_structure(current_nesting-1, exact=exact))
            
            else:
                structure.append(current_nesting)
                
        return structure
    
    def _add_operands_and_ops_placeholders(self, structure):
        operands = []
        
        for placeholder in structure:
            
            if isinstance(placeholder, list):
                operands.append(self._add_operands_and_ops_placeholders(placeholder))
                operands.append('?')
            
            else:
                operands.append(np.random.randint(self.min_operand_value, self.max_operand_value))
                operands.append('?')
        
        return operands[:-1]
    
    def _add_operators(self, expression_ops_placeholders):      
        expression = []
        operators_wout_prod = self.operators.replace('*', '')
        expr_has_more_two_ops = len(expression_ops_placeholders) > 3
        
        for pos, operand in enumerate(expression_ops_placeholders):
            
            if isinstance(operand, list):
                expression.append(self._add_operators(operand))
            
            elif operand == '?':
                
                if expr_has_more_two_ops:
                    expression.append(operators_wout_prod[np.random.randint(len(operators_wout_prod))])
                
                else:
                    
                    if np.random.rand() > .35:
                        expression.append('*')
                    
                    else:
                        expression.append(operators_wout_prod[np.random.randint(len(operators_wout_prod))])

            else:
                expression.append(operand)
        
        return expression

    def _compute_steps(self):
        self.steps = []
        expression = copy.deepcopy(self.expression)
        
        reduced_expression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(expression)
        self.steps.append((copy.deepcopy(reduced_expression), subexpression_string, subexpression_value))
        
        while isinstance(reduced_expression, list):
            reduced_expression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(reduced_expression)
            self.steps.append((copy.deepcopy(reduced_expression), subexpression_string, subexpression_value))
        
    def _compute_rightmost_deepmost_step(self, expression=None):

        if expression is None:
            assert self.expression is not None, "Cannot evaluate before building expression"
            expression = copy.deepcopy(self.expression)
        
        expression_types = set([type(v) for v in expression])
        
        if list in expression_types:
            
            for value_pos in range(len(expression)-1, -1, -1):
                value = expression[value_pos]
                
                if isinstance(value, list):
                    new_subexpression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(value)
                    reduced_expression = expression
                    reduced_expression[value_pos] = new_subexpression
                    
                    return (reduced_expression, subexpression_string, subexpression_value)
        
        else:

            if self.mini_steps:
                expression_string = ''.join(str(v) for v in expression[:3])
                value = eval(expression_string)
                value_modulo = 0 if value == 0 else value % (np.sign(value) * self.modulo)
                
                if len(expression) == 1:   # case halting condition

                    if value_modulo == expression[0]:
                        reduced_expression = _HAL
                        value_modulo = _HAL

                        if self.simplify_signs:
                            expression_string = self._simplify_signs(expression_string)

                        return (reduced_expression, expression_string, value_modulo)
                    
                if len(expression) > 3:
                    reduced_expression = [value_modulo] + expression[3:]
                    
                    if self.simplify_signs:
                        expression_string = self._simplify_signs(expression_string)
                    
                    return (reduced_expression, expression_string, value_modulo)
                
                else:
                    reduced_expression = value_modulo

                    if self.simplify_signs:
                        expression_string = self._simplify_signs(expression_string)            
                    
                    return (reduced_expression, f"({expression_string})", value_modulo)
            
            else:
                expression_string = ''.join(str(v) for v in expression)
                value = eval(expression_string)
                value_modulo = 0 if value == 0 else value % (np.sign(value) * self.modulo)

                if self.simplify_signs:
                    expression_string = self._simplify_signs(expression_string)
                
                return (value_modulo, f"({expression_string})", value_modulo)
    
    @staticmethod
    def _simplify_signs(string_expr):
        return (string_expr.replace('--', '+')
                           .replace('+-', '-')
                           .replace('-+', '-'))

    def build(self, nesting, num_operands, exact=False):
        self.max_num_operands = num_operands
        self.input_nesting = nesting
        structure = self._make_structure(nesting, exact=exact)
        expression_ops_placeholders = self._add_operands_and_ops_placeholders(structure)
        self.expression = self._add_operators(expression_ops_placeholders)
        self._compute_steps()
    
    def to_string(self, expression=None):
        string_expr = ''

        if expression is None:
            expression = self.expression
        
        if isinstance(expression, list):

            if len(expression) == 1:
                return str(expression[0])
            
            for value in expression:
                
                if isinstance(value, list):
                    string_expr += self.to_string(value)
                
                else:
                    string_expr += str(value)

        elif isinstance(expression, str) and expression == '$':
            return expression

        else:
            string_expr = str(expression)
            return string_expr

        if self.simplify_signs:
            string_expr = self._simplify_signs(string_expr)
        
        return f"({string_expr})"
    
    def __repr__(self):
        return self.to_string()


class ArithmeticExpressionGenerator(AbstractGenerator):
    
    def __init__(self,
                 device,
                 specials_in_x=False,
                 min_operand_value=-99,
                 max_operand_value=99,
                 modulo=100,
                 operators='+-*',
                 mini_steps=False):     
        vocab_chars = string.digits + '()+*-' + _SEP
        super().__init__(vocab_chars, vocab_chars, device, specials_in_x)
        self.min_operand_value = min_operand_value
        self.max_operand_value = max_operand_value
        self.modulo = modulo
        self.operators = operators
        self.mini_steps = mini_steps
            
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
        expression = ArithmeticExpression(min_operand_value=self.min_operand_value,
                                          max_operand_value=self.max_operand_value,
                                          modulo=self.modulo,
                                          operators=self.operators,
                                          mini_steps=self.mini_steps,
                                          simplify_signs=simplify_signs)
        current_split = ''
        
        while current_split != split:
            expression.build(nesting=nesting, num_operands=num_operands, exact=exact)
            sample_hash = hash(expression.to_string())

            if sample_hash % 3 == 0:
                current_split = 'train'
            
            elif sample_hash % 3 == 1:
                current_split = 'valid'
            
            else:
                current_split = 'test'
        return expression
    
    def _build_simplify_w_value(self, samples):
        X_str = []
        Y_str = []

        for sample in samples:
            X_str.append(sample.to_string())

            if '+-' in X_str[-1]:
                Y_str.append(f"{_SOS}+-{_SEP}-{_EOS}")
            
            elif '-+' in X_str[-1]:
                Y_str.append(f"{_SOS}-+{_SEP}-{_EOS}")
            
            elif '--' in X_str[-1]:
                Y_str.append(f"{_SOS}--{_SEP}+{_EOS}")

            elif sample.steps[0][2] == _HAL:
                Y_str.append(f"{_SOS}{_HAL}{_EOS}")
            
            else:
                Y_str.append(f"{_SOS}{sample.steps[0][1]}{_SEP}{sample.steps[0][2]}{_EOS}")
        
        return X_str, Y_str
    
    def _build_s2e_target(self, samples):
        return [str(sample.steps[-1][2]) for sample in samples]
    
    def _build_s2e_baseline_target(self, samples):
        s2e_target = self._build_s2e_target(samples)
        return [f"{_SOS}{sample}{_EOS}" for sample in s2e_target]

    def _build_combiner_target(self, samples):
        Y_str = []

        for sample in samples:
            sample_str = sample.to_string()
            
            if '+-' in sample_str:
                Y_str.append(sample_str.replace('+-', '-', 1))

            elif '-+' in sample_str:
                Y_str.append(sample_str.replace('-+', '-', 1))

            elif '--' in sample_str:
                Y_str.append(sample_str.replace('--', '+', 1))

            else:
                Y_str.append(sample.to_string(sample.steps[0][0]))

        return Y_str

    def _build_simplify_target(self, samples):
        combiner_target = self._build_combiner_target(samples)
        return [f"{_SOS}{sample}{_EOS}" for sample in combiner_target]
