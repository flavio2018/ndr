from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .itersolv_arithmetic_mixin import IterSolvArithmeticMixin
from .. import task, args
import framework


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-itersolv_arithmetics.iid_nesting", default=2)
    parser.add_argument("-itersolv_arithmetics.iid_num_operands", default=3)
    parser.add_argument("-itersolv_arithmetics.ood_nesting", default=4)
    parser.add_argument("-itersolv_arithmetics.ood_num_operands", default=4)


@task()
class ItersolvArithmeticTrafo(IterSolvArithmeticMixin, TransformerClassifierMixin, SimpleTask):
    pass