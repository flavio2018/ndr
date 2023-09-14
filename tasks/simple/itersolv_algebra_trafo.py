from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .itersolv_algebra_mixin import IterSolvAlgebraMixin
from .. import task, args
import framework


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-itersolv_algebra.iid_nesting", default=2)
    parser.add_argument("-itersolv_algebra.iid_num_operands", default=3)
    parser.add_argument("-itersolv_algebra.ood_nesting", default=4)
    parser.add_argument("-itersolv_algebra.ood_num_operands", default=4)


@task()
class ItersolvAlgebraTrafo(IterSolvAlgebraMixin, TransformerClassifierMixin, SimpleTask):
    pass