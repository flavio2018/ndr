from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .itersolv_logic_mixin import IterSolvLogicMixin
from .. import task, args
import framework


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-itersolv_logic.iid_nesting", default=2)
    parser.add_argument("-itersolv_logic.iid_num_operands", default=3)
    parser.add_argument("-itersolv_logic.ood_nesting", default=4)
    parser.add_argument("-itersolv_logic.ood_num_operands", default=4)


@task()
class ItersolvListopsTrafo(IterSolvLogicMixin, TransformerClassifierMixin, SimpleTask):
    pass