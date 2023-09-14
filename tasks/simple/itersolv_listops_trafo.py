from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .itersolv_listops_mixin import IterSolvListopsMixin
from .. import task, args
import framework


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-itersolv_listops.iid_nesting", default=2)
    parser.add_argument("-itersolv_listops.iid_num_operands", default=3)
    parser.add_argument("-itersolv_listops.ood_nesting", default=4)
    parser.add_argument("-itersolv_listops.ood_num_operands", default=4)


@task()
class ItersolvListopsTrafo(IterSolvListopsMixin, TransformerClassifierMixin, SimpleTask):
    pass