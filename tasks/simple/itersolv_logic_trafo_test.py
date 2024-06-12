from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .itersolv_logic_test_mixin import IterSolvLogicTestMixin
from .. import task, args
import framework


@task()
class ItersolvLogicTrafoTest(IterSolvLogicTestMixin, TransformerClassifierMixin, SimpleTask):
    pass